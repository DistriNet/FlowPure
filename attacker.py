import time
import torch
import torch.nn.functional as F
import kornia.augmentation as K
from collections import defaultdict
from utils import *
from foolbox.attacks import LinfPGD
from torch.utils.data import TensorDataset
import foolbox as fb
import torchattacks as attacks


class Attacker:

    def __init__(self, model, config, logger, seeder):
        self.model = model
        self.config = config.ATTACK
        self.logger = logger
        self.seeder = seeder
        self.__dict__.update({k.lower(): v 
                              for k, v in self.config.items()})
        self.norm = 'inf' if self.norm == 'inf' else int(self.norm)
        self.B, self.N = config.DATA.BATCH_SIZE, config.DATA.NUM
        self.n_batch = self.N // self.B
        self.path = f'{config.OUTPUT_DIR}/{config.EXP or config.DEFENSE.METHOD}/{config.NAME}/'

    def initialize(self, x):
        t = torch.rand_like(x)
        if self.norm == 'inf':
            x_adv = self.eps * (2 * t - 1) + x
        else:
            x_adv = x + (t - 0.5) * 2 * self.eps
        return x_adv.detach().clamp(0., 1.)

    def worst(self, x):
        return x.max(0)[0].sum().item()

    def avg(self, x):
        x = x.view(-1, self.n_eval, x.shape[-1])
        return x.sum(1).max(0)[0].sum().item() / self.n_eval
    
    def unit(self, x, dim=-1):
        if self.norm == 'inf':
            return x.sign()
        else:
            return x / x.norm(dim=dim, p=self.norm, keepdim=True)
    
    @property
    def batch_str(self):
        return f'Batch.{self.batch + 1:02d}/{self.n_batch:02d}'
    
    @property
    def restart_str(self):
        return f'rs.{self.restart + 1:02d}/{self.n_restart:02d}'
    
    @property
    def iter_str(self):
        return f'it.{self.iter + 1:03d}/{self.n_iter:03d}'

    @staticmethod
    def log_metrics(log, logits, loss, success, prefix=''):
        return update_log(log, {prefix + 'logits': logits.unsqueeze(0),    # 1, B, -1
                                prefix + 'loss': loss.unsqueeze(0),        # 1, B
                                prefix + 'success': success.unsqueeze(0)}) # 1, B
    
    def seeds_select(self, grads, loss):
        eval_grads_a = grads.unsqueeze(0) # 1, n_eval, B, d 
        eval_grads_b = torch.transpose(eval_grads_a, 0, 1) # n_eval, 1, B, d
        delta = (self.unit(eval_grads_a) * eval_grads_b).sum(-1) # n_eval, n_eval, B, d -> n_eval, n_eval, B
        expected_loss = F.elu(loss.view(-1, 1, self.B) + self.eps * delta).sum(0)
        idxs = expected_loss.topk(k=self.n_eot, dim=0)[1]
        return idxs
    
    def EM(self, eval_dict):
        # Extension to n grads case, when self.n_eot=1 is the default version
        loss, seeds_list, grads = eval_dict['loss'], eval_dict['seeds'], eval_dict['grads']
        seeds = torch.tensor(seeds_list).to(device=loss.device)
        idxs = self.seeds_select(grads, loss)
        r = grads[idxs[0], torch.arange(self.B), :].unsqueeze(0) # 1, 1, B, d 
        for _ in range(self.em_steps):
            expected_loss = loss.view(-1, self.B) + (self.unit(r) * grads).sum(-1)
            w = F.sigmoid(self.em_lam * expected_loss).view(-1, self.B, 1) # n_eval, B, 1
            r0 = (w * grads).sum(0, keepdim=True) / w.sum(0, keepdim=True)
        r_grads = grads[idxs, torch.arange(self.B), :].unsqueeze(0) # 1, K, B, d
        sub_weight = F.cosine_similarity(grads.unsqueeze(1), r_grads, dim=-1) # n_eval, K, B
        sub_weight = F.softmax(sub_weight, dim=1).transpose(0, 1).unsqueeze(-1) # K, n_eval, B, 1
        sub_weight = self.n_eot * sub_weight * w.unsqueeze(0) # n_eval, B, K
        r = (sub_weight * grads).sum(1) / sub_weight.sum(1) # K, B, d 
        self.em_grad = r
        self.attack_seeds = [seeds[i].tolist() for i in idxs]

    def evaluate_accuracy(self, test_loader):
        start_time = time.time()
        success = []
        for i, (x, y) in enumerate(test_loader):
            x, y = x.cuda(), y.cuda()
            for _ in range(self.n_eval):
                eval_seed = next(self.seeder)
                set_seed(eval_seed)
                logits = self.model(x)
                success.append(judge_success(logits, y))
            # print(1-torch.cat(success).sum().item() / (i+1) / self.B / self.n_eval)
        success = torch.cat(success).view(-1, self.n_eval, self.B).transpose(1, 2)
        acc_avg = 1 - (asr_avg := success.sum() / self.n_eval / self.N)
        acc_wor = 1 - (asr_wor := success.any(-1).sum() / self.N)
        self.eval_time = time.time() - start_time
        self.logger.info(f'Clean Evaluation Acc Avg: {acc_avg:.2%} | Acc Wor: {acc_wor:.2%} ' \
                         f'with time {int(self.eval_time // 60)}m{self.eval_time % 60:.2f}s')
        return {'clean_acc_avg': acc_avg, 'clean_acc_wor': acc_wor,
                'clean_asr_avg': asr_avg, 'clean_asr_wor': asr_wor}
    
    def evaluate_time(self, test_loader):
        # start_time = time.time()
        times = np.array([])
        for i, (x, y) in enumerate(test_loader):
            x, y = x.cuda(), y.cuda()

            start_time = time.time()
            logits = self.model(x)
            end_time = time.time()
            if i > 1:
                times = np.append(times, [end_time-start_time])
                self.logger.info(f'Avg. time: {np.mean(times)}, Std. time: {np.std(times)}')
    
    def reload_data(self, data):
        status = [(i, self.data_log['adv_asr_wor'][i], self.data_log['adv_loss_avg'][i]) 
                  for i in range(self.N) if self.data_log['adv_asr_avg'][i] <= self.restart_thr]
        status = sorted(status, key=lambda x: x[1] * 1e2 - x[2])
        if len(status) % self.B: status += status[:self.B - len(status) % self.B]
        indexs = [i[0] for i in status]
        reload_dataset = TensorDataset(data['x'][indexs], data['y'][indexs])
        reload_dataloader = DataLoader(reload_dataset, batch_size=self.B, num_workers=8, pin_memory=True)
        return reload_dataloader, indexs

    def update_batch_metrics(self, success, loss, idxs):
        metrics = {}
        bests = []
        log = self.data_log
        for k, v in (('loss', loss), ('asr', success)):
            v = v.view(-1, self.n_eval, self.B)
            wor = v.to(torch.int8).max(0)[0].max(0)[0].tolist()
            avg = (v.sum(1).max(0)[0] / self.n_eval).tolist()
            metrics[f'adv_{k}_avg'], metrics[f'adv_{k}_wor'] = avg, wor
        for i, idx in enumerate(idxs):
            for k, v in metrics.items():
                if ('loss' in k and log[k][idx] == 0) or v[i] > log[k][idx]:
                    log[k]['sum'] += v[i] - log[k][idx]
                    log[k][idx] = v[i]
                    if k == 'adv_asr_wor' or \
                        (k == 'adv_loss_avg' and wor[i] >= log['adv_asr_wor'][idx]):
                        bests.append((i, idx))
        return bests
    
    @property
    def rob_metrics(self):
        num = min(max(self.batch * self.B + self.restart * self.N, 1), self.N)
        dic = {k: self.data_log[k]['sum'] / num 
               for k in ('adv_loss_avg', 'adv_loss_wor', 'adv_asr_avg', 'adv_asr_wor')}
        return {**dic, **{k.replace('asr', 'acc'): 1 - dic[k] for k in ('adv_asr_avg', 'adv_asr_wor')}}

    def evaluate_robustness(self, test_loader, y_target=None):
        if self.resume:
            data = torch.load(self.path + 'data.pt')
            self.data_log = load_dict(self.path + 'data_log.pkl')
            resume_rs, resume_batch = data['rs'], data['batch']
        else:
            data = {rs: {} for rs in range(self.n_restart)}
            self.data_log = {}
            for key in ('adv_loss_avg', 'adv_loss_wor', 'adv_asr_avg', 'adv_asr_wor'):
                self.data_log[key] = defaultdict(int)

        start_time = time.time()
        
        for rs in range(self.n_restart):

            if self.resume and rs < resume_rs: continue

            data['rs'] = self.restart = rs
            if rs: loader, indexs = self.reload_data(data)
            else: loader, indexs = test_loader, list(range(self.N))
            self.n_iter = self.n_iters[rs]
            self.loss = get_loss_func(self.loss_names[rs])
            self.n_batch = len(loader)
            data[rs]['indexs'] = indexs

            total = 0

            for i, (x, y) in enumerate(loader):

                
                total += x.size(0)
                # print(total)
                if self.resume and i < resume_batch: continue

                x, y = x.cuda(), y.cuda()
                idxs = indexs[i * self.B: (i+1) * self.B]
                log = {}

                data['batch'] = self.batch = i
                self.iter = -1
                self.x_adv = self.initialize(x)
                self.metric_str = f'[A{self.rob_metrics["adv_asr_avg"]:7.2%}|' \
                                  f'W{self.rob_metrics["adv_asr_wor"]:7.2%}]'
                self.eval_dict = defaultdict(list)
                try:
                    torch.save(data, self.path + 'data.pt')
                    save_dict(self.data_log, self.path + 'data_log.pkl')
                except BaseException:
                    pass
                if x.size(0) < self.B: # if batch is smaller due to some bug, the code fails...
                    break
                for j in range(self.n_eval):
                    eval_seed = next(self.seeder)
                    set_seed(eval_seed)
                    self.eval_dict['seeds'].append(eval_seed)
                    if self.em:
                        g, logits, loss = self.model.gradient(x, y, self.loss, 'bpda', eval_seed)
                        self.eval_dict['grads'].append(g.view(self.B, -1))
                        self.eval_dict['loss'].append(loss)
                    else:
                        logits = self.model(x)
                        loss = self.loss(logits, y)
                    success = judge_success(logits, y, y_target)
                    log = self.log_metrics(log, logits, loss, success)
                    
                if self.em: 
                    for eval_key in ('grads', 'loss'): 
                        self.eval_dict[eval_key] = torch.stack(self.eval_dict[eval_key])
                    self.EM(self.eval_dict)

                x_adv, log = self.perturb(x, y, log, idxs, y_target)
                # print(log)
                self.update_batch_metrics(log['success'], log['loss'], idxs)
                if 'x' not in data or data['x'].shape[0] < self.N: 
                    data = update_log(data, {'x': x.cpu(), 'y': y.cpu(), 'x_adv': x_adv.cpu()})
                else:
                    for k, v in enumerate(idxs): data['x_adv'][v] = x_adv[k].cpu()
                data[rs] = update_log(data[rs], {k: v.cpu() for k, v in log.items()}, dim=1)
        self.batch += 1
        self.eval_time = time.time() - start_time
        self.logger.info(f'Adversarial Evaluation Rob Avg: {self.rob_metrics["adv_asr_avg"]:.2%} | ' \
                         f'Rob Wor: {self.rob_metrics["adv_asr_wor"]:.2%} with time ' \
                         f'{int(self.eval_time // 60)}m{self.eval_time % 60:.2f}s')
        return data, self.rob_metrics
    
    def init_params(self):
        self.params = {}
    
    def update_adv(self, x, grad):
        pass

    def update_params(self, log):
        pass
    
    def perturb(self, x, y, log, idxs, y_target=None):

        if self.config.METHOD_ACTUAL == 'class_pgd':
            fmodel = fb.PyTorchModel(self.model.classifier, bounds=(0., 1.))
            atk_fb = LinfPGD(abs_stepsize=self.config.PGD_STEP_SIZE, steps=self.n_iter)
            atk = lambda samples, labels: atk_fb(fmodel, samples.detach(), labels, epsilons=self.config.EPS)[1]
            adv_sample = atk(x, y)
            for j in range(self.n_eval):
                logits = self.model(adv_sample)
                loss = self.loss(logits, y)
                success = judge_success(logits, y, y_target)
                log = self.log_metrics(log, logits, loss, success)
            self.x_adv_best = adv_sample
            self.x_adv = adv_sample
            self.logger.info(f"{self.metric_str} {self.batch_str} " \
                             f"{self.restart_str} {self.iter_str}: " \
                             f"adv loss avg. {self.avg(log['loss']):.2f} " \
                             f"wor. {self.worst(log['loss']):.2f}, " \
                             f"adv asr avg. {self.avg(log['success']):.2f} " \
                             f"wor. {self.worst(log['success']):d} in {self.B:d}.")
            # logits = self.model.classifier(adv_sample)
            # print(judge_success(logits, y, y_target).sum())
            # logits = self.model.classifier(x)
            # print(judge_success(logits, y, y_target).sum())
            return adv_sample, log
        if self.config.METHOD_ACTUAL == 'class_cw':
            fmodel = fb.PyTorchModel(self.model.classifier, bounds=(0., 1.))
            atk_fb = fb.attacks.carlini_wagner.L2CarliniWagnerAttack(steps=50)
            atk = lambda samples, labels: atk_fb(fmodel, samples.detach(), labels, epsilons=3)[1]
            adv_sample = atk(x, y)
            for j in range(self.n_eval):
                logits = self.model(adv_sample)
                loss = self.loss(logits, y)
                success = judge_success(logits, y, y_target)
                log = self.log_metrics(log, logits, loss, success)
            self.x_adv_best = adv_sample
            self.x_adv = adv_sample
            self.logger.info(f"{self.metric_str} {self.batch_str} " \
                             f"{self.restart_str} {self.iter_str}: " \
                             f"adv loss avg. {self.avg(log['loss']):.2f} " \
                             f"wor. {self.worst(log['loss']):.2f}, " \
                             f"adv asr avg. {self.avg(log['success']):.2f} " \
                             f"wor. {self.worst(log['success']):d} in {self.B:d}.")
            # logits = self.model.classifier(adv_sample)
            # print(judge_success(logits, y, y_target).sum())
            # logits = self.model.classifier(x)
            # print(judge_success(logits, y, y_target).sum())
            return adv_sample, log

        self.x_adv_best = self.x_adv.clone()
        self.init_params()

        for i in range(self.n_iter):
            self.iter = i
            if torch.all((s := log['success'])[-self.n_eval:]):
                skip = self.n_iter - s.shape[0] // self.n_eval + 1
                for k, v in log.items():
                    log[k] = torch.cat((log[k], v[-self.n_eval:].repeat_interleave(skip, 0)))
                self.logger.info(f'Skip {skip} iters.')
                break
            if not self.em:
                self.attack_seeds = [next(self.seeder) for _ in range(self.n_eot)]
            grad = torch.zeros_like(x)
            for j in range(self.n_eot):
                g, logits, loss = self.model.gradient(self.x_adv, y, self.loss, 
                                                      self.grad_mode, self.attack_seeds[j], 
                                                      self.aug if 'aug' in self.__dict__ else None,
                                                      self.em_grad[j].view(*x.shape))
                grad += g
            grad /= float(self.n_eot)
            if self.em and self.iter: 
                v = (self.iter + 1) ** -self.em_alpha
                grad = v * grad + (1 - v) * past_grad
            self.update_adv(x, grad, y)
            self.eval_dict = defaultdict(list)
            for j in range(self.n_eval):
                eval_seed = next(self.seeder)
                self.eval_dict['seeds'].append(eval_seed)
                set_seed(eval_seed)
                if self.em:
                    g, logits, loss = self.model.gradient(self.x_adv, y, self.loss, 'bpda', eval_seed)
                    self.eval_dict['grads'].append(g.view(self.B, -1))
                    self.eval_dict['loss'].append(loss)
                else:
                    logits = self.model(self.x_adv)
                    loss = self.loss(logits, y)
                success = judge_success(logits, y, y_target)
                log = self.log_metrics(log, logits, loss, success)
                self.logger.debug(f"{self.metric_str} {self.batch_str} {self.restart_str} {self.iter_str}: " \
                                f"evaluating adv sample {j+1:02d}/{self.n_eval:02d} with adv loss {loss.sum().item():.2f} "\
                                f"and adv asr {success.sum().item():d}/{self.B:d} (seed:{eval_seed:d}).")
                
            bests = self.update_batch_metrics(log['success'], log['loss'], idxs)
            for idx, _ in bests: self.x_adv_best[idx] = self.x_adv[idx]

            if self.em:
                for eval_key in ('grads', 'loss'): 
                    self.eval_dict[eval_key] = torch.stack(self.eval_dict[eval_key])
                self.EM(self.eval_dict)
                past_grad = grad.clone()
            self.update_params(log)
            self.logger.info(f"{self.metric_str} {self.batch_str} " \
                             f"{self.restart_str} {self.iter_str}: " \
                             f"adv loss avg. {self.avg(log['loss']):.2f} " \
                             f"wor. {self.worst(log['loss']):.2f}, " \
                             f"adv asr avg. {self.avg(log['success']):.2f} " \
                             f"wor. {self.worst(log['success']):d} in {self.B:d}.")
        
        return self.x_adv_best, log
        

@register(name='apgd', funcs='attacks')
class APGD(Attacker):

    def __init__(self, model, config, logger, seeder):
        super().__init__(model, config, logger, seeder)


    def init_params(self):
        self.check_list = set(int(i * self.n_iter) 
                              for i in [0, 0.22, 0.41, 0.57, 0.7, 0.8, 0.87, 0.93])
        self.check_list = sorted(list(self.check_list))
        params = {}
        params['x_adv_last'] = self.x_adv.clone()
        params['step_size'] = 2. * self.eps * torch.ones_like(self.x_adv[:, :1, :1, :1])
        params['step_size_last'] = params['step_size'].clone()
        params['rho'] = 0.75
        self.params = params

    def update_adv(self, x, grad, y=None):
        x_adv = self.x_adv
        step_size, x_adv_last = self.params['step_size'], self.params['x_adv_last']
        if self.norm == 'inf':
            r = step_size * grad.sign()
        else:
            r = step_size * grad / grad.norm(dim=(1, 2, 3), p=self.norm, keepdim=True)
        z = clamp(x_adv + r, x, self.eps, self.norm)
        a = 0.75 if self.iter else 1.0
        x_adv = x_adv + a * (z - x_adv) + (1 - a) * (x_adv - x_adv_last)
        x_adv = clamp(x_adv, x, self.eps, self.norm)
        self.x_adv = torch.clamp(x_adv, 0, 1)
    
    def update_params(self, log):
        i = self.iter
        rho = self.params['rho']
        step_size, step_size_last = self.params['step_size'], self.params['step_size_last']
        if i + 1 in self.check_list:
            last = self.check_list[self.check_list.index(i + 1) - 1]
            loss_best = log['loss'].view(-1, self.n_eval, self.B).max(1)[0]
            
            mask1 = (loss_best[last + 1: i + 2] > loss_best[last: i + 1]).sum(0) < rho * (i + 1 - last)
            mask2 = (step_size == step_size_last).view(-1) & (loss_best[i + 1] == loss_best[last])
            mask = mask1 | mask2
            self.params['step_size_last'] = step_size.clone()
            self.params['step_size'] = torch.where(mask.view(-1, 1, 1, 1), step_size / 2., step_size).detach()
            self.x_adv = self.x_adv_best.clone()
            self.logger.debug(f"{self.metric_str} {self.batch_str} {self.restart_str}: "\
                              f"{mask.sum().item():d}/{self.B:d} step sizes are halved in iter {i+1:d}.")


@register(name='pgd', funcs='attacks')
class PGD(Attacker):

    def __init__(self, model, config, logger, seeder):
        super().__init__(model, config, logger, seeder)
        kerbel_size = 3
        if 'D' in self.pgd_cmd:
            self.aug = K.RandomAffine(degrees=0, scale=(0.9, 1.1), p=0.5)
        self.blur = K.RandomGaussianBlur((kerbel_size, kerbel_size), 
                                         (1 / kerbel_size ** 0.5, 1 / kerbel_size ** 0.5), p=1.)
        
    def init_params(self):
        params = {}
        params['step_size'] = self.pgd_step_size if self.pgd_step_size else 1.2 * self.eps / self.n_iter
        params['grad_last'] = torch.zeros_like(self.x_adv)
        self.params = params
    
    def update_adv(self, x, grad, y=None):
        step_size, grad_last = self.params['step_size'], self.params['grad_last']
        if 'T' in self.pgd_cmd: grad = self.blur(grad)
        if 'M' in self.pgd_cmd: grad = grad_last + grad / grad.norm(1)
        self.params['grad_last'] = grad
        if self.norm == 'inf':
            r = step_size * grad.sign()
        else:
            r = step_size * grad / grad.norm(dim=(1, 2, 3), p=self.norm, keepdim=True)
        x_adv = clamp(self.x_adv + r, x, self.eps, self.norm)
        self.x_adv = torch.clamp(x_adv, 0, 1)


@register(name='vmi', funcs='attacks')
class VMI(Attacker):

    def init_params(self):
        params = {}
        params['grad_last'] = torch.zeros_like(self.x_adv)
        params['v'] = torch.zeros_like(self.x_adv)
        self.params = params

    def update_adv(self, x, grad, y=None):
        x_adv = self.x_adv
        step_size = 2 * self.eps / self.n_iter
        v, grad_last = self.params['v'], self.params['grad_last']
        self.params['grad_last'] = grad_last + (grad + v) / (grad + v).norm(1)
        grads_mean = self.eval_dict['grads'].mean(0).view(*x.shape)
        self.params['v'] = grads_mean - self.em_grad.view(*x.shape)
        if self.norm == 'inf':
            r = step_size * grad.sign()
        else:
            r = step_size * grad / grad.norm(dim=(1, 2, 3), p=self.norm, keepdim=True)
        x_adv = clamp(x_adv + r, x, self.eps, self.norm)
        self.x_adv = torch.clamp(x_adv, 0, 1)
        

@register(name='svre', funcs='attacks')
class SVRE(Attacker):

    def init_params(self):
        params = {}
        params['G'] = torch.zeros_like(self.x_adv)
        self.params = params

    def update_adv(self, x, grad, y=None):
        x_adv = self.x_adv
        step_size = 1.6 * self.eps / self.n_iter
        seeds = self.eval_dict['seeds']
        grads = self.eval_dict['grads'].view(-1, *x.shape)
        G_m = torch.zeros_like(x)
        z = x_adv.clone()
        for i in range(self.n_eval):
            g, _, _ = self.model.gradient(z, y, self.loss, 'bpda', seeds[i])
            g_m = g - grads[i] + grad
            G_m = 1.0 * G_m + g_m
            z = clamp(z + step_size * G_m.sign(), x, self.eps, self.norm)
            z = torch.clamp(z, 0, 1)
        self.params['G'] = 1.0 * self.params['G'] + G_m
        x_adv = clamp(x_adv + step_size * self.params['G'].sign(), 
                      x, self.eps, self.norm)
        self.x_adv = torch.clamp(x_adv, 0, 1)
