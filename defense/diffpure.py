import torch
from torch import nn
from utils import register, set_seed


@register(name='diffpure', funcs='defenses')
class DiffPureClassifier(nn.Module):

    def __init__(self, diffusion, classifier, config):
        super().__init__()
        self.diffusion = diffusion
        self.classifier = classifier
        self.config = config
        self.__dict__.update({k.lower(): v 
                              for k, v in config.items()})
        
        self.history = {'xt': {}, 'eps': {}, 'diff': {}, 'diff_sum': {}}
        self.betas = torch.linspace(1e-4, 2e-2, 1000).cuda()
        self.alphas = (1 - torch.cat([torch.zeros_like(self.betas[:1]), self.betas])).cumprod(0)
        self.eta = 1 if self.sampling_method == 'ddpm' else 0
        self.att_max_timesteps, self.attack_steps = self.get_seq(self.att_max_timesteps, 
                                                                 self.att_denoising_steps)
        self.def_max_timesteps, self.defense_steps = self.get_seq(self.def_max_timesteps, 
                                                                  self.def_denoising_steps)

    @staticmethod
    def get_seq(max_timesteps, denoising_steps):
        max_timesteps = [int(i) - 1 for i in max_timesteps.split(',')]
        denoising_steps = [int(i) for i in denoising_steps.split(',')]
        steps = [[(j + 1) * denoising_steps[k] - 1 
                for j in range((i + 1) // denoising_steps[k])] 
                for k, i in enumerate(max_timesteps)]
        return max_timesteps, steps
    
    def get_coefficient(self, i, j):
        at, at_next = self.alphas[i + 1], self.alphas[j + 1]
        c_eps = self.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c_xt = at_next.sqrt() / at.sqrt()
        c_et = ((1 - at_next) - c_eps ** 2).sqrt() - c_xt * (1 - at).sqrt()
        return c_xt, c_et, c_eps
    
    def GDMP_step(self, i, seq):
        c_eps = self.get_coefficient(i, seq[idx - 1] if (idx := seq.index(i)) else -1)[-1]
        return c_eps * 2e-3 * (1.0 - self.alphas[i + 1]).sqrt() / (self.eps * self.alphas[i + 1].sqrt())

    def denoising_process(self, x, T, backward=False):
        xt = x.clone().detach()
        seq = self.defense_steps[T]
        att_seq = self.attack_steps[T]
        if backward: self.history['xt'][T] = []
        if self.diff_attack: 
            self.history['diff'][T] = []
            self.history['diff_sum'][T] = torch.zeros_like(x)
        for k in reversed(range(len(seq))):
            i, j = seq[k], seq[k - 1] if k else -1
            if backward and i in att_seq: self.history['xt'][T].append(xt.detach().cpu())
            et = self.diffusion(xt, i * torch.ones_like(x[:, 0, 0, 0]))
            if et.shape[1] == 6: # imagenet case
                et, _ = torch.split(et, 3, dim=1)
            eps = torch.randn_like(x)
            c_xt, c_et, c_eps = self.get_coefficient(i, j)
            if self.guided:
                s = self.GDMP_step(i, seq)
                xt = c_xt * xt + c_et * et + c_eps * eps - s * (xt - self.noised(x, i))
            else: xt = c_xt * xt + c_et * et + c_eps * eps
            xt = xt.detach()
            if self.diff_attack and (j in att_seq or j == -1):
                eps = self.history['eps'][T]
                diff = self.alphas[j + 1] * (xt - self.noised(x, j, eps))
                self.history['diff_sum'][T] += self.alphas[j + 1].sqrt() * diff
                self.history['diff'][T].append((diff.detach().cpu()))
        return xt
    
    def noised(self, x, timestep, epsilon=None):
        if epsilon is None: epsilon = torch.rand_like(x)
        x_scaled = x * self.alphas[timestep + 1].sqrt()
        return x_scaled + epsilon * (1.0 - self.alphas[timestep + 1]).sqrt()
    
    def purify(self, x, backward=False, seeds=None):
        x_diff = (x - 0.5) * 2
        for i, v in enumerate(self.def_max_timesteps):
            if isinstance(seeds, int): set_seed(seeds + 1000 * i)
            if isinstance(seeds, list):
                assert len(seeds) == x.shape[0]
                epsilon = []
                for k, seed in enumerate(seeds):
                    set_seed(seed + 1000 * i)
                    epsilon.append(torch.randn_like(x)[k])
                epsilon = torch.stack(epsilon, dim=0)
            else:
                epsilon = torch.randn_like(x)
            x_noised = self.noised(x_diff, v, epsilon)
            if self.diff_attack: 
                self.history['eps'][i] = epsilon.detach()
            x_diff = self.denoising_process(x_noised, i, backward)
        return (x_diff / 2) + 0.5 

    def forward(self, x, backward=False):
        p = self.purify(x, backward).detach().clone()
        logits = self.classifier(p)
        return logits
    
    def gradient(self, x0, y, loss_fn, grad_mode='full', seeds=None, aug=None, g0=None):
        x0 = x0.detach()
        if grad_mode == 'classifier':
            x0.requires_grad_(True)
            logits = self.classifier(x0)
            loss_indiv = loss_fn(logits, y)
            loss = loss_indiv.sum()
            loss.backward()
            return x0.grad.clone(), logits, loss_indiv
        x = aug(x0) if aug else x0.clone()
        x = self.purify(x, True, seeds).detach().clone()
        x.requires_grad_(True)
        logits = self.classifier(x)
        loss_indiv = loss_fn(logits, y)
        loss = loss_indiv.sum()
        if g0 is None: 
            loss.backward()
            x_grad = x.grad.clone().detach() / 2
            x.grad.zero_()
        else: x_grad = g0.clone() / 2
        if grad_mode == 'bpda': 
            if aug:
                x0.requires_grad_(True)
                (x_grad.detach() * aug(x0, params=aug._params)).sum().backward()
                x_grad = x0.grad.clone()
            return x_grad.detach() * 2, logits, loss_indiv
        
        total_steps = sum([len(i) for i in self.attack_steps]) * x.view(y.shape[0], -1).shape[1]
        for T in reversed(range(len(self.att_max_timesteps))):
            xs = [i.cuda() for i in self.history['xt'][T]]
            seq = self.attack_steps[T]
            for k, v in enumerate(seq):
                i, j = seq[k], seq[k - 1] if k else -1
                c_xt, c_et, _ = self.get_coefficient(i, j)
                if self.diff_attack:
                    x_grad += self.history['diff'][T][-1 - k].cuda() / total_steps
                xs[-1 - k].requires_grad_(True)
                t = i * torch.ones_like(x[:, 0, 0, 0]).detach()
                et = self.diffusion(xs[-1 - k], t)
                if et.shape[1] == 6: # imagenet case
                    et, _ = torch.split(et, 3, dim=1)
                r = (x_grad.detach() * et).sum()
                r.backward()
                e_grad = xs[-1 - k].grad.clamp(-1e7, 1e7).detach()
                if self.guided: 
                    x_grad = (c_xt - self.GDMP_step(i, seq)) * x_grad + c_et * e_grad
                else: x_grad = c_xt * x_grad + c_et * e_grad
            x_grad *= self.alphas[self.att_max_timesteps[T] + 1].sqrt()
            if self.guided: 
                x_grad += sum([self.GDMP_step(i, seq) * self.alphas[i + 1].sqrt() for i in seq])
            if self.diff_attack: x_grad -= self.history['diff_sum'][T] / total_steps
        if aug:
            x0.requires_grad_(True)
            (x_grad.detach() * aug(x0, params=aug._params)).sum().backward()
            x_grad = x0.grad.clone()
        return x_grad.clamp(-1e7, 1e7).detach() * 2, logits, loss_indiv

    