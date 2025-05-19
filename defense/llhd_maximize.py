import math
import torch
from torch import nn
from torch.autograd.functional import hvp
from functools import partial
from utils import clamp, register, set_seed


@register(name='llhd_maximize', funcs='defenses')
class LikelihoodMaximizedClassifier(nn.Module):

    def __init__(self, diffusion, classifier, config):
        super().__init__()
        self.diffusion = diffusion
        self.classifier = classifier
        self.config = config
        self.__dict__.update({k.lower(): v 
                              for k, v in config.items()})
        self.history = {}

    def update_step(self, T, g, m, v, x, ori_x, backward=False):
        if backward: 
            for i in (m, v, x, ori_x): i.requires_grad_(False)
        with torch.set_grad_enabled(backward):
            m = (self.beta1 * m + (1 - self.beta1) * g) / (1 - self.beta1 ** (T + 1))
            v = (self.beta2 * v + (1 - self.beta2) * g ** 2).detach() / (1 - self.beta2 ** (T + 1))
            x -= self.lr * m / (v ** 0.5 + 1e-8)
            x = clamp(x, ori_x, eps=self.eps * 2, p=2)
        return g, m, v, x

    def diffusion_step(self, x, t, epsilon):
        noised = x + t.view(-1, 1, 1, 1) * epsilon
        denoised = self.diffusion(noised, t)
        if denoised.shape[1] == 6: # imagenet case
            denoised, _ = torch.split(denoised, 3, dim=1)
        return torch.nn.functional.mse_loss(denoised, x)
    
    def scale_t(self, t):
        return t * (self.t_max - self.t_min) + self.t_min

    def purify(self, x, backward=False, seeds=None):
        x = (x - 0.5) * 2
        ori_x = x.clone()
        m = torch.zeros_like(x)
        v = torch.zeros_like(x)
        for T in range(self.n_lm):
            x.requires_grad_(True)
            if isinstance(seeds, int): set_seed(seeds)
            if isinstance(seeds, list):
                assert len(seeds) == x.shape[0]
                t, epsilon = [], []
                for i, seed in enumerate(seeds):
                    set_seed(seed)
                    t.append(self.scale_t(torch.rand_like(x[:, 0, 0, 0])[i]))
                    epsilon.append(torch.randn_like(x)[i])
                t, epsilon = torch.stack(t, 0), torch.stack(epsilon, dim=0)
            else:
                t = self.scale_t(torch.rand_like(x[:, 0, 0, 0]))
                epsilon = torch.randn_like(x)
            loss = self.diffusion_step(x, t, epsilon)
            loss.backward()
            g = x.grad.clone()
            x.grad.zero_()
            if backward:
                self.history[T] = [i.detach().cpu() for i in (g, m, v, x, t, epsilon)]
            g, m, v, x = self.update_step(T, g, m, v, x, ori_x, False)
        x = torch.clamp(x / 2 + 0.5, min=0, max=1)
        return x.detach()

    def forward(self, x):
        x = self.purify(x, False)
        return self.classifier(x)

    def gradient(self, x0, y, loss_fn, grad_mode='full', 
                 seeds=None, aug=None, g0=None):
        x0 = x0.detach()
        if grad_mode == 'classifier':
            x0.requires_grad_(True)
            logits = self.classifier(x0)
            loss_indiv = loss_fn(logits, y)
            loss = loss_indiv.sum()
            loss.backward()
            return x0.grad.clone(), logits, loss_indiv
        x = aug(x0) if aug else x0.clone()
        x = self.purify(x, True, seeds)
        x.requires_grad_(True)
        logits = self.classifier(x)
        loss_indiv = loss_fn(logits, y)
        loss = loss_indiv.sum()
        if g0 is None:
            loss.backward()
            x_grads = [x.grad.clone().detach() / 2]
            x.grad.zero_()
        else: x_grads = [g0.clone() / 2]
        if grad_mode == 'bpda':
            x_grad = x_grads[-1]
            if aug:
                x0.requires_grad_(True)
                (x_grad.detach() * aug(x0, params=aug._params)).sum().backward()
                x_grad = x0.grad.clone()
            return x_grad.detach() * 2, logits, loss_indiv
        for T in range(self.n_lm - 1, -1, -1):
            x_grad = x_grads[-1]
            g, m, v, x_T, t, epsilon = [i.cuda() for i in self.history[T]]
            g.requires_grad_(True)
            r = (x_grad.detach() * self.update_step(T, g, m, v, x_T, x0, True)[-1]).sum()
            r.backward()
            g_grad = g.grad.clamp(-1e7, 1e7).detach()
            x_grads.append(x_grad + hvp(partial(self.diffusion_step, t=t, epsilon=epsilon), 
                                        x_T, g_grad)[1])
        if aug:
            x0.requires_grad_(True)
            (x_grads[-1].detach() * aug(x0, params=aug._params)).sum().backward()
            x_grads[-1] = x0.grad.clone()
            return x_grads[-1].detach() * 2, logits, loss_indiv
        return x_grads[-1].clamp(-1e7, 1e7).detach() * 2, logits, loss_indiv
