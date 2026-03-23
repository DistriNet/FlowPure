import random
import torch
from torch import nn
from utils import gen_seed, register, set_seed
from torchdyn.numerics import odeint
import numpy as np


@register(name='flowpure_stoch', funcs='defenses')
class FlowPureClassifier(nn.Module):

    def __init__(self, flowmodel, classifier, config):
        super().__init__()
        self.flowmodel = flowmodel
        self.classifier = classifier
        self.config = config
        self.def_steps = config.DEF_STEPS
        self.atk_steps = config.ATK_STEPS
        self.t_start = config.T_START
        self.memory_saving = config.MEMORY_SAVING
        self.eta = config.ETA

    def gen_noise(self, x, seeds, offset=0):
        if isinstance(seeds, int): set_seed(gen_seed(seeds, offset))
        elif seeds is not None:
            epsilon = []
            for k, seed in enumerate(seeds):
                set_seed(gen_seed(seed, offset))
                epsilon.append(torch.randn_like(x)[k])
            return torch.stack(epsilon, dim=0)
        return torch.randn_like(x)
    
    def purify(self, x, backward=False, seeds=None):
        x = x * 2 - 1
        xt = self.t_start * x + (1 - self.t_start) * self.gen_noise(x, seeds)
        t_span = torch.linspace(self.t_start, 1.0, self.def_steps + 1, device=x.device)
        dt = (1.0 - self.t_start) / self.def_steps
        
        with torch.no_grad():
            # t1 is current timestep and t2 is timestep at the end of the loop
            for i, (t1, t2) in enumerate(zip(t_span[:-1],t_span[1:]), start=1):
                # go backward in time (inject noise)
                eta = (self.eta * t1 + 1 - self.eta) / t1
                t =  self.eta * t1 + 1 - self.eta
                dt = t2 - t
                xt = eta * xt + torch.sqrt( (1-t1*eta)**2 - eta**2 * (1-t1)**2 ) * self.gen_noise(xt, seeds, i)
                # compute sample at timestep t2 from sample at timestep t = eta * t1
                xt = xt + dt * self.flowmodel(xt, t) 
        xt = (xt + 1) / 2
        return xt

    def forward(self, x, backward=False):
        p = self.purify(x, backward).detach().clone()
        logits = self.classifier(p)
        return logits

    def gradient(self, x, y, loss_fn, grad_mode='full', seeds=None, aug=None, g0=None):
        if aug is not None:
            raise Exception("Augments are not Implemented!")
        x = x.clone().detach().requires_grad_(True)

        if grad_mode == 'classifier':
            logits = self.classifier(x)
            loss_indiv = loss_fn(logits, y)
            loss = loss_indiv.sum()
            loss.backward()
            return x.grad.clone(), logits, loss_indiv
        
        x = x * 2 - 1
        xt = self.t_start * x + (1 - self.t_start) * self.gen_noise(x, seeds)
        t_span = torch.linspace(self.t_start, 1.0, self.def_steps + 1, device=x.device)

        if grad_mode == 'full' and not self.memory_saving:
            # t1 is current timestep and t2 is timestep at the end of the loop
            for i, (t1, t2) in enumerate(zip(t_span[:-1],t_span[1:]), start=1):
                # go backward in time (inject noise)
                eta = (self.eta * t1 + 1 - self.eta) / t1
                t =  self.eta * t1 + 1 - self.eta
                dt = t2 - t
                xt = eta * xt + torch.sqrt( (1-t1*eta)**2 - eta**2 * (1-t1)**2 ) * self.gen_noise(xt, seeds, i)
                # compute sample at timestep t2 from sample at timestep t = eta * t1
                xt = xt + dt * self.flowmodel(xt, t) 
            xt = (xt + 1) / 2
            logits = self.classifier(xt)
            loss_indiv = loss_fn(logits, y)
            if g0 is not None:
                grad = torch.autograd.grad(xt, x, grad_outputs=g0)[0].clone()
            else:
                loss = loss_indiv.sum()
                loss.backward()
                grad = x.grad.clone()
            return grad.detach(), logits, loss_indiv
            

        with torch.no_grad():
            traj = []
            # t1 is current timestep and t2 is timestep at the end of the loop
            for i, (t1, t2) in enumerate(zip(t_span[:-1],t_span[1:]), start=1):
                # go backward in time (inject noise)
                eta = (self.eta * t1 + 1 - self.eta) / t1
                t =  self.eta * t1 + 1 - self.eta
                dt = t2 - t
                xt = eta * xt + torch.sqrt( (1-t1*eta)**2 - eta**2 * (1-t1)**2 ) * self.gen_noise(xt, seeds, i)
                traj.append((xt.detach().clone(), t, dt + (t2-t1)*(self.def_steps/self.atk_steps-1), eta))
                xt = xt + dt * self.flowmodel(xt, t)
            traj.append((xt, t_span[-1]))
        
        traj = [tup for i, tup in enumerate(traj) if i % (self.def_steps/self.atk_steps) == 0]

        if grad_mode == 'bpda':
            x = ((traj[-1][0]+ 1) / 2).clone().detach().requires_grad_(True)
            logits = self.classifier(x)
            loss_indiv = loss_fn(logits, y)
            loss = loss_indiv.sum()
            loss.backward()
            return x.grad.clone(), logits, loss_indiv
        
        x = traj[-1][0].clone().detach().requires_grad_(True)
        x = (x + 1) / 2
        logits = self.classifier(x)
        loss_indiv = loss_fn(logits, y)
        
        if g0 is not None:
            grad = g0
        else:
            loss = loss_indiv.sum()
            loss.backward()
            grad = x.grad.clone()
        
        for xt, t, dt, eta in reversed(traj[:-1]):
            xt = xt.requires_grad_(True)
            f = self.flowmodel(xt, t)
            grad = (grad + dt * torch.autograd.grad(f, xt, grad_outputs=grad)[0]) * eta
            grad = grad.detach().clone()
        
        return grad, logits, loss_indiv