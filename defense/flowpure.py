import random
import torch
from torchdyn.numerics import odeint
from torch import nn
from utils import register, set_seed


@register(name='flowpure', funcs='defenses')
class FlowPureClassifier(nn.Module):

    def __init__(self, flowmodel, classifier, config):
        super().__init__()
        self.flowmodel = flowmodel
        self.classifier = classifier
        self.config = config
        self.sigma = config.SIGMA / 10
        self.def_steps = config.DEF_STEPS
        self.atk_steps = config.ATK_STEPS
        self.t_start = config.T_START

        
    def purify(self, x, backward=False, seeds=None):
        x_noised = x + torch.randn_like(x) * self.sigma
        xt = self.t_start * x + (1-self.t_start) * x_noised
        with torch.no_grad():
            traj = odeint(self.flowmodel, xt, t_span=torch.linspace(self.t_start, 1, self.def_steps, device="cuda"), solver='euler')
            final_traj = traj[-1][-1, :].view([-1, 3, 32, 32])
        return final_traj

    def forward(self, x, backward=False):
        p = self.purify(x, backward).detach().clone()
        logits = self.classifier(p)
        return logits
    
    def gradient(self, x, y, loss_fn, grad_mode='full', seeds=None, aug=None, g0=None):
        x = x.clone().detach().requires_grad_(True)
        x_noised = x + torch.randn_like(x) * self.sigma
        xt = self.t_start * x + (1-self.t_start) * x_noised
        traj = odeint(self.flowmodel, xt, t_span=torch.linspace(self.t_start, 1, self.atk_steps, device=x.device), solver='euler')
        final_traj = traj[-1][-1, :].view([-1, 3, 32, 32])
        logits = self.classifier(final_traj)
        loss = loss_fn(logits, y)
        grad = torch.autograd.grad(loss.mean(), x)[0]
        return grad.detach(), logits, loss
    