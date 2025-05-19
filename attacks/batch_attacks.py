import math
import torch
import torch.nn as nn
from torchattacks.attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels, eps=None, alpha=None, steps=None):
        r"""
        Overridden.
        """
        if eps is None:
            eps = self.eps
        if alpha is None:
            alpha = self.alpha
        if steps is None:
            steps = self.steps

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            # adv_images = adv_images + torch.empty_like(adv_images).uniform_(
            #     -eps, eps
            # )
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -1, 1
            ) * eps
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        if type(steps) == int:
            for _ in range(steps):
                adv_images.requires_grad = True
                outputs = self.get_logits(adv_images)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

                # Update adversarial images
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                adv_images = adv_images.detach() + alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-eps, max=eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        else:
            for step in range(int(torch.max(steps).item())):
                adv_images.requires_grad = True
                outputs = self.get_logits(adv_images)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

                # Update adversarial images
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
                adv_images = adv_images.detach() + (steps >= step) * alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-eps, max=eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
        
class batched_CW(Attack):

    def __init__(self, model):
        super().__init__("CW", model)
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels, c=1, kappa=0, steps=50, lr=0.01):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        # optimizer = torch.optim.Adam([w], lr=lr, capturable=True)
        optimizer = AdamPerSampleLR([w])
        
        max_steps = int(torch.max(steps).item())
        for step in range(max_steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.get_logits(adv_images)
            if self.targeted:
                f_loss = self.f(outputs, target_labels, kappa).sum()
            else:
                f_loss = self.f(outputs, labels, kappa).sum()

            cost = L2_loss + torch.sum(c * f_loss)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step(lr)

            # Update adversarial images
            # pre = torch.argmax(outputs.detach(), 1)
            # if self.targeted:
            #     # We want to let pre == target_labels in a targeted attack
            #     condition = (pre == target_labels).float()
            # else:
            #     # If the attack is not targeted we simply make these two values unequal
            #     condition = (pre != labels).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            # mask = condition * (best_L2 > current_L2.detach())
            # best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            # mask = mask.view([-1] + [1] * (dim - 1))
            # best_adv_images = (steps >= step) * (mask * adv_images.detach() + (1 - mask) * best_adv_images) + (steps < step) * best_adv_images
            best_adv_images = (steps >= step) * adv_images.detach() + (steps < step) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            # if step % max(max_steps // 10, 1) == 0:
            #     if cost.item() > prev_cost:
            #         return best_adv_images
            #     prev_cost = cost.item()

        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels, kappa):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels * outputs, dim=1)[0]

        if self.targeted:
            return torch.clamp((other - real), min=-kappa)
        else:
            return torch.clamp((real - other), min=-kappa)
        

from torch.optim.optimizer import Optimizer

class AdamPerSampleLR(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamPerSampleLR, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, per_sample_lr=None, closure=None):
        """Performs a single optimization step."""
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                
                # Ensure per_sample_lr is provided and matches batch size
                if per_sample_lr is not None:
                    if per_sample_lr.shape[0] != grad.shape[0]:
                        raise ValueError("per_sample_lr must have the same batch size as the gradients")
                else:
                    per_sample_lr = torch.full((grad.shape[0],), group['lr'], device=grad.device)
                
                state = self.state[p]
                
                # Initialize state variables if not already present
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Compute biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected estimates
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = per_sample_lr.view(-1, 1, 1, 1).expand_as(p) / bias_correction1
                
                # Apply weight decay if set
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update parameters
                with torch.no_grad():
                    p.add_(-step_size * exp_avg / denom)
        