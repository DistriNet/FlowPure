import sys
import yaml
import pickle
import argparse
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet50
from models.unets.EDM import get_edm_cifar_uncond
from models.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from models.SmallResolutionModel import WideResNet_70_16_dropout
import score_sde.models.utils as mutils

import warnings
warnings.filterwarnings("ignore")
_DEFENSES = {}
_ATTACKS = {}

class TransClassifier(torch.nn.Module):
    def __init__(self, trans, classifier_model):
        super().__init__()
        self.model = classifier_model
        self.trans = trans

    def forward(self, x):
        return self.model(self.trans(x))
    
class RevDiff(torch.nn.Module):
    def __init__(self, diffusion):
        super().__init__()
        self.model = diffusion

    def forward(self, x, t):
        return self.model(t,x)


def register(cls=None, *, name=None, funcs='defenses'):
    """A decorator for registering model classes."""

    dic = eval('_' + funcs.upper())

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in dic:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        dic[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_log(log, dic, dim=0):
    for k, v in dic.items():
        v = v.detach()
        if k not in log:
            log[k] = v
        else:
            log[k] = torch.cat((log[k], v), dim)
    return log


def save_dict(dic, path):
    with open(path, 'wb') as f:
        pickle.dump(dic, f)


def load_dict(path):
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    return dic


def clamp(x, ori_x, eps, p=2):
    if p == 'inf':
        return torch.clamp(x, ori_x - eps, ori_x + eps)
    else:
        norm = torch.norm(x - ori_x, dim=(1, 2, 3), p=p, keepdim=True)
        rescale = torch.min(eps / norm, torch.ones_like(norm)).detach()
        return rescale * (x - ori_x) + ori_x


def dlr_loss(logits, y):
    logits_sorted, ind_sorted = logits.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(logits.shape[0])

    return -(logits[u, y] - logits_sorted[:, -2] * ind - logits_sorted[:, -1] * (
        1. - ind)) / (logits_sorted[:, -1] - logits_sorted[:, -3] + 1e-3)


def cw_loss(logits, y):
    idx_onehot = torch.nn.functional.one_hot(y, num_classes=logits.size()[1])
    loss = (logits - 1e5 * idx_onehot).max(1)[0] - (logits * idx_onehot).sum(1)
    return loss


def judge_success(logits, y, y_target=None):
    if y_target is None:
        return logits.max(1)[1] != y
    else:
        return logits.max(1)[1] == y_target


def get_model(model_name):
    if model_name == 'adbm_c10':
        diffusion = mutils.create_model(mutils.parse_config('diffusion_configs/cifar10.yml'))
        # diffusion = diffusion.module
        state_dict = torch.load("./resources/checkpoints/ADBM/checkpoint_c10.pth", weights_only=False)['ema']['shadow_params']
        parameters = [p for p in diffusion.parameters() if p.requires_grad]
        for s_param, param in zip(state_dict, parameters):
            param.data.copy_(s_param.data)
        diffusion.eval()
        diffusion.module
        return diffusion
    elif model_name == 'adbm_c100':
        diffusion = mutils.create_model(mutils.parse_config('diffusion_configs/cifar10.yml'))
        state_dict = torch.load("./resources/checkpoints/ADBM/checkpoint_c100.pth", weights_only=False)['ema']['shadow_params']
        parameters = [p for p in diffusion.parameters() if p.requires_grad]
        for s_param, param in zip(state_dict, parameters):
            param.data.copy_(s_param.data)
        diffusion.eval()
        diffusion.module
        return diffusion
    elif model_name == 'flowpure_gauss_c10':
        diffusion = mutils.create_model(mutils.parse_config('diffusion_configs/cifar10.yml'))
        state_dict = torch.load(f"./resources/checkpoints/flowpure_gauss/checkpoint_c10.pth")[0]
        try:
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
            diffusion = diffusion.module
        except:
            diffusion = diffusion.module
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
        rev_diff = RevDiff(diffusion)
        return rev_diff
    elif model_name == 'flowpure_gauss_c100':
        diffusion = mutils.create_model(mutils.parse_config('diffusion_configs/cifar10.yml'))
        state_dict = torch.load(f"./resources/checkpoints/flowpure_gauss/checkpoint_c100.pth")[0]
        try:
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
            diffusion = diffusion.module
        except:
            diffusion = diffusion.module
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
        rev_diff = RevDiff(diffusion)
        return rev_diff
    elif model_name == 'flowpure_pgd_c10':
        diffusion = mutils.create_model(mutils.parse_config('diffusion_configs/cifar10.yml'))
        state_dict = torch.load(f"./resources/checkpoints/flowpure_pgd/checkpoint_c10.pth")[0]
        try:
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
            diffusion = diffusion.module
        except:
            diffusion = diffusion.module
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
        rev_diff = RevDiff(diffusion)
        return rev_diff
    elif model_name == 'flowpure_pgd_c100':
        diffusion = mutils.create_model(mutils.parse_config('diffusion_configs/cifar10.yml'))
        state_dict = torch.load(f"./resources/checkpoints/flowpure_pgd/checkpoint_c100.pth")[0]
        try:
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
            diffusion = diffusion.module
        except:
            diffusion = diffusion.module
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
        rev_diff = RevDiff(diffusion)
        return rev_diff
    elif model_name == 'flowpure_cw_c10':
        diffusion = mutils.create_model(mutils.parse_config('diffusion_configs/cifar10.yml'))
        state_dict = torch.load(f"./resources/checkpoints/flowpure_cw/checkpoint_c10.pth")[0]
        try:
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
            diffusion = diffusion.module
        except:
            diffusion = diffusion.module
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
        rev_diff = RevDiff(diffusion)
        return rev_diff
    elif model_name == 'flowpure_cw_c100':
        diffusion = mutils.create_model(mutils.parse_config('diffusion_configs/cifar10.yml'))
        state_dict = torch.load(f"./resources/checkpoints/flowpure_cw/checkpoint_c100.pth")[0]
        try:
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
            diffusion = diffusion.module
        except:
            diffusion = diffusion.module
            diffusion.load_state_dict(state_dict)
            diffusion.eval()
        rev_diff = RevDiff(diffusion)
        return rev_diff
    elif model_name == 'edm_unet_c10':
        edm_unet = get_edm_cifar_uncond()
        edm_unet.load_state_dict(torch.load(
            "./resources/checkpoints/EDM/edm_cifar_uncond_vp.pt"))
        return edm_unet.cuda().eval()
    elif model_name == 'edm_unet_c100':
        edm_unet = get_edm_cifar_uncond()
        edm_unet.load_state_dict(torch.load(
            "./resources/checkpoints/EDM/edm_cifar100_uncond_vp.pt"))
        return edm_unet.cuda().eval()
    elif model_name == 'score_sde_c10':
        '''
        Follow https://github.com/NVlabs/DiffPure/blob/master/runners/diffpure_sde.py to load model,
        then save `ema` to score_sde_ema.pth to accerate the loading.
        '''
        diffusion = mutils.create_model(mutils.parse_config('diffusion_configs/cifar10.yml'))
        # diffusion = diffusion.module
        state_dict = torch.load('./resources/checkpoints/score_sde/checkpoint_c10.pth', weights_only=False)['model']
        diffusion.load_state_dict(state_dict, strict=False)
        diffusion.eval()
        diffusion.module
        return diffusion
    elif model_name == 'score_sde_c100':
        '''
        Follow https://github.com/NVlabs/DiffPure/blob/master/runners/diffpure_sde.py to load model,
        then save `ema` to score_sde_ema.pth to accerate the loading.
        '''
        diffusion = mutils.create_model(mutils.parse_config('diffusion_configs/cifar10.yml'))
        state_dict = torch.load(
            './resources/checkpoints/score_sde/checkpoint_c100.pth', weights_only=False)[0]
        diffusion.load_state_dict(state_dict, strict=False)
        diffusion.eval()
        diffusion.module
        return diffusion
    elif model_name == 'guided_diffusion':
        with open('./resources/configs/diffusion_configs/imagenet.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            config = dict2namespace(config)
            model_config = model_and_diffusion_defaults()
            model_config.update(vars(config.model))
            diffusion, _ = create_model_and_diffusion(**model_config)
            diffusion.load_state_dict(torch.load(
                './resources/checkpoints/guided_diffusion/256x256_diffusion_uncond.pt', weights_only=False))
            return diffusion.cuda().eval()
    elif model_name == 'imt_resnet50':
        model = resnet50()
        model.fc = torch.nn.Linear(2048, 10)
        model.load_state_dict(torch.load(
            './resources/checkpoints/models/resnet50.pt', map_location='cpu', weights_only=False))
        return model.cuda().eval() 
    elif model_name == 'wrn_c10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
        transform = transforms.Normalize(mean, std)
        model = torch.load("./resources/checkpoints/victims/wide-resnet.t7", weights_only=False)['net']
        model.eval()
        model.to('cuda')
        # model_trans = lambda x: model(transform(x))
        model_trans = TransClassifier(transform, model)
        model_trans.eval()
        return model_trans
    elif model_name == 'wrn_c100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        transform = transforms.Normalize(mean, std)
        model = torch.load("./resources/checkpoints/victims/wide-resnet-c100.t7", weights_only=False)['net']
        model.eval()
        model.to('cuda')
        # model_trans = lambda x: model(transform(x))
        model_trans = TransClassifier(transform, model)
        model_trans.eval()
        return model_trans
    elif model_name == 'new':
        diffusion = mutils.create_model(mutils.parse_config('diffusion_configs/cifar10.yml'))
        diffusion.module
        rev_diff = RevDiff(diffusion)
        return rev_diff


def get_defense(defense_method):
    return _DEFENSES[defense_method]


def get_attacker(attack_method):
    return _ATTACKS[attack_method]


def get_loss_func(loss_name):
    if loss_name == 'CE':
        return torch.nn.CrossEntropyLoss(reduction='none')
    if loss_name == 'DLR':
        return dlr_loss
    if loss_name == 'CW':
        return cw_loss


def get_dataloader(cfg):
    set_seed(cfg.SEED)
    if cfg.DATA.NAME == 'CIFAR10':
        test_loader = get_CIFAR10_test(batch_size=cfg.DATA.BATCH_SIZE)
        test_loader = [item for i, item in 
        enumerate(test_loader) if i < cfg.DATA.NUM // cfg.DATA.BATCH_SIZE]
    if cfg.DATA.NAME == 'CIFAR100':
        test_loader = get_CIFAR100_test(batch_size=cfg.DATA.BATCH_SIZE)
        test_loader = [item for i, item in 
        enumerate(test_loader) if i < cfg.DATA.NUM // cfg.DATA.BATCH_SIZE]
    if cfg.DATA.NAME == 'imagenette':
        transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor(),])
        dataset = datasets.ImageFolder("./resources/datasets/imagenette2-160/val/", 
            transform=transform)
        test_loader = DataLoader(
            dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
        test_loader = [item for i, item in 
            enumerate(test_loader) if i < cfg.DATA.NUM // cfg.DATA.BATCH_SIZE]
    return test_loader


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_CIFAR10_test(batch_size=256, num_workers=8,
    pin_memory=True, transform=transforms.Compose([transforms.ToTensor(),])):
    dataset = datasets.CIFAR10(root='./resources/datasets/CIFAR10/', train=False, 
                               download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    return loader

def get_CIFAR100_test(batch_size=256, num_workers=8,
    pin_memory=True, transform=transforms.Compose([transforms.ToTensor(),])):
    dataset = datasets.CIFAR100(root='./resources/datasets/CIFAR100/', train=False, 
                               download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    return loader

def get_training_set(dataset, batch_size):
    if dataset == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./resources/datasets/CIFAR10/', train=True, 
                                download=True, transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif dataset == 'CIFAR100':
        dataset = datasets.CIFAR100(root='./resources/datasets/CIFAR100/', train=True, 
                                download=True, transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        raise Exception(f"{dataset} is not implemented!")
    return loader
