import re
from fvcore.common.config import CfgNode as _CfgNode

class CfgNode(_CfgNode):

    @classmethod
    def _open_cfg(cls, filename):
        return PathManager.open(filename, "r")

    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)
    
# methods: diffpure, gdmp, llhd_maximize, adbm, gauss_flowpure, pgd_flowpure, cw_flowpure
def get_config(purification_method, dataset, attack_type, seed, batch_size, data_size):
    cfg = CfgNode()
    cfg.DBG = False
    cfg.SEED = seed 
    update_purification(cfg, purification_method)
    update_dataset_and_models(cfg, purification_method, dataset, batch_size, data_size)
    update_attack(cfg, attack_type)
    if purification_method == 'pgd_flowpure' or purification_method == 'cw_flowpure':
        cfg.ATTACK.N_EVAL = 1
    return cfg
    

def update_attack(cfg, attack_type):
    cfg.ATTACK = CfgNode()
    cfg.ATTACK.RESUME = False
    cfg.ATTACK.METHOD = 'apgd'
    cfg.ATTACK.NORM = 'inf'
    cfg.ATTACK.RESTART_THR = 0.5
    cfg.ATTACK.N_EVAL = 10
    cfg.ATTACK.N_EOT = 1
    cfg.ATTACK.EPS = 8 / 255
    cfg.ATTACK.LOSS_NAMES = ['CW', 'CE', 'DLR']
    cfg.ATTACK.GRAD_MODE = 'full'
    cfg.ATTACK.PGD_CMD = ''
    cfg.ATTACK.EM = True 
    cfg.ATTACK.EM_ALPHA = 0.5
    cfg.ATTACK.EM_LAM = 5.0
    cfg.ATTACK.EM_STEPS = 5
    if attack_type == 'DH':
        cfg.ATTACK.METHOD_ACTUAL = 'pgd'
        cfg.ATTACK.N_ITERS = [50,50,50]
        cfg.ATTACK.N_RESTART = 3
        cfg.ATTACK.PGD_STEP_SIZE = 0.007
    elif attack_type == 'class_pgd':
        cfg.ATTACK.METHOD_ACTUAL = 'class_pgd'
        cfg.ATTACK.N_ITERS = [10]
        cfg.ATTACK.N_RESTART = 1
        cfg.ATTACK.PGD_STEP_SIZE = 2/255
    elif attack_type == 'class_cw':
        cfg.ATTACK.METHOD_ACTUAL = 'class_cw'
        cfg.ATTACK.N_ITERS = [50]
        cfg.ATTACK.N_RESTART = 1
    else:
        raise Exception(f"{attack_type} is not implemented!")

# methods: diffpure, gdmp, llhd_maximize, gauss_flowpure, adbm, pgd_flowpure, cw_flowpure
def update_purification(cfg, purification_method):
    cfg.DEFENSE = CfgNode()
    if purification_method == 'diffpure':
        cfg.DEFENSE.METHOD = 'diffpure' 
        cfg.DEFENSE.DIFFPURE = CfgNode()
        cfg.DEFENSE.DIFFPURE.GUIDED = False 
        cfg.DEFENSE.DIFFPURE.SAMPLING_METHOD = 'ddpm'
        cfg.DEFENSE.DIFFPURE.DEF_MAX_TIMESTEPS = '100'
        cfg.DEFENSE.DIFFPURE.DEF_DENOISING_STEPS = '10' 
        cfg.DEFENSE.DIFFPURE.ATT_MAX_TIMESTEPS = '100'
        cfg.DEFENSE.DIFFPURE.ATT_DENOISING_STEPS = '20'
        cfg.DEFENSE.DIFFPURE.DIFF_ATTACK = False
        cfg.DEFENSE.DIFFPURE.EPS = 8 / 255
    elif purification_method == 'gdmp':
        cfg.DEFENSE.METHOD = 'diffpure' 
        cfg.DEFENSE.DIFFPURE = CfgNode()
        cfg.DEFENSE.DIFFPURE.GUIDED = False 
        cfg.DEFENSE.DIFFPURE.SAMPLING_METHOD = 'ddpm'
        cfg.DEFENSE.DIFFPURE.DEF_MAX_TIMESTEPS = "36,36,36,36"
        cfg.DEFENSE.DIFFPURE.DEF_DENOISING_STEPS = "6,6,6,6"
        cfg.DEFENSE.DIFFPURE.ATT_MAX_TIMESTEPS = "36,36,36,36" 
        cfg.DEFENSE.DIFFPURE.ATT_DENOISING_STEPS = "12,12,12,12"
        cfg.DEFENSE.DIFFPURE.DIFF_ATTACK = False
        cfg.DEFENSE.DIFFPURE.EPS = 8 / 255
    elif purification_method == 'llhd_maximize':
        cfg.DEFENSE.METHOD = 'llhd_maximize'
        cfg.DEFENSE.LLHD_MAXIMIZE = CfgNode()
        cfg.DEFENSE.LLHD_MAXIMIZE.N_LM = 5
        cfg.DEFENSE.LLHD_MAXIMIZE.EPS = 100
        cfg.DEFENSE.LLHD_MAXIMIZE.T_MIN = 0.4
        cfg.DEFENSE.LLHD_MAXIMIZE.T_MAX = 0.6
        cfg.DEFENSE.LLHD_MAXIMIZE.LR = 0.1
        cfg.DEFENSE.LLHD_MAXIMIZE.BETA1 = 0.9
        cfg.DEFENSE.LLHD_MAXIMIZE.BETA2 = 0.999
    elif purification_method == 'gauss_flowpure_0.15':
        cfg.DEFENSE.METHOD = 'flowpure'
        cfg.DEFENSE.FLOWPURE = CfgNode()
        cfg.DEFENSE.FLOWPURE.SIGMA = 3
        cfg.DEFENSE.FLOWPURE.T_START = 1/2 # 0 is the full process and 1 is no purification ! 
        cfg.DEFENSE.FLOWPURE.DEF_STEPS = 10 # note that these are amount of steps (other methods use step sizes)
        cfg.DEFENSE.FLOWPURE.ATK_STEPS = 5 # note that these are amount of steps (other methods use step sizes)
    elif purification_method == 'gauss_flowpure_0.2':
        cfg.DEFENSE.METHOD = 'flowpure'
        cfg.DEFENSE.FLOWPURE = CfgNode()
        cfg.DEFENSE.FLOWPURE.SIGMA = 3
        cfg.DEFENSE.FLOWPURE.T_START = 1/3 # 0 is the full process and 1 is no purification ! 
        cfg.DEFENSE.FLOWPURE.DEF_STEPS = 10 # note that these are amount of steps (other methods use step sizes)
        cfg.DEFENSE.FLOWPURE.ATK_STEPS = 5 # note that these are amount of steps (other methods use step sizes)
    elif purification_method == 'cw_flowpure':
        cfg.DEFENSE.METHOD = 'flowpure'
        cfg.DEFENSE.FLOWPURE = CfgNode()
        cfg.DEFENSE.FLOWPURE.SIGMA = 0
        cfg.DEFENSE.FLOWPURE.T_START = 0 # 0 is the full process and 1 is no purification ! 
        cfg.DEFENSE.FLOWPURE.DEF_STEPS = 10 # note that these are amount of steps (other methods use step sizes)
        cfg.DEFENSE.FLOWPURE.ATK_STEPS = 5 # note that these are amount of steps (other methods use step sizes)
    elif purification_method == 'pgd_flowpure':
        cfg.DEFENSE.METHOD = 'flowpure'
        cfg.DEFENSE.FLOWPURE = CfgNode()
        cfg.DEFENSE.FLOWPURE.SIGMA = 0
        cfg.DEFENSE.FLOWPURE.T_START = 0 # 0 is the full process and 1 is no purification ! 
        cfg.DEFENSE.FLOWPURE.DEF_STEPS = 10 # note that these are amount of steps (other methods use step sizes)
        cfg.DEFENSE.FLOWPURE.ATK_STEPS = 5 # note that these are amount of steps (other methods use step sizes)
    elif purification_method == 'adbm':
        cfg.DEFENSE.METHOD = 'diffpure'
        cfg.DEFENSE.DIFFPURE = CfgNode()
        cfg.DEFENSE.DIFFPURE.SAMPLING_METHOD = 'ddim'
        cfg.DEFENSE.DIFFPURE.DEF_MAX_TIMESTEPS = '100'
        cfg.DEFENSE.DIFFPURE.DEF_DENOISING_STEPS = '20'
        cfg.DEFENSE.DIFFPURE.ATT_MAX_TIMESTEPS = '100'
        cfg.DEFENSE.DIFFPURE.ATT_DENOISING_STEPS = '20'
        cfg.DEFENSE.DIFFPURE.DIFF_ATTACK = False
        cfg.DEFENSE.DIFFPURE.GUIDED = False
    else:
        raise Exception(f"{purification_method} is not implemented!")

# methods: diffpure, gdmp, llhd_maximize, gauss_flowpure, adbm, pgd_flowpure, cw_flowpure
def update_dataset_and_models(cfg, purification_method, dataset, batch_size, data_size):
    cfg.DATA = CfgNode()
    cfg.DATA.NAME = dataset
    cfg.DATA.BATCH_SIZE = batch_size
    cfg.DATA.NUM = data_size
    if cfg.DATA.NAME == 'CIFAR10':
        suffix = '_c10'
    else:
        suffix = '_c100'
    cfg.DEFENSE.CLASSIFIER_NAME = 'wrn' + suffix

    if purification_method == 'diffpure':
        cfg.DEFENSE.DIFFUSION_NAME = 'score_sde' + suffix
    elif purification_method == 'gdmp':
        cfg.DEFENSE.DIFFUSION_NAME = 'score_sde' + suffix
    elif purification_method == 'llhd_maximize':
        cfg.DEFENSE.DIFFUSION_NAME = 'edm_unet' + suffix
    elif 'gauss_flowpure' in purification_method:
        cfg.DEFENSE.DIFFUSION_NAME = 'flowpure_gauss' + suffix
    elif purification_method == 'cw_flowpure':
        cfg.DEFENSE.DIFFUSION_NAME = 'flowpure_cw' + suffix
    elif purification_method == 'pgd_flowpure':
        cfg.DEFENSE.DIFFUSION_NAME = 'flowpure_pgd' + suffix
    else:
        cfg.DEFENSE.DIFFUSION_NAME = 'adbm' + suffix