<div align="center">

# FlowPure :ocean:
Official implementation of the paper:

**[FlowPure: Continuous Normalizing Flows for Adversarial Purification](https://arxiv.org/abs/2505.13280v1)**

> **TL;DR**: This work proposes a novel approach to defend against advesarial examples through purification using continuous normalizing flows.

![FlowPure](<resources/figure/diagram.png>)

</div>

## Installation
We recommend setting up the environment with Conda. The codebase currently uses **Python 3.9.20** and **PyTorch 2.0.0**.
```
conda create -n FlowPure python==3.9.20
conda activate FlowPure
pip install -r requirements.txt
```

## Dataset and Checkpoints

### Dataset
- CIFAR: The CIFAR datasets will automatically be downloaded to `./resources/datasets/CIFAR10/` by pytorch

### Checkpoints for Baselines
- Use [this code](https://github.com/bmsookim/wide-resnet.pytorch) to train a WideResNet-28 and save the checkpoint to `./resources/checkpoints/victims/`
- Diffusion (CIFAR10, DiffPure and GDMP): Download [Score SDE](https://github.com/yang-song/score_sde_pytorch) to `./resources/checkpoints/score_sde/`
- Diffusion (CIFAR10, LM): Download [EDM](https://drive.google.com/drive/folders/1mQoH6WbnfItphYKehWVniZmq9iixRn8L?usp=sharing) to `./resources/checkpoints/EDM/`
- Use [this code](https://github.com/LixiaoTHU/ADBM) to train an ADBM and save the checkpoint to `./resources/checkpoints/ADBM/`

## Usage

### Training
Train a Continuous Normalizng Flow model using `trainer_flowpure.py`, specifying the dataset (CIFAR10 or CIFAR100) and noise type (pgd, cw, gauss):
```
python trainer_flowpure.py --dataset [CIFAR10/CIFAR100] --noise_type [pgd/cw/gauss]
```

### Evaluation
To evaluate the baselines and FlowPure, use either `eval_ppb.py` for preprocessor-blind attacks or `eval_DH.py` for white-box DiffHammer attack. The parameters of the defenses and attacks can be adjusted in `config.py`. This evaluation code extends the implementation from [DiffHammer](https://github.com/Ka1b0/DiffHammer).


## Acknowledgement
If you find this work useful, consider giving the repository a star and citing our paper:

```
@misc{collaert2025flowpurecontinuousnormalizingflows,
      title={FlowPure: Continuous Normalizing Flows for Adversarial Purification}, 
      author={Elias Collaert and Abel Rodríguez and Sander Joos and Lieven Desmet and Vera Rimmer},
      year={2025},
      eprint={2505.13280},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.13280}, 
}
```