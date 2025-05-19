<div align="center">

# DiffHammer :hammer:
Official implementation of the NeurIPS 2024 paper:

**[DiffHammer: Rethinking the Robustness of Diffusion-Based Adversarial Purification](https://neurips.cc/virtual/2024/poster/94646)**

> **TL;DR**: An adaptive attack on diffusion-based purification that avoids gradient dilemmas by using selective attacks and enhances efficiency through gradient grafting.

![DiffHammer](<resources/figure/plot.jpg>)

</div>

## Installation
We recommend setting up the environment with conda. The codebase currently uses Python 3.9.20 and PyTorch 2.0.0.
```
git clone https://github.com/Ka1b0/DiffHammer.git
cd DiffHammer

# create env using conda
conda create -n DH python==3.9.20
conda activate DH
pip install -r requirements.txt
```

## Dataset and Checkpoints

### Dataset
- CIFAR10: Download CIFAR10 dataset to `./resources/datasets/CIFAR10/`
- ImageNette: Download [ImageNette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz) (160px) and unzip to `./resources/datasets/ImageNette/`

### Checkpoints
- Diffusion (CIFAR10, DiffPure and GDMP): Download [Score SDE](https://github.com/yang-song/score_sde_pytorch) to `./resources/checkpoints/score_sde/`
- Diffusion (CIFAR10, LM): Download [EDM](https://drive.google.com/drive/folders/1mQoH6WbnfItphYKehWVniZmq9iixRn8L?usp=sharing) to `./resources/checkpoints/EDM/`
- Diffusion (ImageNette): Download [Guided diffusion](https://drive.google.com/file/d/1zfblaZd64Aye-FMpus85U5PLrpWKusO4/view) to `./resources/checkpoints/guided_diffusion/`
- Classifier (CIFAR10): Download [WRN-70-16-Dropout](https://drive.google.com/drive/folders/1mQoH6WbnfItphYKehWVniZmq9iixRn8L?usp=sharing) to `./resources/checkpoints/models/`
- Classifier (ImageNette): Download [ResNet50](https://drive.google.com/file/d/1aG3tx0f32EGWNrxH4yAAESx4PsTvVHaD/view?usp=sharing) to `./resources/checkpoints/models/`

## Usage

The configuration settings can be found in the `config.py` file. You can customize your configurations either by directly modifying the `config.py` file or by specifying parameters through command-line arguments.

To execute the program with the default settings, use the following command:

```
python main.py
```

To run an evaluation on Diffpure, execute:

```
python main.py DEFENSE.METHOD diffpure
```

For evaluation on GDMP, run:

```
python main.py DEFENSE.METHOD diffpure DEFENSE.DIFFPURE.GUIDED True
```

For evaluation on LM, run:

```
python main.py DEFENSE.METHOD llhd_maximize
```

To perform the DiffHammer attack, use the command:

```
python main.py ATTACK.EM True
```

Set the following parameter for different attacks:
- DiffAttack: `DEFENSE.DIFFPURE.DIFF_ATTACK True`
- PGD: `ATTACK.METHOD pgd ATTACK.PGD_CMD M` (`ATTACK.PGD_CMD` can be any combination of 'M', 'D', 'T' or '')
- BPDA: `ATTACK.GRAD_MODE bpda`
- APGD: `ATTACK.METHOD apgd`

You can also customize the following settings: GPU `GPU`, perturbation budget `ATTACK.EPS`, batch size `DATA.BATCH_SIZE`, etc.

## Acknowledgement
Consider giving this repository a star and cite DiffHammer in your publications if it helps your research.

```
@article{wang2024diffhammer,
  title={DiffHammer: Rethinking the Robustness of Diffusion-Based Adversarial Purification},
  author={Wang, Kaibo and Fu, Xiaowen and Han, Yuxuan and Xiang, Yang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}
```