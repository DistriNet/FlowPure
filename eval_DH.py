import os
import sys
import logging
import argparse
from config import get_config
import torch
from defense import *
from attacker import *
from utils import *



if __name__ == '__main__':

    for dataset in ['CIFAR10', 'CIFAR100']:
        for purification_method in ['gauss_flowpure_0.15', 'gauss_flowpure_0.2', 'cw_flowpure', 'pgd_flowpure',
                                    'diffpure', 'gdmp', 'llhd_maximize', 'adbm']:
            accuracy = None
            for sample_run in [1,2,3]:
                cfg = get_config(purification_method, dataset, 'DH', sample_run, 32, 512)
                cfg.NAME = "DH"
                cfg.OUTPUT_DIR = 'dir'
                cfg.EXP = ''

                base_path = f'results/{dataset}/{purification_method}/DH_{sample_run}/'

                if not os.path.exists(base_path):
                    os.makedirs(base_path)

                logging.basicConfig(
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(base_path + 'exp.log', 'w'),
                            logging.FileHandler(base_path + 'info.log', 'w'),
                            logging.StreamHandler(sys.stdout)]
                )

                logger = logging.getLogger()

                logger.handlers.clear()  # Remove pre-existing handlers

                logging.basicConfig(
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(base_path + 'exp.log', 'w'),
                        logging.FileHandler(base_path + 'info.log', 'w'),
                        logging.StreamHandler(sys.stdout)
                    ]
                )

                # Re-fetch the logger after reconfiguring
                logger = logging.getLogger()


                logger.handlers[1].setLevel(logging.INFO)
                logger.handlers[2].setLevel(logging.INFO)
                seeder = iter(range(int(1e9)))

                logger.info('Configs:\n{:}\n{:}\n'.format(cfg, '-' * 30))
                df_config = cfg.DEFENSE[cfg.DEFENSE.METHOD.upper()]
                diffusion = get_model(cfg.DEFENSE.DIFFUSION_NAME).cuda().eval()
                classifier = get_model(cfg.DEFENSE.CLASSIFIER_NAME)#.cuda().eval()
                model = get_defense(cfg.DEFENSE.METHOD)(diffusion, classifier, df_config)
                test_loader = get_dataloader(cfg)
                attacker = get_attacker(cfg.ATTACK.METHOD)(model, cfg, logger, seeder)
                if accuracy == None:
                    accuracy = attacker.evaluate_accuracy(test_loader)
                data, robustness = attacker.evaluate_robustness(test_loader)
                # torch.save(data, base_path + 'data.pt')
                for k, v in {**accuracy, **robustness}.items():
                    if 'loss' not in k:
                        logger.info('{:13}: {:8.3%}'.format(k, v))
