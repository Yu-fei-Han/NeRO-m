import argparse

from train.trainer import Trainer
from utils.base_utils import load_cfg
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='configs/shape/ours/owl.yaml', type=str)
flags = parser.parse_args()

Trainer(load_cfg(flags.cfg)).run()

