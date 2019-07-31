from __future__ import division

from utils.utils import *
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.parse_yolo_weights import parse_yolo_weights
from models.yolov3 import *
from dataset.cocodataset import *

import os
import argparse
import yaml
import random

import torch
from torch.autograd import Variable
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_default.cfg',
                        help='config file. see readme')
    parser.add_argument('--weights_path', type=str,
                        default=None, help='darknet weights file')
    parser.add_argument('--n_cpu', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--checkpoint_interval', type=int,
                        default=1000, help='interval between saving checkpoints')
    parser.add_argument('--eval_interval', type=int,
                            default=4000, help='interval between evaluations')
    parser.add_argument('--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints',
                        help='directory where checkpoint files are saved')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument(
        '--tfboard', help='tensorboard path for logging', type=str, default=None)
    return parser.parse_args()


def main():
    """
    YOLOv3 trainer. See README for details.
    """
    args = parse_args()
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)

    print("successfully loaded config file: ", cfg)

    momentum = cfg['TRAIN']['MOMENTUM']
    decay = cfg['TRAIN']['DECAY']
    burn_in = cfg['TRAIN']['BURN_IN']
    iter_size = cfg['TRAIN']['MAXITER']
    steps = eval(cfg['TRAIN']['STEPS'])
    batch_size = cfg['TRAIN']['BATCHSIZE']
    subdivision = cfg['TRAIN']['SUBDIVISION']
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    random_resize = cfg['AUGMENTATION']['RANDRESIZE']
    base_lr = cfg['TRAIN']['LR'] / batch_size / subdivision

    print('effective_batch_size = batch_size * iter_size = %d * %d' %
          (batch_size, subdivision))

    # Initiate model
    model = YOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)

    if args.weights_path:
        print("loading darknet weights....", args.weights_path)
        parse_yolo_weights(model, args.weights_path)
    elif args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        state = torch.load(args.checkpoint)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    if cuda:
        print("using cuda") 
        model = model.cuda()


    model.train()

    imgsize = cfg['TRAIN']['IMGSIZE']

    evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                    data_dir='COCO/',
                    img_size=cfg['TEST']['IMGSIZE'],
                    confthre=cfg['TEST']['CONFTHRE'],
                    nmsthre=cfg['TEST']['NMSTHRE'])
    ap50_95, ap50 = evaluator.evaluate(model)


if __name__ == '__main__':
    main()
