import os
import sys
import argparse

import torch
import torch.backends.cudnn as cudnn
from torchvision import models

import yaml
from datetime import datetime

from model.CPNN import ACPNN
from model.CPNN import BCPNN

from utils.seed import setup_seed
from utils.logger import Logger

if __name__ == '__main__':
    cudnn.enabled = True
    cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(description="EmoRank")
    parser.add_argument('--exp_name', type=str, default='demo_exp', help='training experiment name')
    parser.add_argument('--config_path', type=str, default='./config/FI/FI_res50.yaml', help='training config yaml file path')
    parser.add_argument('--resume_path', type=str, default='', help='resume model path')
    parser.add_argument('--log_path', type=str, default='./logs/training', help='training log saving path')
    parser.add_argument('--seed', type=int, default='1234', help='random seed')
    # parser.add_argument("--fp16", default=False)
    args = parser.parse_args()
    
    print("============== start initialization ==============")
    
    setup_seed(int(args.seed))
    
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
        
    log_path = os.path.join(args.log_path, args.exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    config["snapshots"] = os.path.join(log_path, 'snapshots')
    os.makedirs(config["snapshots"], exist_ok=True)
    config["imgshots"] = os.path.join(log_path, 'imgshots')
    os.makedirs(config["imgshots"], exist_ok=True)
    
    now = datetime.now()
    formatted_now = now.strftime("%y%m%d-%H%M%S")

    log_txt = os.path.join(log_path, f'log_{formatted_now}.txt')
    sys.stdout = Logger(log_txt, sys.stdout)
    
    print(config)
    
    print("============== model initialization ==============")
    
    model = EmoRank(config)
    model.cuda()
    
    from utils.parameter import count_parameters
    count_parameters(model)
    
    if args.resume_path:
        print(f"resume from: {args.log_path}")
        checkpoint = torch.load(args.log_path)
        model.load_state_dict(checkpoint, strict=True)
        
    print("============== data initialization ==============")
    
    print("============== start training ==============")
    
