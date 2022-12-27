# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""
import pickle
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from pathlib import PosixPath
import matplotlib.pyplot as plt
from collections import defaultdict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def order_labels(labels):
    if len(labels) == 1:
        lab = labels[0]
        if lab %2 ==0:
            return [None, lab]
        return [lab, None]

    elif len(labels) == 2:
        lab1, lab2 = labels
        if lab1 % 2 == 0:
            return [lab2, lab1]
        return [lab1, lab2]

    else:
        lab1, lab2, lab3 = labels
        if lab1 %2 == 0 and lab2 %2 == 0:
            return [lab3, lab1]
        elif lab1 %2 == 0 and lab2 %2 == 1:
            return [lab2, lab1]
        elif lab1 %2 == 1 and lab2 %2 ==1:
            return [lab1, lab3]
        elif lab1 %2 ==1 and lab2 %2 == 0:
            return [lab1, lab2]
        else:
            raise ValueError("Fuck")
        



def get_labels(path_examples, path_labels):
    with open(path_examples) as f:
        lines = f.readlines()
        label_names = [l.split('.')[0]+'.txt' for l in lines]
    all_labels = []
    for label in label_names:
        path = os.path.join(path_labels, label)
        with open(path) as f:
            boxs = f.readlines()
            labels = [int(b.split(' ')[0]) for b in boxs]
            all_labels.append(order_labels(labels))   
    return all_labels




def test():  # hyp is path/to/hyp.yaml or hyp dictionary
    hyp = 'yolov5/data/hyps/hyp.scratch-low.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict

    test_loader = create_dataloader('/home/student/HW1/datasets/HW1_dataset/test_fixed.txt',
                                       640,
                                       16 // WORLD_SIZE * 2,
                                       32,
                                       False,
                                       hyp=hyp,
                                       cache=None,
                                       rect=True,
                                       rank=-1,
                                       workers=8 * 2,
                                       pad=0.5,
                                       prefix=colorstr('test: '))[0]
    
    callbacks = Callbacks()
    data_dict = {
                'path': PosixPath('/home/student/HW1/datasets/HW1_dataset/train_fixed.txt'),
                'train': '/home/student/HW1/datasets/HW1_dataset/valid_fixed.txt',
                'val': '/home/student/HW1/datasets/HW1_dataset/valid_fixed.txt',
                'test': None,
                'names': {0: 'Right_Scissors', 1: 'Left_Scissors', 2: 'Right_Needle_driver', 3: 'Left_Needle_driver', 4: 'Right_Forceps', 5: 'Left_Forceps', 6: 'Right_Empty', 7: 'Left_Empty'},
                'nc': 8}
    
    #### CHANGE HERE ####
    best_exp = "exp8"
    best_dict = torch.load(f"yolov5/runs/train/{best_exp}/weights/best.pt")
    size = best_dict['opt']['weights'][-4]
    lr = 0.01
    opt = "Adam"
    #### CHANGE HERE ####

    model_for_val = best_dict["model"].to(device)
    ema = ModelEMA(model_for_val)
    
    ap25, ap50, ap75, classes = validate.run(data_dict,
                        batch_size=16 // WORLD_SIZE * 2,
                        imgsz=640,
                        half=True,
                        model=ema.ema,
                        single_cls=False,
                        dataloader=test_loader,
                        save_dir="t",
                        plots=False,
                        callbacks=callbacks, 
                        test_results=True)
    
    names = data_dict['names']
    barWidth = 0.25
    br1 = range(len(classes)) 
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    plt.bar(br1, ap25, color ='r', width = barWidth,
        edgecolor ='grey', label ='mAP@25')
    plt.bar(br2, ap50, color ='g', width = barWidth,
        edgecolor ='grey', label ='mAP@50')
    plt.bar(br3, ap75, color ='b', width = barWidth,
        edgecolor ='grey', label ='mAP@75')
    plt.xticks([r + barWidth for r in range(len(classes))], [names[i] for i in classes], rotation = 45)
    plt.legend(loc = "upper center", fancybox= True)
    plt.title(f"Test results for {best_exp}: size = {size}, lr = {lr}, optimizer = {opt}")
    plt.tight_layout()
    plt.savefig('test.jpg')
    plt.show()
    model = torch.hub.load('yolov5', 'custom', path=fr"yolov5/runs/train/{best_exp}/weights/best.pt", source='local') 
    test_lines = []
    with open('/home/student/HW1/datasets/HW1_dataset/test_fixed.txt') as f:
        for line in f.readlines():
            test_lines.append("datasets/HW1_dataset/"+line[1:-1])
    output = model(test_lines)
    all_preds = []
    for boxes in output.pred:
        if len(boxes) == 0:
            all_preds.append([None, None])
        elif len(boxes) == 1:
            label = boxes[:, -1].int().item()
            if label %2 ==0:
                all_preds.append([None, label])
            else:
                all_preds.append([label, None])
        else:
            labels = boxes[:, -1].int().tolist()
            confs = boxes[:, -2].tolist()
            confs_sum_right = defaultdict(lambda: 0)
            confs_sum_left = defaultdict(lambda: 0)
            for l, c in zip(labels, confs):
                if l % 2 == 0:
                    confs_sum_right[l]+=c
                else:
                    confs_sum_left[l]+=c
            right_pred = max((p, l) for l, p in confs_sum_right.items())[1] if len(confs_sum_right) else None
            left_pred = max((p, l) for l, p in confs_sum_left.items())[1] if len(confs_sum_left) else None
            all_preds.append([left_pred, right_pred])
    
    all_labels = get_labels('/home/student/HW1/datasets/HW1_dataset/test.txt', "/home/student/HW1/datasets/HW1_dataset/labels")
    with open("labels", "wb") as fp:
        pickle.dump(all_labels, fp)
    with open("preds", "wb") as fp:
        pickle.dump(all_preds, fp)



        
    

if __name__ == '__main__':
    test()