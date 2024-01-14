# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch

from config import opt
from get_model_data import get_dataset
from pre_train_encoder_vae import pre_train_encoder_vae
# from meta_infer import meta_inference
from test import test
def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)
def seed_torch(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_gpu(opt.gpu)
    seed_torch(opt.seed)
    
    (dataset_train, dataset_val, dataset_test, data_loader) = get_dataset(opt)

    if opt.phase == 'pretrain_encoder':
        print('pretrain_encoder')
        pre_train_encoder_vae(opt,dataset_train, dataset_val, dataset_test, data_loader)
    elif opt.phase == 'test':
        test(opt,dataset_test,data_loader)