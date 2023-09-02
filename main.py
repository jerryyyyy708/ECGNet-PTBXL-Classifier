import torch
import numpy as np
import torch.nn as nn
from ECG_Models import *
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm
from utils import *
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description = "PTB-XL Dataset Training with ECGNet and ECGNetLite")
    parser.add_argument("--mode", default = "train", type = str, required = False, help = "train/test" )
    parser.add_argument("--model", "-m", default = "ECGNet", type = str, required = False, help = 'ECGNet or ECGNetLite(can typically type Lite)')
    parser.add_argument("--device", "-d", default = "cuda:0", type = str, required = False, help = "cuda:0 or cpu")
    parser.add_argument("--ckpt", "-c", default = "", type = str, required = False, help = 'load pretrained network')
    parser.add_argument("--batch", "-b", default = 128, type = int, required = False, help = "batch size")
    parser.add_argument("--epoch", "-e", default = 50, type = int, required = False, help = "training/finetuning epoch")
    parser.add_argument("--lr", "-l", default = 1e-2, type = float, required = False, help = "learning rate")
    parser.add_argument("--optim", "-o", default = "Adam", type = str, required = False, help = "Optimizer: SGD or Adam")
    parser.add_argument("--momentum", "-t", default = 0.9, type = float, required = False, help = "set momentum for SGD")
    parser.add_argument("--per_save", "-p", default = 10000, type = int, required = False, help = "save checkpoint every n epochs")
    parser.add_argument("--save_root", "-r", default = ".", type = str, required = False, help = "folder to save the checkpoint")
    parser.add_argument("--norm", "-n", default = "None", type = str, required = False, help = "normalize method")
    parser.add_argument("--step_size", "-z", default = 20, type = int, required = False, help = "LRScheduler step size")
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()

    csv_name, device, ckpt, batch_size, epochs, lr, optim_name, momentum, per_save, save_root = args.model, args.device, args.ckpt, args.batch, args.epoch, args.lr, args.optim, args.momentum, args.per_save, args.save_root
    
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    
    if args.mode == "train":
        X_train = np.load('Dataset/PTB-XL_X_train.npy', allow_pickle=True)
        y_train = np.load('Dataset/PTB-XL_y_train.npy', allow_pickle=True)
        trainset = ECG_Data(X_train, y_train)

    X_test = np.load('Dataset/PTB-XL_X_test.npy', allow_pickle=True)
    y_test = np.load('Dataset/PTB-XL_y_test.npy',allow_pickle=True)
    testset = ECG_Data(X_test, y_test)
    
    if 'l2' in args.norm:
        X_train = l2_norm(X_train)
        X_test = l2_norm(X_test)
    elif 'Z' in args.norm:
        X_train = Zscore_norm(X_train)
        X_test = Zscore_norm(X_test)
    elif 'max' in args.norm:
        X_train = minmax_norm(X_train)
        X_test = minmax_norm(X_test)

    if 'Lite' in csv_name:
        model = ECGNetLite().to(device)
    else:
        model = ECGNet().to(device)
    
    if ".pt" in ckpt:
        model = torch.load(ckpt).to(device)

    if optim_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    
    print("Model: ", csv_name)
    print("Num of Parameters: ", count_parameters(model))

    if args.mode == "train":
        train(model, trainset, testset, batch_size = batch_size,  epochs = epochs, lr = lr, csv_name = csv_name, save_root = save_root, per_save = per_save, step_size = args.step_size)
    else:
        test(model, testset, batch_size, save_root, device)