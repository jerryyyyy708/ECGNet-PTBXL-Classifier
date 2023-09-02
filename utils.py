from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score
import csv
import torch
import numpy as np
import torch.nn as nn
from ECG_Models import *
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def l2norm(signal):
    norms = np.linalg.norm(signal, axis=2, keepdims=True)
    norm_signal = signal / norms
    return norm_signal
    
def calculate_metrics(y_true, y_pred_prob, partial = False):
    # Calculate accuracy
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    macro_auroc = roc_auc_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    
    return accuracy, macro_auroc, micro_f1

def SaveCsvForPlot(filename, train_acc_plot, train_auroc_plot, train_f1_plot, val_acc_plot, val_auroc_plot, val_f1_plot):
    data = [(a,b,c,d,e,f) for a,b,c,d,e,f in zip(train_acc_plot, train_auroc_plot, train_f1_plot, val_acc_plot, val_auroc_plot, val_f1_plot)]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def train(model, train_set, test_set, batch_size, epochs, lr = 1e-2, loss_function = nn.CrossEntropyLoss(), optimizer = None, csv_name = "report", save_root = "result", per_save = 10):
    dev = "cuda:0"
    #model = model.to(dev)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = batch_size)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_acc_plot = []
    train_auroc_plot = []
    train_f1_plot = []
    val_acc_plot = []
    val_auroc_plot = []
    val_f1_plot = []
    best_acc = 0.0

    for epoch in range(epochs):
        
        train_labels_all = []
        train_probs_all = []
        val_labels_all = []
        val_probs_all = []
        
        model.train()
        for x, y in tqdm(train_loader):
            x, y = x.to(dev).float(), y.to(dev).float()
            optimizer.zero_grad()
            output=model(x).float()
            loss=loss_function(output,y.to(torch.int64))
            loss.backward()
            optimizer.step()
            probs = F.softmax(output, dim = 1)
            train_probs_all.append(probs[:, 1].detach().cpu().numpy())
            train_labels_all.append(y.detach().cpu().numpy())
        
        train_probs_all = np.concatenate(train_probs_all)
        train_labels_all = np.concatenate(train_labels_all)
        train_accuracy, train_macro_auroc, train_micro_f1 = calculate_metrics(train_labels_all, np.array(train_probs_all))
        train_acc_plot.append(train_accuracy)
        train_auroc_plot.append(train_macro_auroc)
        train_f1_plot.append(train_micro_f1)
        
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(test_loader):
                x, y = x.to(dev).float(), y.to(dev).to(torch.int64)
                output=model(x).float()
                probs = F.softmax(output, dim = 1)
                val_probs_all.append(probs[:, 1].detach().cpu().numpy())
                #val_probs = output.cpu().numpy()
                val_labels_all.append(y.cpu().numpy())
        
        val_probs_all = np.concatenate(val_probs_all)
        val_labels_all = np.concatenate(val_labels_all)
        val_accuracy, val_macro_auroc, val_micro_f1 = calculate_metrics(val_labels_all, np.array(val_probs_all))
        val_acc_plot.append(val_accuracy)
        val_auroc_plot.append(val_macro_auroc)
        val_f1_plot.append(val_micro_f1)
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model, os.path.join(save_root, f'{csv_name}_best.pt'))
        torch.save(model, os.path.join(save_root, f'{csv_name}_last.pt'))
        if (epoch+1) % per_save == 0:
            torch.save(model, os.path.join(save_root, f'{csv_name}_epoch{epoch+1}.pt'))
            
        print(f"Epoch {epoch+1}:")
        print(f"Train: accuracy = {train_accuracy}, AUROC = {train_macro_auroc}, F1 = {train_micro_f1}")  
        print(f"Test: accuracy = {val_accuracy}, AUROC = {val_macro_auroc}, F1 = {val_micro_f1}")
        SaveCsvForPlot(os.path.join(save_root, f"{csv_name}.csv"), train_acc_plot, train_auroc_plot, train_f1_plot, val_acc_plot, val_auroc_plot, val_f1_plot)