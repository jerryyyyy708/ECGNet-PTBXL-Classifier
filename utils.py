from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
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
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import seaborn as sns

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def l2_norm(signal):
    norms = np.linalg.norm(signal, axis=2, keepdims=True)
    norm_signal = signal / norms
    return norm_signal

def minmax_norm(data):
    min_val = np.min(data, axis=(0,2), keepdims=True)
    max_val = np.max(data, axis=(0,2), keepdims=True)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def Zscore_norm(data):
    mean = np.mean(data, axis=(0,2), keepdims=True)
    std = np.std(data, axis=(0,2), keepdims=True)
    normalized_data = (data - mean) / std
    return normalized_data

    
def calculate_metrics(y_true, y_pred_prob, test = False, root = ""):
    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Calculate AUC (Area Under the Receiver Operating Characteristic Curve)
    auc = roc_auc_score(y_true, y_pred_prob)  # Note: for AUC, we use y_pred_prob instead of y_pred

    if test:
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(root, 'Test_Result.png'), format='png', dpi=300)

    return accuracy, precision, recall, f1, auc

def SaveCsvForPlot(filename, train_acc_plot, train_precision_plot, train_recall_plot, train_f1_plot, train_auc_plot, val_acc_plot, val_precision_plot, val_recall_plot, val_f1_plot, val_auc_plot):
    data = [(a,b,c,d,e,f,g,h,i,j) for a,b,c,d,e,f,g,h,i,j in zip(train_acc_plot, train_precision_plot, train_recall_plot, train_f1_plot, train_auc_plot, val_acc_plot, val_precision_plot, val_recall_plot, val_f1_plot, val_auc_plot)]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

def train(model, train_set, test_set, batch_size, epochs, lr = 1e-2, loss_function = nn.CrossEntropyLoss(), optimizer = None, csv_name = "report", save_root = "result", per_save = 10, min_lr = 0.0001, step_size = 20):
    dev = "cuda:0"
    #model = model.to(dev)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = batch_size)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    train_acc_plot = []
    train_precision_plot = []
    train_recall_plot = []
    train_f1_plot = []
    train_auc_plot = []
    val_acc_plot = []
    val_precision_plot = []
    val_recall_plot = []
    val_f1_plot = []
    val_auc_plot = []
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
        
        scheduler.step()
        if optimizer.param_groups[0]['lr'] < min_lr:
            optimizer.param_groups[0]['lr'] = min_lr

        train_probs_all = np.concatenate(train_probs_all)
        train_labels_all = np.concatenate(train_labels_all)
        train_accuracy, train_precision, train_recall, train_f1, train_auc = calculate_metrics(train_labels_all, np.array(train_probs_all))
        train_acc_plot.append(train_accuracy)
        train_precision_plot.append(train_precision)
        train_recall_plot.append(train_recall)
        train_f1_plot.append(train_f1)
        train_auc_plot.append(train_auc)
        
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
        val_accuracy, val_precision, val_recall, val_f1, val_auc = calculate_metrics(val_labels_all, np.array(val_probs_all))
        val_acc_plot.append(val_accuracy)
        val_precision_plot.append(val_precision)
        val_recall_plot.append(val_recall)
        val_f1_plot.append(val_f1)
        val_auc_plot.append(val_auc)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model, os.path.join(save_root, f'{csv_name}_best.pt'))
        torch.save(model, os.path.join(save_root, f'{csv_name}_last.pt'))
        if (epoch+1) % per_save == 0:
            torch.save(model, os.path.join(save_root, f'{csv_name}_epoch{epoch+1}.pt'))
            
        print(f"Epoch {epoch+1}:")
        print(f"Train: accuracy = {train_accuracy}, Precision = {train_precision}, Recall = {train_recall}, F1 = {train_f1}, AUC = {train_auc}")  
        print(f"Test: accuracy = {val_accuracy}, Precision = {val_precision}, Recall = {val_recall}, F1 = {val_f1}, AUC = {val_auc}")
        SaveCsvForPlot(os.path.join(save_root, f"{csv_name}.csv"), train_acc_plot, train_precision_plot, train_recall_plot, train_f1_plot, train_auc_plot, val_acc_plot, val_precision_plot, val_recall_plot, val_f1_plot, val_auc_plot)

def test(model, test_set, batch_size, root, dev):
    test_loader = DataLoader(test_set, batch_size = batch_size)
    model.eval()
    val_probs_all = []
    val_labels_all = []
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
    val_accuracy, val_precision, val_recall, val_f1, val_auc = calculate_metrics(val_labels_all, np.array(val_probs_all), test = True, root = root)
    print(f"Test: accuracy = {val_accuracy}, Precision = {val_precision}, Recall = {val_recall}, F1 = {val_f1}, AUC = {val_auc}")