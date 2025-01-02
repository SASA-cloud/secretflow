'''
Author: Ruijun Deng
Date: 2024-04-12 21:00:21
LastEditTime: 2024-05-30 10:56:30
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/utils/performance_metrics.py
Description: 
'''
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score,recall_score,roc_curve,precision_recall_curve,classification_report
import torch.nn as nn


# 多分类：torch.argmax(y_preds, dim=1)
# 二分类任务，测试test accuracy precision，recall f1
# sigmoid+threshol=0.5
# 对评价指标这些做一个统计
# _ML_Efficacy在用
def test_acc(net, test_loader):
    device = next(net.parameters()).device # 设备
    test_epoch_outputs = []
    test_epoch_labels = []
    sigmoid = nn.Sigmoid()  # 分类层 这里sigmoid都是写在外面的吗
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)

        pred_labels = (sigmoid(outputs) > 0.5).int()

        test_epoch_outputs.append(pred_labels.cpu().detach())
        test_epoch_labels.append(labels.cpu().detach())

    test_acc = accuracy_score(torch.cat(test_epoch_labels), torch.cat(test_epoch_outputs))
    test_precision = precision_score(torch.cat(test_epoch_labels), torch.cat(test_epoch_outputs)) # 增加
    test_recall = recall_score(torch.cat(test_epoch_labels), torch.cat(test_epoch_outputs))
    test_f1 = f1_score(torch.cat(test_epoch_labels), torch.cat(test_epoch_outputs))
    test_classification_report = classification_report(torch.cat(test_epoch_labels),torch.cat(test_epoch_outputs),labels = [0,1])
    
    # print(f"test_acc: {test_acc}")
    # print(f"test_precision: {test_precision}")
    # print(f"test_recall: {test_recall}")
    # print(f"classification_report:")
    # print(test_classification_report)
    
    return test_acc,test_precision,test_recall,test_f1

# pr 曲线，roc曲线
# raw outputs
# 目前也是ML Efficacy在用
def test_pr_roc(net, test_loader):
    device = next(net.parameters()).device
    test_epoch_labels = []
    test_epoch_real_outputs = []
    sigmoid = nn.Sigmoid()  # 分类层 这里sigmoid都是写在外面的吗
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)

        test_epoch_labels.append(labels.cpu().detach())
        test_epoch_real_outputs.append(outputs.cpu().detach())

    precision, recall, thresholds = precision_recall_curve(torch.cat(test_epoch_labels), torch.cat(test_epoch_real_outputs))
    fpr, tpr, thresholds = roc_curve(torch.cat(test_epoch_labels), torch.cat(test_epoch_real_outputs))
    test_auc = roc_auc_score(torch.cat(test_epoch_labels), torch.cat(test_epoch_real_outputs))
    
    return [recall,precision],[fpr,tpr],test_auc


