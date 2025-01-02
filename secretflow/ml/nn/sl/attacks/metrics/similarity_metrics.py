'''
Author: Ruijun Deng
Date: 2023-12-12 16:00:55
LastEditTime: 2024-08-02 11:28:44
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/utils/similarity_metrics.py
Description: 
'''

import torch
import torch.nn as nn
from skimage import metrics # 测量SSIM
import numpy as np
import pandas as pd
import os
from .performance_metrics import  test_acc, test_pr_roc

class SimilarityMetrics():
    def __init__(self,gpu=True, type=0) -> None:
        # 0 表格, 1 图像
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

        # 5种相似度计算方法
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.euclidean_distance = torch.nn.PairwiseDistance(p=2,eps=1e-6,keepdim=False)
        self.mse_loss = nn.MSELoss()
        self.ssim = self.ssim_metric
        self.accuracy = self._accuracy
        self.rebuild_acc = self._tabRebuildAcc

        self.sim_metric_dict = {'cos':[],'euc':[],'mse':[],'acc':[],'ssim':[]}
        # if type == 0: # tabular
        #     self.sim_metric_dict = {'cos':[],'euc':[],'mse':[],'acc':[]}
        # elif type == 1: # image
        #     self.sim_metric_dict = {'ssim':[],'mse':[],}

    def ssim_metric(self,deprocessImg_raw,deprocessImg_inversed,deprocess=None):
        # ssim = measure.compare_ssim(
        # X=np.moveaxis(ref_img, 0, -1),  # 把dim=0的维度移动到最后
        # Y=np.moveaxis(inv_img, 0, -1),  
        # data_range = inv_img.max() - inv_img.min(), # 为什么是inv_img的呢
        # multichannel=True)
        ref_img = deprocessImg_raw.clone().detach().cpu().numpy().squeeze()
        inv_img = deprocessImg_inversed.clone().detach().cpu().numpy().squeeze()
        ssim = metrics.structural_similarity(np.moveaxis(inv_img,0,-1), np.moveaxis(ref_img,0,-1), 
                                                data_range=inv_img.max() - inv_img.min(), channel_axis=-1)
        return ssim

    # 将输出的0-1之间的值根据0.5为分界线分为两类,同时统计正确率
    def _compute_correct_prediction(self, y_targets, y_prob_preds, threshold=0.5):
        y_hat_lbls = []
        pred_pos_count = 0
        pred_neg_count = 0
        correct_count = 0
        for y_prob, y_t in zip(y_prob_preds, y_targets):
            print('y_prob.size: ',y_prob.size())
            if y_prob <= threshold: # 小于threshold
                pred_neg_count += 1
                y_hat_lbl = 0
            else: # 大于threshold
                pred_pos_count += 1
                y_hat_lbl = 1
            y_hat_lbls.append(y_hat_lbl)
            if y_hat_lbl == y_t:
                correct_count += 1

        return np.array(y_hat_lbls), [pred_pos_count, pred_neg_count, correct_count]

    def _accuracy(self, y_targets, y_prob_preds, threshold=0.5):
        # 计算正确率
        _, ans = self._compute_correct_prediction(
            y_targets=y_targets, y_prob_preds=y_prob_preds, threshold=threshold)
        pred_pos_count, pred_neg_count, correct_count = ans

        return correct_count/len(y_targets)
    
    def report_similarity(self):
        for (key,value) in self.sim_metric_dict.items():
            if len(value)>0: # 这个list存储了数据
                print(f"average {key}: {np.mean(value)}")

    def store_similarity(self,inverse_route):
        dic = {}
        for (key,value) in self.sim_metric_dict.items():
            if len(value)>0: # 这个list存储了数据
                dic[key] = value
        pd.DataFrame(dic).to_csv(inverse_route,index=False)

    def collect_sim_tab(self,raw,inverted,tab):
        cos = self.cosine_similarity(raw, inverted).item()
        euc = self.euclidean_distance(raw, inverted).mean().item()
        mse = self.mse_loss(raw, inverted).item()
        # accuracy = self.accuracy(raw, inverted).item()
        accuracy,_,_ = self.rebuild_acc(raw, inverted,tab)
        
        self.sim_metric_dict['cos'].append(cos)
        self.sim_metric_dict['euc'].append(euc)
        self.sim_metric_dict['mse'].append(mse)
        self.sim_metric_dict['acc'].append(accuracy)

    def collect_sim_img(self,raw,inverted,deprocess):
        if deprocess == None:
            ssim = self.ssim_metric(raw.clone(), inverted.clone()).item()
        else:
            ssim = self.ssim_metric(deprocess(raw.clone()), deprocess(inverted.clone())).item()
        mse = self.mse_loss(raw, inverted).item()
        euc = self.euclidean_distance(raw, inverted).mean().item()

        self.sim_metric_dict['ssim'].append(ssim)
        self.sim_metric_dict['mse'].append(mse)
        self.sim_metric_dict['euc'].append(euc)

    def _tabRebuildAcc(self,originData, rebuildData, tab, sigma=0.2): # 数值性不超过0.2就算正确。
        originData = originData.detach().clone()
        rebuildData = rebuildData.detach().clone()

        # 1. 计算onehot列的准确率
        onehot_index = tab['onehot']

        onehotsuccessnum = 0
        numsuccessnum = 0

        num_acc, onehot_acc = 0.0,0.0

        if len(tab['onehot']):
            for item in onehot_index: # 遍历keys 是看对应的idex是否一致
                origin = torch.argmax(originData[:, onehot_index[item]], dim=1) # 提取几列，找到 argmax的
                rebuild = torch.argmax(rebuildData[:, onehot_index[item]], dim=1) # 提取几列

                onehotsuccessnum += torch.eq(origin, rebuild).sum().item()

            onehot_acc = onehotsuccessnum/(len(onehot_index) * rebuildData.shape[0])

        # 2. 计算num列的准确率
        if len(tab['numList']):
            for i in tab['numList']:
                origin = originData[:, i]
                rebuild = rebuildData[:, i]
                diff = torch.abs(origin - rebuild)
                # print("diff:", diff)
                # if diff < sigma:
                numsuccessnum += (diff < sigma).sum().item()

            # print("numsuccessnum:", numsuccessnum)
            num_acc = numsuccessnum/(len(tab['numList']) * rebuildData.shape[0])

        return (numsuccessnum+onehotsuccessnum)/((len(tab['numList']) + len(onehot_index)) *\
                                                  rebuildData.shape[0]), onehot_acc, num_acc

    def _ML_Efficacy(self,
                     raw_net_route,inverted_net_route,
                     raw_loader,fake_loader,test_loader,
                     rawNet=None,fakeNet=None):
        # TODO: 未完成
        print("---test using ML efficacy----")
        # 先将csv数据载入
        inverse_data = pd.read_csv(fake_data_route, low_memory=False,  header=None, index_col=False)
        origin_data = pd.read_csv(raw_data_route, low_memory=False,  header=None, index_col=False)
        label_data = pd.read_csv(label_route, low_memory=False,  header=None, index_col=False)

        # fake data训练数据集加载
        fake_dataset = DFDataset(inverse_data, label_data)
        fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=256, shuffle=False, num_workers=16)

        # 原始数据集合加载
        test_dataset = DFDataset(origin_data, label_data)
        raw_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=16)


        # Similarity检测：
        # 使用fake data 对模型进行训练
        print("----Using Fake input to train a ML model----")

        # 用真实数据 训练similairty 网络
        if os.path.exists(raw_net_route):
            rawNet = torch.load(raw_net_route)
        else:
            if rawNet == None:
                print("rawNet = None")
                exit(-1)
            train(net=rawNet, train_loader=raw_loader, gpu=True,
                save_model_route=raw_net_route,BatchSize=256)

        # 用fake data 训练similarity 网络
        if os.path.exists(inverted_net_route):
            fakeNet = torch.load(inverted_net_route)
        else:
            if fakeNet == None:
                print("fakeNet = None")
                exit(-1)
            train(net=fakeNet, train_loader=fake_loader, gpu=True,
                save_model_route=inverted_net_route,BatchSize=256)

        print("------print ML efficacy----")
        # 测试ML efficacy on fakeNet and raw Net 并打印结果
        fake_acc,fake_precision,fake_recall,fake_f1 = test_acc(net=fakeNet, test_loader=test_loader)
        raw_acc,raw_precision,raw_recall,raw_f1 = test_acc(net=rawNet, test_loader=test_loader)

        [fake_recall_list,fake_precision_list],[fake_fpr_list,fake_tpr_list],fake_auc = test_pr_roc(net=fakeNet, test_loader=test_loader)
        [raw_recall_list,raw_precision_list],[raw_fpr_list,raw_tpr_list],raw_auc = test_pr_roc(net=rawNet, test_loader=test_loader)

        print("fakeNet ML efficacy: ", "accuracy =", fake_acc, "f1 =", fake_f1, "auc =", fake_auc, "precision =", fake_precision, "recall =", fake_recall)
        print("rawNet ML efficacy: ", "accuracy =", raw_acc, "f1 =", raw_f1, "auc =", raw_auc, "precision =", raw_precision, "recall =", raw_recall)

        # # 绘制pr曲线
        # plt.figure()
        # plt.plot(fake_recall_list,fake_precision_list)
        # plt.plot(raw_recall_list,raw_precision_list)
        # plt.legend(["fake","raw"])
        # plt.savefig("./p_r_curve.jpg")

        # # 绘制roc曲线
        # plt.figure()
        # plt.plot(fake_fpr_list,fake_tpr_list)
        # plt.plot(raw_fpr_list,raw_tpr_list)
        # plt.legend(["fake","raw"])
        # plt.savefig("./roc_curve.jpg")
