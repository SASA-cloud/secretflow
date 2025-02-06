'''
Author: Ruijun Deng
Date: 2023-12-12 12:42:45
LastEditTime: 2024-10-02 04:46:41
LastEditors: Ruijun Deng
FilePath: /PP-Split/ppsplit/attacks/model_inversion/inverse_model.py
Description: 
'''
import torch.nn as nn
import torch
import os
import tqdm
import numpy as np
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能是由于是MacOS系统的原因
import pandas as pd
from torch.utils.data import Dataset
torch.multiprocessing.set_sharing_strategy('file_system')

from metrics.similarity_metrics import SimilarityMetrics
import torchvision
import time

# 隐语
from secretflow import reveal
from secretflow.device import PYU
from secretflow.ml.nn.core.torch import TorchModel
from secretflow.ml.nn.sl.backend.torch.sl_base import SLBaseTorchModel
from secretflow.ml.nn.callbacks.attack import AttackCallback

import logging



class InverseModelAttackCallbacks(AttackCallback):
    def __init__(self,
                data_type = 0, # 表格数据
                inverse_dir = './inverted/',
                decoder_route = './decoder_net.pth',
                device = "cuda:0",
                trainloader=None,
                testloader=None,

                attack_party: PYU = None,
                victim_party: PYU = None,

                decoder_model_wrapper=None,
                tab=None,
                **params,
                ):
        super().__init__(**params)

        self.metrics = None
        self.data_type = data_type
        self.device = device
        self.inverse_dir = inverse_dir
        self.decoder_route = decoder_route
        self.trainloader = trainloader
        self.testloader = testloader
        self.tab=tab

        print("trainloader in callbacks: ",trainloader)
        print("testloader in callbacks: ",testloader)

        self.attack_party = attack_party
        self.victim_party = victim_party

        self.decoder_model_wrapper = decoder_model_wrapper


    def on_train_end(self,): # 因为是针对inference的attack，并不需要参与训练
        # 获取训练好的victim的base模型
        def get_victim_model(victim_worker: SLBaseTorchModel):
            return victim_worker.model_base.to(self.device)
        # victim_model.to(self.device)

        def get_decoder_model(attack_worker: SLBaseTorchModel, victim_model):
            # 创建Inverse Model Attack对象
            im_attack = InverseModelAttack(
                        decoder_route=self.decoder_route,
                        data_type=self.data_type,
                        inverse_dir=self.inverse_dir,
                        device=self.device,
                        )
            

            # 训练inverse model
            # 加载decoder模型
            if not os.path.isfile(self.decoder_route): # 如果没有训练decoder
                # 训练decoder
                # trainloader,testloader = self.data_builder(train_bs = self.train_bs,test_bs = self.test_bs) 
                self.decoder_net = self.decoder_model_wrapper.model_fn().to(self.device)
                trainloader,testloader = self.trainloader,self.testloader 

                self.decoder_net = im_attack.train_decoder(client_net=victim_model,decoder_net=self.decoder_net,
                                        train_loader=trainloader,test_loader=None,
                                        epochs=20)
            else:
                self.decoder_net = torch.load(self.decoder_route)
                logging.info(f"Load decoder model from:{self.decoder_route}")
            print("decoder_net: ",self.decoder_net)

            # 实现攻击,恢复testloader中所有图片
            # trainloader,testloader = self.data_builder(train_bs = self.train_bs,test_bs = self.test_bs)
            trainloader,testloader = self.trainloader,self.testloader 
            im_attack.inverse(client_net=victim_model,decoder_net=self.decoder_net,
                            train_loader=trainloader,test_loader=testloader,
                            deprocess=None,
                            save_fake=True,
                            tab=self.tab)
            print("inverse done")

            return {}
        
        victim_model = reveal(self._workers[self.victim_party].apply(get_victim_model))
        self.metrics = reveal(self._workers[self.attack_party].apply(get_decoder_model, victim_model))

    def get_attack_metrics(self):
        return self.metrics

class InverseModelAttack():
    def __init__(self,gpu=True,decoder_route=None,data_type=0,inverse_dir=None,device=None) -> None:

        self.data_type=data_type # 0 是表格数据集，1是图像数据集

        # device
        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
        else:
            self.device = device

        # 储存或加载攻击模型的路径
        self.inverse_dir = inverse_dir if inverse_dir else './inverted/'
        self.decoder_route = decoder_route if decoder_route else './decoder_net.pth'
        if not os.path.exists(self.inverse_dir):
            os.makedirs(self.inverse_dir)

    def train_decoder(self,client_net,decoder_net,
                      train_loader,test_loader,
                      epochs,optimizer=None):
        
        # 打印相关信息
        print("----train decoder----")
        print("client_net: ")
        print(client_net)
        print("decoder_net: ")
        print(decoder_net)

        # 网络搬到设备上
        client_net.to(self.device)
        decoder_net.to(self.device)
        

        # loss function 统一采用MSELoss？
        if not optimizer:
            # optimizer = torch.optim.SGD(decoder_net.parameters(), 1e-3)
            optimizer = torch.optim.Adam(decoder_net.parameters())
        criterion = nn.MSELoss()
        

        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            # train and update
            epoch_loss = []
            for i, (trn_X,) in enumerate(tqdm.tqdm(train_loader)):
                trn_X = trn_X.to(self.device)
                batch_loss = []

                optimizer.zero_grad()

                out = decoder_net(client_net(trn_X))

                loss = criterion(out, trn_X)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss)) # 多个batch的平均


            print("--- epoch: {0}, train_loss: {1}".format(epoch, epoch_loss))

        # 储存decoder模型
        torch.save(decoder_net, self.decoder_route)
        print("model saved")
        return decoder_net

    def inverse(self,client_net,decoder_net,
                train_loader,test_loader,
                deprocess=None,
                save_fake=True,
                tab=None):
        if self.data_type==1 and deprocess==None: # 图像数据集没给deprocess方法
            print("图像数据没给deprocess 函数")
            exit(-1)
        
        if self.data_type==0: # 表格数据
            return self._inverse_tab(client_net,decoder_net,train_loader,test_loader,save_fake,tab)
        else: # 图像数据
            return self._inverse_image(client_net,decoder_net,train_loader,test_loader,deprocess,save_fake)
        
    def _inverse_tab(self,client_net,decoder_net,
                train_loader,test_loader,
                save_fake=True,
                tab=None):
        
        # 打印相关信息
        print("----train decoder----")
        print("client_net: ")
        print(client_net)
        print("decoder_net: ")
        print(decoder_net)

        # 网络搬到设备上
        client_net.to(self.device)
        decoder_net.to(self.device)
        client_net.eval() 
        decoder_net.eval()
        
        # 记录数据:
        sim_metrics = SimilarityMetrics(type = self.data_type)

        X_fake_list = []
        time_list = []
        infer_time_list = []
        for i, (trn_X,) in enumerate(tqdm.tqdm(test_loader)):  # 对testloader遍历
            originData = trn_X.to(self.device)
            start_infer = time.time()
            smashed_data = client_net(originData)  # edge 推理
            
            start = time.time()
            inverted_data = decoder_net(smashed_data)  # inverse
            
            cos = sim_metrics.cosine_similarity(originData,inverted_data).item()
            time_list.append(time.time()-start)
            infer_time_list.append(start-start_infer)

            sim_metrics.sim_metric_dict['cos'].append(cos)
            euc = sim_metrics.euclidean_distance(originData,inverted_data).item()
            sim_metrics.sim_metric_dict['euc'].append(euc)
            mse = sim_metrics.mse_loss(originData,inverted_data).item()
            sim_metrics.sim_metric_dict['mse'].append(mse)

            accuracy,onehotacc,numacc = sim_metrics.rebuild_acc(originData,inverted_data,tab)
            # print("onehotacc,numacc: ",onehotacc,numacc)
            sim_metrics.sim_metric_dict['acc'].append(accuracy)


            X_fake_list.append(inverted_data.cpu().detach().squeeze().numpy())

        # print(f"cosine: {np.mean(sim_metrics.sim_metric_dict['cos'])}, \
        #       Euclidean: {np.mean(sim_metrics.sim_metric_dict['euc'])},\
        #       MSE:{np.mean(sim_metrics.sim_metric_dict['mse'])}")
        sim_metrics.report_similarity()
        print("average time: {}".format(sum(time_list)/len(time_list)),
              "avg infer time:{}".format(sum(infer_time_list)/len(infer_time_list)))

        # 存储数据
        # self.inverse_dir = f'../results/1-8/inverted/{split_layer}/' # 每层一个文件夹
        # 储存similairty相关文件
        # pd.DataFrame({'cos': sim_metrics.sim_metric_dict['cos'],
        #                 'euc': sim_metrics.sim_metric_dict['euc'],
        #                 'mse':sim_metrics.sim_metric_dict['mse']}).to_csv(self.inverse_dir + f'inv-sim.csv', index = False)

        sim_metrics.store_similarity(inverse_route=self.inverse_dir + f'inv-sim.csv')

        # 存储inverse data
        if save_fake:
            pd.DataFrame(X_fake_list).to_csv(self.inverse_dir+f'inv-X.csv', index = False)
    
    def _inverse_image(self,client_net,decoder_net,
                train_loader,test_loader,
                deprocess, # unormalize的函数
                save_fake=True):
        
        # 打印相关信息
        print("----train decoder----")
        print("client_net: ")
        print(client_net)
        print("decoder_net: ")
        print(decoder_net)

        if not os.path.exists(self.inverse_dir+'/images') and save_fake==True: # 创建存储inverted images的文件夹
            os.makedirs(self.inverse_dir+'/images')
            
        # 网络搬到设备上
        client_net.to(self.device)
        decoder_net.to(self.device)
        
        # 记录数据:
        sim_metrics = SimilarityMetrics(type = self.data_type)

        # X_fake_list = []
        time_list = []
        infer_time_list = []

        for i, (trn_X,) in enumerate(tqdm.tqdm(test_loader)):  # 对testloader遍历
            raw_input = trn_X.to(self.device)
            start_infer = time.time()
            smashed_data = client_net(raw_input)  # edge 推理
            start = time.time()
            inverted_input = decoder_net(smashed_data)  # inverse

            # print('raw_input:',raw_input.shape)
            # print('inverted_input:',inverted_input.shape)
            
            deprocessImg_raw = deprocess(raw_input.clone()) # x_n
            deprocessImg_inversed = deprocess(inverted_input.clone()) # s_n

            ssim = sim_metrics.ssim_metric(deprocessImg_raw, deprocessImg_inversed).item()
            time_list.append(time.time()-start)
            infer_time_list.append(start-start_infer)
            sim_metrics.sim_metric_dict['ssim'].append(ssim)
            mse = sim_metrics.mse_loss(raw_input, inverted_input).item()
            sim_metrics.sim_metric_dict['mse'].append(mse)
            euc = sim_metrics.euclidean_distance(raw_input, inverted_input).mean().item()
            sim_metrics.sim_metric_dict['euc'].append(euc)

            # 保存图片
            if save_fake == True: # 储存原始图像+inv图像
                torchvision.utils.save_image(deprocessImg_raw, self.inverse_dir + '/images/' + str(i) + '-ref.png')
                torchvision.utils.save_image(deprocessImg_inversed, self.inverse_dir + '/images/' + str(i) + '-inv.png')
            
        # print(f"SSIM: {np.mean(sim_metrics.sim_metric_dict['ssim'])},\
        #       MSE:{np.mean(sim_metrics.sim_metric_dict['mse'])}")
        sim_metrics.report_similarity()
        print("average time: {}".format(sum(time_list)/len(time_list)),
              "avg infer time:{}".format(sum(infer_time_list)/len(infer_time_list)))
        
        # 储存similairty相关文件
        # pd.DataFrame({'ssim': sim_metrics.sim_metric_dict['ssim'],
        #                 'mse':sim_metrics.sim_metric_dict['mse']}).to_csv(self.inverse_dir + f'inv-sim.csv', index = False)
        sim_metrics.store_similarity(inverse_route=self.inverse_dir + f'inv-sim.csv')



