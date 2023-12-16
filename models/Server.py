import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import time
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data import random_split, DataLoader, ConcatDataset
import uuid
import argparse

class Metrics:

    def __init__(self, model_type, actual_model, stage, epoch):

        self.model_type= model_type
        self.stage= stage
        self.epoch= epoch
        self.actual_model = actual_model
        self.metrics={}

    def updateMetric(self, key, value):
        self.metrics[key]= value

    def addClientId(self, id):
        self.client_id = id

class Server:

    def __init__(self, teacher_model,student_model,labeled_data, unlab_data, device, testset):
        self.global_student_model= student_model.to(device)
        self.global_teacher_model = teacher_model.to(device)
        self.device= device
        self.client_list=[]
        self.labeled_data = labeled_data
        self.unlab_data = unlab_data
        self.batch_size = 128
        self.metrics=[]
        self.testset = testset

    def addClient(self, client):
        self.client_list.append(client)

    def updateClient(self, idx, client):
        self.client_list[idx]= client

    def broadcastTeacher(self):

        for idx, client in enumerate(self.client_list):
            self.client_list[idx].teacher_model = self.global_teacher_model

    def broadcastStudent(self):
          for idx, client in enumerate(self.client_list):
            self.client_list[idx].student_model = self.global_student_model

    def aggregate_teacher(self):

        with torch.no_grad():
            teacher_model_lists =list()
            for client in self.client_list:
                teacher_model_lists.append(client.teacher_model.parameters())
            res=list()
            for client_layers in zip(*teacher_model_lists):
              layer_list= list()
              for client_idx in range(len(self.client_list)):
                layer_list.append( self.policy_multiplier(client_idx)*(client_layers[client_idx].data) )
              layer_tensor = torch.stack(layer_list)
              res.append(torch.mean(layer_tensor, dim =0))

            print(len(res))
            print("___________RES________")
            for i in res:
                print(i.shape)
            print("________RES____END_______")
            for i in self.global_student_model.parameters():
                    print(i.shape)
            for idx, param in enumerate(self.global_teacher_model.parameters()):
                param.data = nn.parameter.Parameter(res[idx])

    def aggregate_student(self):

         with torch.no_grad():
            student_model_lists =list()
            for client in self.client_list:
                student_model_lists.append(client.student_model.parameters())
            res=list()
            for client_layers in zip(*student_model_lists):
              layer_list= list()
              for client_idx in range(len(self.client_list)):
                layer_list.append( (client_layers[client_idx].data)*self.policy_multiplier(client_idx) )
              layer_tensor = torch.stack(layer_list)
              res.append(torch.mean(layer_tensor, dim =0))

            for idx, param in enumerate(self.global_student_model.parameters()):
                param.data = nn.parameter.Parameter(res[idx])

    def policy_multiplier(self, client_id,policy='iden'):

        if(policy=='iden'):
            return 1.0
        elif(policy=='teacher_test_acc'):
            return self.client_list[client_id].metrics['teacher_test_acc']
        elif(policy=='student_test_acc'):
            return self.client_list[client_id].metrics['student_test_acc']
    ## to change
    def compute_loss_accuracy(self,model, loss_fn, dataset):
        num_instances=0
        num_matched=0
        total_loss=0.0
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=2)
        for index, (data, labels) in enumerate(dataloader):
          data= data.to(self.device)
          labels= labels.to(self.device)
          y_pred= model(data)
          with torch.no_grad():
            loss = loss_fn(y_pred, labels)
            total_loss += loss.cpu().detach().numpy()

          _, predicted = torch.max(y_pred.data, 1)

          num_instances += labels.size(0)
          num_matched += (predicted == labels).sum().item()
          num_instances+=1

        return (total_loss, num_matched/num_instances)

    def save_model(self, file_name):
        with open(f'Results/server_{file_name}', 'wb') as f:
            pickle.dump(self,f)