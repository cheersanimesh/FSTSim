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

class Client:

    def __init__(self, id, student_model,teacher_model, labset, unlabset, device, testset, batch_size= 128):
        self.id =id
        self.student_model= student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.labset = labset
        self.device =device
        self.unlabset= unlabset
        self.batch_size= batch_size
        self.metrics =[]
        self.testset = testset
        

    def genererate_psuedo_labels(self):

        top_k = 200
        unlab_loader = torch.utils.data.DataLoader(self.unlabset, batch_size=self.batch_size, num_workers=2)
        with torch.no_grad():
          #model_outputs = self.teacher_model(self.unlabset)
          images =[]
          model_outputs=[]
          for (data, _) in unlab_loader:
               data = data.to(self.device)
               model_output = self.teacher_model(data)
               model_outputs.append(model_output)
               images.append(data)
          unlab_images = torch.cat(tuple(images), dim =0)
          images_probs= torch.cat(tuple(model_outputs), dim =0)
          top_k_probabilities, top_k_indices = torch.topk(images_probs, top_k, dim=0)
          new_dataset=[]
          for row in range(top_k):
              for lab in range(10):  ##num_classes
                  image_idx = top_k_indices[row][lab]
                  new_dataset.append( (unlab_images[image_idx], lab) )
        pseudo_lab_dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=self.batch_size, shuffle =True, drop_last=True)
        
        return pseudo_lab_dataloader



    def train_teacher(self, lss_fn):
        optimizer = optim.Adam(self.teacher_model.parameters())

        num_epochs = 50
        trainloader = torch.utils.data.DataLoader(self.labset, batch_size= self.batch_size, num_workers=2)
        print(f"Training Teacher in Client id:{self.id} for {num_epochs} epochs")
        for epch in range(num_epochs):
            for data, labels in trainloader:
                print(data.shape)
                if(data.shape[0]==1):
                  continue
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = self.teacher_model(data)
                loss= lss_fn(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Completed Epoch {epch}")
        print(f"End train Teacher in Cliend for id:{self.id} for {num_epochs} epochs")

    def train_student(self, lss_fn, dataloader):
        optimizer = optim.Adam(self.student_model.parameters())
        num_epochs = 50
        for epch in range(num_epochs):
            for (data, labels) in dataloader:
                if(data.shape[0]==1):
                  #print("hit")
                  continue
                data = data.to(self.device)
                labels = labels.to(self.device)
                #print(data.shape)
                output = self.teacher_model(data)
                loss= lss_fn(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Completed Epoch {epch}")


    def finetune_student(self, lss_fn):
        optimizer = optim.Adam(self.student_model.parameters())
        num_epochs = 10
        trainloader = torch.utils.data.DataLoader(self.labset, batch_size= self.batch_size, num_workers=2)
        print(f"Fine Tune Client id:{self.id} for {num_epochs} epochs")
        for epch in range(num_epochs):
            for data, labels in trainloader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                output = self.teacher_model(data)
                loss= lss_fn(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Completed Epoch {epch}")
        print(f"End Fine Tune Client id:{self.id} for {num_epochs} epochs")

    def training_loop(self, round):

        print(f"Starting Training for Client id= {self.id}")

        start_time = time.time()
        lss_fn = nn.CrossEntropyLoss()
        self.train_teacher(lss_fn)

        print("Generating Psuedo label")
        pseudo_lab_dataloader = self.genererate_psuedo_labels()

        print("Training Student model against Pseudolabels")
        self.train_student(lss_fn, pseudo_lab_dataloader)

        print("Fine Tuning Studet Model")
        self.finetune_student(lss_fn=lss_fn)

        teacher_test_loss, teacher_test_acc = self.compute_loss_accuracy(self.teacher_model, nn.CrossEntropyLoss(), self.testset)
        teacher_train_loss, teacher_train_acc = self.compute_loss_accuracy(self.teacher_model, nn.CrossEntropyLoss(), self.labset)
        student_train_loss, student_train_acc = self.compute_loss_accuracy(self.student_model, nn.CrossEntropyLoss(), self.labset)
        student_test_loss, student_test_acc = self.compute_loss_accuracy(self.student_model, nn.CrossEntropyLoss(), self.testset)
        time_elap = time.time() - start_time
        metric_val= {
            'round': round,
            'teacher_test_loss':teacher_test_loss,
            'teacher_test_acc': teacher_test_acc,
            'teacher_train_loss':teacher_train_loss,
            'teacher_train_acc':teacher_train_acc,
            'student_train_loss':student_train_loss,
            'student_train_acc':student_train_acc,
            'student_test_loss':student_test_loss,
            'student_test_acc':student_test_acc,
            'time_elapsed':time_elap
            }
        self.metrics.insert(0,metric_val)

        print(f"Ending Training for Client id= {self.id}")

    ## to change

    def save_model(self, file_string):
        with open(f"Results/client_id={self.id}_{file_string}","wb") as f:
            pickle.dump(self,f)
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

