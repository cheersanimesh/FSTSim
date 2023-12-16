import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import time
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split
import numpy as np

from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data import random_split, DataLoader, ConcatDataset
import uuid
import argparse

from models.Client import Client
from models.Server import Server, Metrics
from Datafetcher import AugmentedDataset, get_datasets


if(torch.cuda.is_available()):
    device = torch.device('cuda')
    num_devices = torch.cuda.device_count()
    print(f"Using GPU, num: {num_devices}")
else:
    device = torch.device('cpu')
    num_devices = torch.cpu.device_count()
    print(f"Using CPU, num: {num_devices}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required = True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--teacher_backbone', type=str, required = True)
    parser.add_argument('--student_backbone', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--label_percent', type = str, required = True)
    parser.add_argument('--num_rounds', type=int, required = True)

    return parser.parse_args()

def get_model(mode='resnet_18'):
    if(mode=='resnet_18'):
        model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        inftrs = model.fc.in_features
        model.fc = nn.Sequential(
                  nn.Linear(inftrs, num_classes, bias= True),
                  nn.Softmax(dim= 0)
                )
    elif(mode=='resnet_50'):
        model = torchvision.models.resnet50(weights=None, num_classes= num_classes)
        inftrs = model.fc.in_features
        model.fc = nn.Sequential(
              nn.Linear(inftrs, num_classes, bias = True),
              nn.Softmax(dim=0)
        )
    if(dataset =='mnist'):
        model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7), stride=(2,2), padding=(3,3), bias = False )
    return model

curr_device= 0
def getDevice():

    if(not torch.cuda.is_available()):
        return torch.device('cpu')
    num_devices = torch.cuda.device_count()

    global curr_device
    dev_string = f"cuda:{curr_device}"

    curr_device = (curr_device+1) % num_devices

    return torch.device(dev_string)


if __name__=='__main__':

    args = parse_arguments()

    session_id = str(uuid.uuid4())

    batch_size = args.batch_size
    num_clients = args.num_clients
    drop_label_percent = args.label_percent
    dataset = args.dataset
    num_rounds = args.num_rounds
    teacher_string = args.teacher_backbone
    student_string = args.student_backbone

    environment_vars = {
        "batch_size":batch_size,
        "num_clients":num_clients,
        "label_percent":drop_label_percent,
        "dataset":dataset,
        "teacher_backbone":teacher_model_string,
        "student_backbone":student_model_string
    }

    file_string=session_id

    for k in environment_vars:
        temp_string = f"_{k}_{environment_vars[k]}"
        file_string = file_string+temp_string
    
    federated_datasets = get_datasets(n=5, drop_label_percent=drop_label_percent, augment=True)

    # Accessing the first federated dataset
    labeled_data, unlabeled_data = federated_datasets[0]

    data_segregated= get_datasets(n=num_clients+1,drop_label_percent = drop_label_percent)

    server = Server( get_model(teacher_model_string), get_model(student_model_string), labeled_data=data_segregated[0][0], unlab_data= data_segregated[0][1], device = getDevice() )

    for i in range(num_clients):

        client = Client( i, get_model(teacher_model_string), get_model(student_model_string), labset= data_segregated[i+1][0], unlabset = data_segregated[i+1][1], device= getDevice() )
        server.addClient(client)
    

    for round in range(num_rounds):
        print(f"Starting Round {round}")
        start_time = time.time()
        for client_idx in range(num_clients):
            client = server.client_list[client_idx]
            client.training_loop(round)
            server.updateClient(client_idx, client=client)
        server.aggregate_student()
        server.aggregate_teacher()
        teacher_test_loss, teacher_test_acc = server.compute_loss_accuracy(server.global_teacher_model, nn.CrossEntropyLoss(), testset)
        teacher_train_loss, teacher_train_acc = server.compute_loss_accuracy(server.global_teacher_model, nn.CrossEntropyLoss(), server.labeled_data )
        student_train_loss, student_train_acc = server.compute_loss_accuracy(server.global_student_model, nn.CrossEntropyLoss(), server.labeled_data )
        student_test_loss, student_train_acc = server.compute_loss_accuracy(server.global_student_model, nn.CrossEntropyLoss(), testset)
        time_elap = time.time() - start_time
        metric_val= {
            'round': round,
            'teacher_test_loss':teacher_test_loss,
            'teacher_test_acc': teacher_test_acc,
            'teacher_train_loss':teacher_train_acc,
            'teacher_train_acc':teacher_train_acc,
            'student_train_loss':student_train_loss,
            'student_train_acc':student_train_acc,
            'student_test_loss':student_test_loss,
            'student_train_acc':student_train_acc,
            'time_elapsed':time_elap
            }
        server.metrics.insert(0,metric_val)


        server.broadcastStudent()
        server.broadcastTeacher()
        print(metric_val)
        print(f"Ending Round {round}")
    
    
    
    server.save_model(file_string)

    for i in range(num_clients):
        server.client_list[i].save_model(file_string)



