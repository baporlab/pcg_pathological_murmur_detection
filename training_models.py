import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary
from load_models import *
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def training_models(train, valid, save_path = '/model_save_folder/',
                    gpu_num = 2, random_seed = 42, num_epochs = 200, batch_size = 128, learning_rate = 0.001, patience = 50,
                    n_ch = 1, length = 2000,
                    model_name = 'vgg16'):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    rs = random_seed
    torch.manual_seed(rs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name = model_name, n_ch = n_ch).to(device)
    print(torchsummary.summary(model, (n_ch,length)))
    print(model(torch.zeros((1,n_ch,length)).to(device)))

    ep = num_epochs
    bs = batch_size
    lr = learning_rate
    pa = patience
    loss_function = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)

    train_loader = torch.utils.data.DataLoader(dataset = train,
                                              batch_size = bs,
                                              shuffle = True,
                                              drop_last = True)
    valid_loader = torch.utils.data.DataLoader(dataset = valid,
                                              batch_size = bs,
                                               shuffle = True,
                                              drop_last = True)

    # training
    train_loss_list = []
    train_metric_list = []
    
    valid_loss_list = []
    valid_metric_list = []
    
    stop_count = 0
    model_save_path = save_path
    for epoch in range(num_epochs): # interation

        model.train()
        running_loss = 0.0
        running_metric = 0.0
        # auroc = binary_auroc()
        for current_x, current_y in tqdm(train_loader): # mini batch training
            inputs = current_x.to(device).float()
            labels = current_y.to(device).float()

            optimizer.zero_grad()
            # forward + backward
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # loss, metric
            running_loss += loss.item()/len(train_loader)
            # auroc.update([labels, outputs])
            running_metric += roc_auc_score(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())/len(train_loader)
        train_loss_list.append(running_loss)
        train_metric_list.append(running_metric.item())
        # Training error
        print('Epoch: ', str(epoch + 1))
        print('Loss: ', running_loss)
        print('AUROC: ', running_metric)

        # Validation error
        model.eval()
        valid_loss = 0.0
        valid_metric = 0.0
        # auroc = binary_auroc()
        with torch.no_grad():
            for current_x, current_y in tqdm(valid_loader): # mini batch training
                inputs = current_x.to(device).float()
                labels = current_y.to(device).float()
                outputs = model(inputs)
                eval_loss = loss_function(outputs, labels)
                # loss, metric
                valid_loss += eval_loss.item()/len(valid_loader)
                # auroc.update([labels, outputs])
                valid_metric += roc_auc_score(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy())/len(valid_loader)
        valid_loss_list.append(valid_loss)
        valid_metric_list.append(valid_metric.item())
        if valid_metric < np.max(valid_metric_list):
            stop_count = stop_count + 1
        if valid_metric >= np.max(valid_metric_list):
            stop_count = 0
            torch.save(model.state_dict(), model_save_path + 'model.pt')
            print('model_saved!')
        if stop_count == pa:
            break
        print('Epoch: ', str(epoch + 1))
        print('Val_Loss: ', valid_loss)
        print('Val_AUROC: ', valid_metric)
        
    print('Finished Training')
    return train_loss_list, train_metric_list, valid_loss_list, valid_metric_list
