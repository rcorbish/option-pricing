#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import time

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from netwk import Net 
from dataset import CSVDataset 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train( net, loader ) :
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=.003 )  
    num_epochs = 50 
    iteration_count = num_epochs # * len(loader) / loader.batch_size
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=num_epochs )
    for e in range( num_epochs ) :
        running_loss = 0
        for data in loader :
            # data pixels and labels to GPU if available
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            labels = labels.squeeze()

            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()
 
            running_loss += loss.item()
        scheduler.step()
        print('Epoch', (e+1), " loss:", running_loss/len(loader) )
 
    print('Done Training')


def test( net, loader ) :

    print( "OUTPUT", "IDEAL" )

    start = time.time()
    n = 0 
    with torch.no_grad() :
        for data in loader :
            n = n + 1
            input, label = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            label = label.squeeze()
            output = net(input)
            print( "{:6.2f} {:6.2f}".format( output.item(), label.item() ) )


    print('Done Testing in', int(1000000 * (time.time() - start) / n), "uS per px")


if __name__ == "__main__" :
    net = Net().to( device ) 

    df_train = pd.read_csv( 'pricing.csv', dtype=np.float32 )
    train_labels = df_train.iloc[:, 0]
    train_data = df_train.iloc[:, 1:]

    train_dataset = CSVDataset( train_data, labels=train_labels ) 
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    train( net, train_loader ) 

    df_test = pd.read_csv( 'pricing-test.csv' )
    test_labels = df_test.iloc[:, 0]
    test_data = df_test.iloc[:, 1:]

    test_dataset = CSVDataset( test_data, labels=test_labels ) 
    test_loader = DataLoader(test_dataset, batch_size=1 )

    test( net, test_loader ) 
