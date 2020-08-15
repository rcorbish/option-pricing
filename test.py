#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import grad
import time

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from netwk import Net 
from dataset import CSVDataset 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def price( net, inputs ) :
    return net(inputs)


def delta( net, inputs ) :
    inputs.requires_grad = True
    pv = price( net, inputs ) 
    pv.backward()
    first_order = grad( pv, inputs, create_graph=True)
    return pv, first_order[0]


def gamma( net, inputs ) :
    inputs.requires_grad = True
    pv = price( net, inputs ) 
    first_order = grad( pv, inputs, create_graph=True )
    second_order = grad( first_order[0][0][2], inputs )
    return pv, first_order[0][0], second_order[0][0]


def test( net, loader ) :

    print( "IDEAL", "PV", "DELTA", "GAMMA", "PV_ERROR" )

    op = []

    # with torch.no_grad() :
    # load & pre-compile the class for actual timing
    for data in loader :
        input, label = data[0], data[1] 
        net(input)
        break 

    start = time.time()
    for data in loader :
        pv, o1, o2 = gamma( net, data[0] )
        op.append( (pv.item(), o1[2].item(), o2[2].item()) )
    elapsed = time.time() - start

    n = 0 
    for data in loader :
        n = n + 1
        label = data[1].item()
        ac = op.pop(0)
        px = ac[0]
        print( "{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.4f}".format( label, px, ac[1], ac[2], abs(px-label) ) )

    print('Done Testing in', int(1000000 * elapsed / n), "uS per px")



if __name__ == "__main__" :

    model = Net().to( device )
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    df_test = pd.read_csv( 'pricing-test.csv' )
    test_labels = df_test.iloc[:, 0]
    test_data = df_test.iloc[:, 1:]

    test_dataset = CSVDataset( test_data, labels=test_labels, device=device ) 
    test_loader = DataLoader(test_dataset, batch_size=1 )

    test( model, test_loader ) 

