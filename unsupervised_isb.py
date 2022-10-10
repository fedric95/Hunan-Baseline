#from __future__ import print_function
import torch
import torch.nn.init
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from useg.kanezaki.isb.model import Net, CustomLoss
from useg.kanezaki.isb.data import Dataset

import numpy as np
import os
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything


from hunan import read_dataset, HunanTransform
from useg.utils import getdevice


n_clusters = 100
epochs = 1000
learning_rate = 0.1
n_layers = 2
stepsize_sim = 1.0 # 'step size for similarity loss'
stepsize_con = 1.0 # 'step size for continuity loss'
use_gpu = 1
logger = 'TensorBoardLogger'
#logger = 'NeptuneLogger'
transform = HunanTransform()

device = getdevice(use_gpu)

input_files, label_files = read_dataset(r'C:/Users/federico/Downloads/Hunan_Dataset/', sensor='s2', split='train')

seed_everything(12345)

dataset = Dataset(input_files, label_files = label_files, transform=transform)

in_channels = 3
model = Net( 
    in_channels = in_channels, 
    out_channels = n_clusters, 
    n_layers = n_layers, 
    stepsize_sim=stepsize_sim,
    stepsize_con=stepsize_con,
    learning_rate=learning_rate
)

if(logger=='TensorBoardLogger'):
    logger = TensorBoardLogger('logs')
elif(logger=='NeptuneLogger'):
    logger = NeptuneLogger(prefix = '', api_key=os.environ['NEPTUNE_API_TOKEN'], project=os.environ['NEPTUNE_PROJECT'])


trainer = pl.Trainer(
    accelerator=device,
    devices=1,
    max_epochs=epochs, 
    log_every_n_steps=1,
    logger = logger
)
trainer.fit(model, DataLoader(dataset, batch_size = 8, shuffle=True))