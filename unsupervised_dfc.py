#from __future__ import print_function
import torch
import torch.nn.init
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from useg.kanezaki.dfc.model import Net
from useg.kanezaki.dfc.data import Dataset

import numpy as np
import os
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything


from hunan import read_dataset, HunanTransform
from useg.utils import getdevice

from pytorch_lightning.callbacks import ModelCheckpoint

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

train_input_files, train_label_files = read_dataset(r'C:/Users/federico/Downloads/Hunan_Dataset/', sensor='s2', split='train')
val_input_files, val_label_files = read_dataset(r'C:/Users/federico/Downloads/Hunan_Dataset/', sensor='s2', split='val')

train_dataset = Dataset(train_input_files, label_files = train_label_files, transform=transform)
val_dataset   = Dataset(val_input_files, label_files = val_label_files, transform=transform)

seed_everything(12345)

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
    logger = TensorBoardLogger(save_dir = 'logs', name = 'dfc')
elif(logger=='NeptuneLogger'):
    logger = NeptuneLogger(prefix = '', api_key=os.environ['NEPTUNE_API_TOKEN'], project=os.environ['NEPTUNE_PROJECT'])

checkpoint_callback = ModelCheckpoint(
    monitor="val/ari", 
    mode="max", 
    filename="hunab-best-epoch{epoch:02d}-ari{val/ari:.2f}",
    auto_insert_metric_name = False,
    #save_last=True
)

trainer = pl.Trainer(
    accelerator=device,
    devices=1,
    max_epochs=epochs, 
    log_every_n_steps=1,
    logger = logger,
    callbacks=[checkpoint_callback]
)
trainer.fit(
    model, 
    train_dataloaders = DataLoader(train_dataset, batch_size = 8, shuffle=True),
    val_dataloaders   = DataLoader(val_dataset, batch_size = 8, shuffle=False)
)