from torch.utils.data import DataLoader
from useg.wnet.model import WNet
from useg.wnet.data import Dataset

import pytorch_lightning as pl
import torch
from useg.utils import isimage
import os.path

from hunan import read_dataset, HunanTransform
from useg.utils import getdevice
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from pytorch_lightning.callbacks import ModelCheckpoint

n_clusters = 100
epochs = 100
use_gpu = 1
in_channels = 3
learning_rate_clust = 0.0
learning_rate_recon = 0.03
radius = 5
logger = 'TensorBoardLogger'

transform = HunanTransform()

device = getdevice(use_gpu)

train_input_files, train_label_files = read_dataset(r'C:/Users/federico/Downloads/Hunan_Dataset/', sensor='s2', split='train')
val_input_files, val_label_files = read_dataset(r'C:/Users/federico/Downloads/Hunan_Dataset/', sensor='s2', split='val')

train_dataset = Dataset(train_input_files, label_files = train_label_files, transform=transform)
val_dataset   = Dataset(val_input_files, label_files = val_label_files, transform=transform)

seed_everything(12345)

# Load the arguments
wnet = WNet(in_channels=in_channels, 
    hiddden_dim=8, 
    intermediate_channels=n_clusters, 
    learning_rate_clust=learning_rate_clust, 
    learning_rate_recon=learning_rate_recon, 
    radius=radius
)

if(logger=='TensorBoardLogger'):
    logger = TensorBoardLogger(save_dir = 'logs', name = 'wnet')
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
    wnet, 
    train_dataloaders = DataLoader(train_dataset, batch_size = 4, shuffle=True),
    val_dataloaders   = DataLoader(val_dataset, batch_size = 4, shuffle=False)
)
