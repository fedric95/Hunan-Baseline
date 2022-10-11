from torch.utils.data import DataLoader
import pytorch_lightning as pl

from useg.kanezaki.isb.model import Net
from useg.kanezaki.isb.data import Dataset

import os
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything


from hunan import read_dataset, HunanTransform, presegment
from useg.utils import getdevice

from pytorch_lightning.callbacks import ModelCheckpoint

n_clusters = 100
epochs = 1000
learning_rate = 0.001
n_layers = 2
in_channels = 3
use_gpu = 1
logger = 'TensorBoardLogger'
transform = HunanTransform()
segmentation_force = False
segmentation_args = {'compactness': 10, 'n_segments': 100, 'start_label':0, 'multichannel': True}

device = getdevice(use_gpu)

input_files, label_files = read_dataset(r'C:/Users/federico/Downloads/Hunan_Dataset/', sensor='s2', split='train')



train_input_files, train_label_files = read_dataset(r'C:/Users/federico/Downloads/Hunan_Dataset/', sensor='s2', split='train')
train_preseg_files = presegment(train_input_files, 'C:/Users/federico/Downloads/Hunan_Dataset/train/preseg/', segmentation_args, force=segmentation_force)


val_input_files, val_label_files = read_dataset(r'C:/Users/federico/Downloads/Hunan_Dataset/', sensor='s2', split='val')
val_preseg_files = presegment(val_input_files, 'C:/Users/federico/Downloads/Hunan_Dataset/val/preseg/', segmentation_args, force=segmentation_force)

train_dataset = Dataset(train_input_files,  preseg_files=train_preseg_files, label_files = train_label_files, transform=transform)
val_dataset   = Dataset(val_input_files,  preseg_files=val_preseg_files, label_files = val_label_files, transform=transform)

seed_everything(12345)

model = Net( 
    in_channels = in_channels, 
    out_channels = n_clusters, 
    n_layers = n_layers, 
    learning_rate=learning_rate
)


if(logger=='TensorBoardLogger'):
    logger = TensorBoardLogger(save_dir = 'logs', name = 'isb')
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
    train_dataloaders = DataLoader(train_dataset, batch_size = 4, shuffle=True),
    val_dataloaders   = DataLoader(val_dataset, batch_size = 4, shuffle=False)
)
