import segmentation_models_pytorch as smp
import torch
import pytorch_lightning as pl
import torch.nn as nn
import os.path
from useg.utils import isimage
from useg.kanezaki.dfc.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

from hunan import read_dataset, HunanTransform
from useg.utils import getdevice
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from useg.utils import logimage
import sklearn.metrics


class PLSMPModel(pl.LightningModule):
    def __init__(self, args_smp, out_chanels, class_names = None):
        super(PLSMPModel, self).__init__()
        self.save_hyperparameters()
        self.model = smp.Unet(**args_smp, classes=out_chanels)
        self.out_chanels = out_chanels
        # cross entropy loss definition
        self.loss = nn.CrossEntropyLoss(reduction='none')
        
        self.label_colours = np.random.randint(255,size=(self.out_chanels,3))

    def forward(self, x):
        return(self.model(x))
    
    def _shared_step(self, batch, batch_idx, step):

        image = batch['image']
        label = batch['label'][:, 0, :, :].long()
        mask  = batch['mask'][:, 0, :, :].long()

        output = self(image)
        
        l = self.loss(output, label)*mask
        l = l.sum()/mask.sum()
        


        output = output.argmax(dim=1).cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        

        im_label = label[0, : ,:]
        im_output = output[0, : ,:]


        im_input = np.transpose(image[0].cpu().numpy(),[1,2,0])
        for c in range(im_input.shape[-1]):
            im_input[:, :, c] = (im_input[:, :, c]-im_input[:, :, c].min())/(im_input[:, :, c].max()-im_input[:, :, c].min())
        

        logimage(self.logger, step+'/input',  im_input,  self.global_step)
        logimage(self.logger, step+'/output', im_output, self.global_step, self.label_colours)
        logimage(self.logger, step+'/label',  im_label,  self.global_step, self.label_colours)
        logimage(self.logger, step+'/mask',   np.expand_dims(mask[0, :, :], -1),  self.global_step)


        output = output[mask==1]
        label = label[mask==1]

        f1 = sklearn.metrics.f1_score(label, output, labels = range(0, self.out_chanels), average=None)
        accuracy = sklearn.metrics.accuracy_score(label, output)
        
        for i in range(self.out_chanels):
            self.log(step+'/f1_'+str(i), f1[i], prog_bar=False, on_epoch=True)
        self.log(step+'/f1', f1.mean(), prog_bar=True, on_epoch=True)
        self.log(step+'/accuracy', accuracy, prog_bar=True, on_epoch=True)
        
        return(l)


    def training_step(self, batch, batch_idx):
        return(self._shared_step(batch, batch_idx, step='train'))
    def validation_step(self, batch, batch_idx):
        return(self._shared_step(batch, batch_idx, step='val'))

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=0.001)
        return(opt)

args_smp = {
    'encoder_name': 'resnet18',
    'encoder_weights': 'imagenet',
    'in_channels': 3
}
use_gpu = 1
epochs = 1000
logger = 'TensorBoardLogger'
#logger = 'NeptuneLogger'
out_chanels = 4
transform = HunanTransform()

device = getdevice(use_gpu)

train_input_files, train_label_files = read_dataset(r'C:/Users/federico/Downloads/Hunan_Dataset/', sensor='s2', split='train')
val_input_files, val_label_files = read_dataset(r'C:/Users/federico/Downloads/Hunan_Dataset/', sensor='s2', split='val')

train_dataset = Dataset(train_input_files, label_files = train_label_files, transform=transform)
val_dataset   = Dataset(val_input_files, label_files = val_label_files, transform=transform)


model = PLSMPModel(args_smp = args_smp, out_chanels=out_chanels)



if(logger=='TensorBoardLogger'):
    logger = TensorBoardLogger(save_dir = 'logs', name = 'supervised')
elif(logger=='NeptuneLogger'):
    logger = NeptuneLogger(prefix = '', api_key=os.environ['NEPTUNE_API_TOKEN'], project=os.environ['NEPTUNE_PROJECT'])

trainer = pl.Trainer(
    accelerator=device,
    devices=1,
    max_epochs=epochs, 
    log_every_n_steps=1,
    logger = logger
)
trainer.fit(
    model, 
    train_dataloaders = DataLoader(train_dataset, batch_size = 16, shuffle=True),
    val_dataloaders   = DataLoader(val_dataset, batch_size = 16, shuffle=False)
)