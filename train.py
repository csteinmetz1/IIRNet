import torch
from argparse import ArgumentParser
import pytorch_lightning as pl

from iirnet.data import IIRFilterDataset
from iirnet.model import IIRNet

parser = ArgumentParser()

parser.add_argument('--shuffle', action="store_true")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)

parser = IIRNet.add_model_specific_args(parser)     # add model specific args
parser = pl.Trainer.add_argparse_args(parser)       # add all the available trainer options to argparse
args = parser.parse_args()                          # parse them args                      

num_examples = 10000
num_points = 64

# init the trainer and model 
trainer = pl.Trainer(max_epochs=10, auto_lr_find=False)

# setup the dataloaders
train_dataset = IIRFilterDataset(num_points=num_points, num_examples=num_examples)
train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               shuffle=args.shuffle,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)

# build the model
model = IIRNet(num_points=num_points)

# Run learning rate finder
#lr_finder = trainer.tuner.lr_find(model, train_dataloader, min_lr=1e-08, max_lr=0.01)
# Pick point based on plot, or get suggestion
#new_lr = lr_finder.suggestion()
# update hparams of the model
model.hparams.lr = 0.001
#print(new_lr)

# train!
trainer.fit(model, train_dataloader, train_dataloader)