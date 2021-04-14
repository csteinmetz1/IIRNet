import torch
from argparse import ArgumentParser
import pytorch_lightning as pl

from iirnet.data import IIRFilterDataset
from iirnet.base import IIRNet
from iirnet.mlp import MLPModel
from iirnet.lstm import LSTMModel

parser = ArgumentParser()

parser.add_argument('--shuffle', action="store_true")
parser.add_argument('--precompute', action="store_true")
parser.add_argument('--filter_method', type=str, default='char_poly')
parser.add_argument('--max_train_order', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--model_name', type=str, default='mlp', help='mlp or lstm')
parser.add_argument('--num_train_examples', type=int, default=100000)
parser.add_argument('--num_val_examples', type=int, default=10000)

temp_args, _ = parser.parse_known_args()

# let the model add what it wants
if temp_args.model_name == 'mlp':
    parser = MLPModel.add_model_specific_args(parser)
elif temp_args.model_name == 'lstm':
    parser = LSTMModel.add_model_specific_args(parser)

parser = pl.Trainer.add_argparse_args(parser)       # add all the available trainer options to argparse
args = parser.parse_args()                          # parse them args                      


# init the trainer and model 
trainer = pl.Trainer.from_argparse_args(args)

# setup the dataloaders
train_datasetA = IIRFilterDataset(method="gaussian_peaks",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_train_examples,
                               precompute=args.precompute)

train_datasetB = IIRFilterDataset(method="pass",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_train_examples,
                               precompute=args.precompute)

train_datasetC = IIRFilterDataset(method="parametric",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_train_examples,
                               precompute=args.precompute)

train_datasetD = IIRFilterDataset(method="uniform_parametric",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_train_examples,
                               precompute=args.precompute)

train_dataset = torch.utils.data.ConcatDataset([train_datasetA])

train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                             shuffle=args.shuffle,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers)


val_datasetA = IIRFilterDataset(method="gaussian_peaks",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetB = IIRFilterDataset(method="pass",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetC = IIRFilterDataset(method="parametric",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

# this one does not work at the moment
val_datasetD = IIRFilterDataset(method="uniform_parametric",
                               num_points=args.num_points, 
                               max_order=args.max_train_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_dataset = torch.utils.data.ConcatDataset([val_datasetA])

val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=args.shuffle,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers)

# build the model
if args.model_name == 'mlp':
    model = MLPModel(**vars(args))
elif args.model_name == 'lstm':
    model = LSTMModel(**vars(args))

# Run learning rate finder
#lr_finder = trainer.tuner.lr_find(model, train_dataloader, min_lr=1e-08, max_lr=0.01)
# Pick point based on plot, or get suggestion
#new_lr = lr_finder.suggestion()
# update hparams of the model
#model.hparams.lr = 0.001
#print(new_lr)

# train!
trainer.fit(model, train_dataloader, val_dataloader)