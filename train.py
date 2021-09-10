import os
import torch
import numpy as np
from argparse import ArgumentParser
import pytorch_lightning as pl

from iirnet.data import IIRFilterDataset
from iirnet.base import IIRNet
from iirnet.mlp import MLPModel
from iirnet.lstm import LSTMModel

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

pl.seed_everything(13)


def wif(id):  # worker init function
    np.random.seed((id + torch.initial_seed()) % np.iinfo(np.int32).max)


parser = ArgumentParser()

parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--precompute", action="store_true")
parser.add_argument("--filter_method", type=str, default="char_poly")
parser.add_argument("--max_train_order", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--model_name", type=str, default="mlp", help="mlp or lstm")
parser.add_argument("--num_train_examples", type=int, default=100000)
parser.add_argument("--num_val_examples", type=int, default=1000)

temp_args, _ = parser.parse_known_args()

# let the model add what it wants
if temp_args.model_name == "mlp":
    parser = MLPModel.add_model_specific_args(parser)
elif temp_args.model_name == "lstm":
    parser = LSTMModel.add_model_specific_args(parser)

parser = pl.Trainer.add_argparse_args(
    parser
)  # add all the available trainer options to argparse
args = parser.parse_args()  # parse them args

# set the log/checkpoint directory
args.default_root_dir = os.path.join(
    "logs",
    f"{args.experiment_name}",
    f"epochs={args.max_epochs}_filter-method={args.filter_method}_filter-order={args.model_order}_hidden-dim={args.hidden_dim}",
)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    save_last=False,
    filename=f"{args.filter_method}" + "-{epoch:02d}-{step}",
)

learning_rate_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

# init the trainer and model
trainer = pl.Trainer.from_argparse_args(
    args,
    callbacks=[
        checkpoint_callback,
        learning_rate_callback,
    ],
)

# setup the dataloaders
train_datasetA = IIRFilterDataset(
    method="normal_poly",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_train_examples,
    precompute=args.precompute,
)

train_datasetB = IIRFilterDataset(
    method="normal_biquad",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_train_examples,
    precompute=args.precompute,
)

train_datasetC = IIRFilterDataset(
    method="uniform_disk",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_train_examples,
    precompute=args.precompute,
)

train_datasetD = IIRFilterDataset(
    method="uniform_mag_disk",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_train_examples,
    precompute=args.precompute,
)

train_datasetE = IIRFilterDataset(
    method="char_poly",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_train_examples,
    precompute=args.precompute,
)

train_datasetF = IIRFilterDataset(
    method="uniform_parametric",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_train_examples,
    precompute=args.precompute,
)

filter_datasets = {
    "normal_poly": train_datasetA,
    "normal_biquad": train_datasetB,
    "uniform_disk": train_datasetC,
    "uniform_mag_disk": train_datasetD,
    "char_poly": train_datasetE,
    "uniform_parametric": train_datasetF,
    "all": torch.utils.data.ConcatDataset(
        [
            train_datasetA,
            train_datasetB,
            train_datasetC,
            train_datasetD,
            train_datasetE,
            train_datasetF,
        ]
    ),
}

train_dataset = filter_datasets[args.filter_method]

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=args.shuffle,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    worker_init_fn=wif,
    pin_memory=True,
)

val_datasetA = IIRFilterDataset(
    method="normal_poly",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_val_examples,
    precompute=args.precompute,
)

val_datasetB = IIRFilterDataset(
    method="normal_biquad",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_val_examples,
    precompute=args.precompute,
)

val_datasetC = IIRFilterDataset(
    method="uniform_disk",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_val_examples,
    precompute=args.precompute,
)

val_datasetD = IIRFilterDataset(
    method="uniform_mag_disk",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_val_examples,
    precompute=args.precompute,
)

val_datasetE = IIRFilterDataset(
    method="char_poly",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_val_examples,
    precompute=args.precompute,
)

val_datasetF = IIRFilterDataset(
    method="uniform_parametric",
    num_points=args.num_points,
    max_order=args.max_train_order,
    num_examples=args.num_val_examples,
    precompute=args.precompute,
)

filter_val_datasets = {
    "normal_poly": val_datasetA,
    "normal_biquad": val_datasetB,
    "uniform_disk": val_datasetC,
    "uniform_mag_disk": val_datasetD,
    "char_poly": val_datasetE,
    "uniform_parametric": val_datasetF,
    "all": torch.utils.data.ConcatDataset(
        [
            val_datasetA,
            val_datasetB,
            val_datasetC,
            val_datasetD,
            val_datasetE,
            val_datasetF,
        ]
    ),
}

val_dataloader = torch.utils.data.DataLoader(
    filter_val_datasets[args.filter_method],
    shuffle=args.shuffle,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    worker_init_fn=wif,
    pin_memory=True,
)

# build the model
if args.model_name == "mlp":
    model = MLPModel(**vars(args))
elif args.model_name == "lstm":
    model = LSTMModel(**vars(args))

# train!
trainer.fit(model, train_dataloader, val_dataloader)
