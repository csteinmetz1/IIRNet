import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from .base import IIRNet

class MLPModel(IIRNet):
    """ Multi-layer perceptron module. """
    def __init__(self, 
                num_points = 512,
                num_layers = 4,
                hidden_dim = 128,
                filter_order = 2,
                lr = 3e-4,
                **kwargs):
        super(MLPModel, self).__init__()

        self.save_hyperparameters()

        self.layers = torch.nn.ModuleList()

        for n in range(self.hparams.num_layers):
        in_features = self.hparams.hidden_dim if n != 0 else self.hparams.num_points
        out_features = self.hparams.hidden_dim
        self.layers.append(torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.PReLU(),
        ))

        n_coef = (self.hparams.filter_order + 1) * 2
        self.layers.append(torch.nn.Linear(out_features, n_coef))

    def forward(self, mag, phs=None):
        x = mag
        for layer in self.layers:
        x = layer(x) 
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--num_points', type=int, default=512)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--filter_order', type=int, default=2)
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-3)

        return parser