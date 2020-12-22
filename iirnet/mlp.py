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
                max_order = 10,
                normalization = "none",
                lr = 3e-4,
                **kwargs):
        super(MLPModel, self).__init__()

        self.save_hyperparameters()

        self.layers = torch.nn.ModuleList()

        for n in range(self.hparams.num_layers):
            in_features = self.hparams.hidden_dim if n != 0 else self.hparams.num_points*2
            out_features = self.hparams.hidden_dim
            if n+1 == self.hparams.num_layers: # no activation at last layer
                self.layers.append(torch.nn.Sequential(
                    torch.nn.Linear(in_features, out_features),
                ))
            else:
                self.layers.append(torch.nn.Sequential(
                    torch.nn.Linear(in_features, out_features),
                    torch.nn.PReLU(),
                ))

        n_coef = (self.hparams.max_order//2) * 6
        self.layers.append(torch.nn.Linear(out_features, n_coef))

        if self.hparams.normalization == "bn":
            self.bn = torch.nn.BatchNorm1d(self.hparams.num_points*2)

    def forward(self, real, imag):
        x = torch.cat((real, imag), dim=-1)

        if self.hparams.normalization == "tanh":
            x = torch.tanh(x)
        elif self.hparams.normalization == "bn":
            x = self.bn(x)

        for layer in self.layers:
            x = layer(x) 

        # reshape into sos format (n_section, (b0, b1, b2, a0, a1, a2))
        n_sections = self.hparams.max_order//2
        x = x.view(-1,n_sections,6)

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
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--max_order', type=int, default=10)
        parser.add_argument('--normalization', type=str, default="none")
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-3)

        return parser