import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

class LSTMModel(pl.LightningModule):
    """ LSTM module. """
    def __init__(self, 
                num_points = 512,
                num_layers = 1,
                hidden_dim = 128,
                filter_order = 2,
                lr = 3e-4,
                **kwargs):
        super(LSTMModel, self).__init__()

        self.save_hyperparameters()

        self.lstm = torch.nn.LSTM(self.hparams.num_points,
                                self.hparams.hidden_dim,
                                self.hparams.num_layers)

        n_coef = (self.hparams.filter_order + 1) * 2
        self.output = torch.nn.Linear(self.hparams.hidden_dim, n_coef)

    def forward(self, mag, phs=None):
        
        out, _ = self.lstm(mag)
        out = self.output(out)

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
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-3)

        return parser