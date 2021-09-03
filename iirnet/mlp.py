import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from iirnet.base import IIRNet
import iirnet.loss as loss


class MLPModel(IIRNet):
    """Multi-layer perceptron module."""

    def __init__(
        self,
        num_points=512,
        num_layers=2,
        hidden_dim=8192,
        max_order=2,
        normalization="none",
        lr=3e-4,
        **kwargs,
    ):
        super(MLPModel, self).__init__()
        self.save_hyperparameters()

        self.layers = torch.nn.ModuleList()

        for n in range(self.hparams.num_layers):
            in_features = self.hparams.hidden_dim if n != 0 else self.hparams.num_points
            out_features = self.hparams.hidden_dim
            if n + 1 == self.hparams.num_layers:  # no activation at last layer
                linear_layer = torch.nn.Linear(in_features, out_features)
                # my_layer.weight.data.fill_(0.0)
                # my_layer.bias.data.uniform_(0.6, 0.9)
                self.layers.append(linear_layer)
            else:
                self.layers.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_features, out_features),
                        torch.nn.LayerNorm(out_features),
                        torch.nn.LeakyReLU(0.2),
                    )
                )

        n_coef = (self.hparams.model_order // 2) * 6
        self.layers.append(torch.nn.Linear(out_features, n_coef))

        if self.hparams.normalization == "bn":
            self.bn = torch.nn.BatchNorm1d(self.hparams.num_points * 2)

    def forward(self, mag, phs=None):
        x = mag

        for layer in self.layers:
            x = layer(x)

        # reshape into sos format (n_section, (b0, b1, b2, a0, a1, a2))
        n_sections = self.hparams.model_order // 2
        sos = x.view(-1, n_sections, 6)

        # extract gain, poles, and zeros
        g = sos[:, :, 0]
        pole_real = sos[:, :, 1]
        pole_imag = sos[:, :, 2]
        zero_real = sos[:, :, 4]
        zero_imag = sos[:, :, 5]

        # ensure stability
        pole = torch.complex(pole_real, pole_imag)
        pole = pole * torch.tanh(pole.abs()) / pole.abs()

        # ensure zeros inside unit circle
        zero = torch.complex(zero_real, zero_imag)
        zero = zero * torch.tanh(zero.abs()) / zero.abs()

        # Fix filter gain to be 1
        # b0 = torch.ones(g.shape, device=g.device)
        # b1 = -2 * zero_real
        # b2 = ((zero_real ** 2) + (zero_imag ** 2))
        # a0 = torch.ones(g.shape, device=g.device)
        # a1 = -2 * pole.real
        # a2 = (pole.real ** 2) + (pole.imag ** 2)

        # Apply gain g to numerator by multiplying each coefficient by g
        b0 = g
        b1 = g * -2 * zero.real
        b2 = g * ((zero.real ** 2) + (zero.imag ** 2))
        a0 = torch.ones(g.shape, device=g.device)
        a1 = -2 * pole.real
        a2 = (pole.real ** 2) + (pole.imag ** 2)

        # reconstruct SOS
        out_sos = torch.stack([b0, b1, b2, a0, a1, a2], dim=-1)

        return out_sos

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        ms1 = int(self.hparams.max_epochs * 0.8)
        ms2 = int(self.hparams.max_epochs * 0.95)
        milestones = [ms1, ms2]
        print(
            "Learning rate schedule:",
            f"1:{self.hparams.lr:0.2e} ->",
            f"{ms1}:{self.hparams.lr*0.1:0.2e} ->",
            f"{ms2}:{self.hparams.lr*0.01:0.2e}",
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones,
            gamma=0.1,
            verbose=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument("--num_points", type=int, default=512)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--hidden_dim", type=int, default=512)
        parser.add_argument("--model_order", type=int, default=10)
        parser.add_argument("--normalization", type=str, default="none")
        # --- training related ---
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--priority_order", action="store_true")

        return parser
