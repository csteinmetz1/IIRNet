import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from .plotting import plot_compare_response

class IIRNet(pl.LightningModule):
    """ Base IIRNet module. """
    def __init__(self, **kwargs):
        super(IIRNet, self).__init__()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        mag, phs, coef = batch
        pred_coef = self(mag)
        loss = torch.nn.functional.mse_loss(pred_coef, coef)

        self.log('train_loss', 
                    loss, 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True, 
                    logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mag, phs, coef = batch
        pred_coef = self(mag)
        loss = torch.nn.functional.mse_loss(pred_coef, coef)
        
        self.log('val_loss', loss)

        # move tensors to cpu for logging
        outputs = {
            "pred_coef" : pred_coef.cpu().numpy(),
            "coef": coef.cpu().numpy(),
            "mag"  : mag.cpu().numpy()}

        return outputs

    def validation_epoch_end(self, validation_step_outputs):
        # flatten the output validation step dicts to a single dict
        outputs = res = {k: v for d in validation_step_outputs for k, v in d.items()} 

        pred_coef = outputs["pred_coef"][0]
        coef = outputs["coef"][0]

        self.logger.experiment.add_image("mag", plot_compare_response(pred_coef, coef), self.global_step)

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