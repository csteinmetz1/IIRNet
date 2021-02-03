import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

import iirnet.loss as loss
import iirnet.plotting as plotting

class IIRNet(pl.LightningModule):
    """ Base IIRNet module. """
    def __init__(self, **kwargs):
        super(IIRNet, self).__init__()
        self.magfreqzloss = loss.LogMagFrequencyLoss()
        self.complexfreqzloss = loss.ComplexLoss()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        mag, phs, real, imag, sos = batch
        pred_sos = self(mag)
        loss = self.magfreqzloss(pred_sos, sos)

        self.log('train_loss', 
                    loss, 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True, 
                    logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mag, phs, real, imag, sos = batch
        pred_sos = self(mag)
        loss = self.magfreqzloss(pred_sos, sos)

        self.log('val_loss', loss)

        # move tensors to cpu for logging
        outputs = {
            "pred_sos" : pred_sos.cpu(),
            "sos": sos.cpu(),
            "mag"  : mag.cpu()}

        return outputs

    def validation_epoch_end(self, validation_step_outputs):

        pred_sos = torch.split(validation_step_outputs[0]["pred_sos"], 1, dim=0)       
        sos = torch.split(validation_step_outputs[0]["sos"], 1, dim=0)

        #self.logger.experiment.add_image("mag", plotting.plot_compare_response(pred_sos, sos), self.global_step)
        self.logger.experiment.add_image("mag-grid", plotting.plot_response_grid(pred_sos, sos), self.global_step)


    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--num_points', type=int, default=32)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--filter_order', type=int, default=2)
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-3)

        return parser