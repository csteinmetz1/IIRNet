import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from .plotting import plot_compare_response

class IIRNet(pl.LightningModule):
  """ Multi-layer perceptron module. """
  def __init__(self, 
               num_points = 512,
               num_layers = 4,
               hidden_features = 128,
               filter_order = 2,
               lr = 3e-4,
               **kwargs):
    super(IIRNet, self).__init__()

    self.save_hyperparameters()

    self.layers = torch.nn.ModuleList()

    for n in range(self.hparams.num_layers):
      in_features = self.hparams.hidden_features if n != 0 else self.hparams.num_points
      out_features = self.hparams.hidden_features
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

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

  # add any model hyperparameters here
  @staticmethod
  def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # --- model related ---
    parser.add_argument('--num_points', type=int, default=512)
    parser.add_argument('--filter_order', type=int, default=2)

    # --- training related ---
    parser.add_argument('--lr', type=float, default=1e-3)

    return parser