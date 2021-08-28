import os
import torch
import torchaudio
import numpy as np
import scipy.signal

from data.fir import get_ir_magnitude 
from iirnet.mlp import MLPModel
from iirnet.plotting import plot_responses
from iirnet.data import IIRFilterDataset

# Audio examples come from Freesound
# https://freesound.org/people/karolist/sounds/370934/
# https://freesound.org/people/matt141141/sounds/469283/

# Random filter generator
val_datasetE = IIRFilterDataset(method="char_poly",
                               num_points=512, 
                               max_order=8, 
                               num_examples=1,
                               precompute=False)

val_datasetF = IIRFilterDataset(method="uniform_parametric",
                               num_points=512, 
                               max_order=8, 
                               num_examples=1,
                               precompute=False)

# load our target spectrum
target_dB = get_ir_magnitude('data/KCIRs_16bit/001a-SM57-V30-4x12.wav')

# load our guitar signal to filter
x, sr = torchaudio.load('docs/audio/469283__matt141141__cm7-dm7-115bpm-loop.wav')
x *= 0.0125

#x = (torch.rand(1,44100) * 2) - 1

# load our model
ckpt = 'lightning_logs/char_poly/lightning_logs/version_1/checkpoints/epoch=82-step=64905.ckpt'
model = MLPModel.load_from_checkpoint(ckpt)
model.eval()

out = val_datasetE[0]
#target_dB = out[0]
target_sos = out[4]

# predict filter
with torch.no_grad():
    pred_sos = model(target_dB.view(1,1,-1))
pred_sos = pred_sos.view(-1,6)

# apply filter to guitar signal
x_filt = scipy.signal.sosfilt(pred_sos.numpy(), x.view(-1).numpy())
x_filt /= np.max(np.abs(x_filt))

x_filt_target = scipy.signal.sosfilt(target_sos.numpy(), x.view(-1).numpy())
x_filt_target /= np.max(np.abs(x_filt_target))

outfile_iirnet = os.path.join('docs', 'audio', 'x_filt_iirnet.wav')
outfile_target = os.path.join('docs', 'audio', 'x_filt_target.wav')

torchaudio.save(outfile_iirnet, torch.tensor(x_filt).view(1,-1), sr)
torchaudio.save(outfile_target, torch.tensor(x_filt_target).view(1,-1), sr)

# plot responses
filename = os.path.join('docs', 'audio', 'response.png')
plot_responses(pred_sos.detach(), target_dB, filename=filename)
