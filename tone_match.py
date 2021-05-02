import os
import sys
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

clean_gtr = 'docs/audio/469283__matt141141__cm7-dm7-115bpm-loop.wav'
solo_gtr = 'docs/audio/370934__karolist__guitar-solo.wav'
vocal = 'docs/audio/330662__womb-affliction__whimsical-female-vocal.wav'

# ---- Parameters -----

mode = "static"    # 'dyanmic' or 'static'
block_size = 8192  
hop_size = 2048

# --------------------

# load our guitar signal to filter
x, sr = torchaudio.load(vocal)
x *= 0.0125

# load our target guitar tone signal
#y, sr = torchaudio.load('docs/audio/370934__karolist__guitar-solo.wav')
#y *= 0.0125

# get the average spectral magnitude of the target signal
target_dB = get_ir_magnitude(filename=solo_gtr)

# load our model
ckpt = 'lightning_logs/char_poly/lightning_logs/version_3/checkpoints/last.ckpt'
model = MLPModel.load_from_checkpoint(ckpt)
model.eval()

nblocks = int(x.shape[-1] / hop_size)
y = torch.zeros(x.shape)

# analyze the input in overlapping blocks
if mode == 'dynamic':
    for n in range(nblocks):

        start = n * hop_size
        stop  = start + block_size

        block = x[:,start:stop].clone()
        input_dB = get_ir_magnitude(x=block)

        filt_dB = target_dB - input_dB

        # apply Hann window
        wn = torch.hann_window(block.shape[-1])
        block *= wn

        # predict filter
        with torch.no_grad():
            pred_sos = model(filt_dB.view(1,1,-1))
        pred_sos = pred_sos.view(-1,6)

        # apply filter to guitar signal
        x_filt = scipy.signal.sosfilt(pred_sos.numpy(), block.view(-1).numpy())
        x_filt /= np.max(np.abs(x_filt))
        y[:,start:stop] += x_filt

        # plot responses
        sys.stdout.write(f"* Matching {n+1}/{nblocks}...\r")
        sys.stdout.flush()
        #filename = os.path.join(f'docs', 'audio', f'response{n}.png')
        #plot_responses(pred_sos.detach(), filt_dB, filename=filename)
else:

    # get the average spectral magnitude of the input signal
    input_dB = get_ir_magnitude(filename=vocal)

    # compute inverse filter
    filt_dB = target_dB - input_dB

    # predict filter
    with torch.no_grad():
        pred_sos = model(filt_dB.view(1,1,-1))
    pred_sos = pred_sos.view(-1,6)

    # apply filter to guitar signal
    x_filt = scipy.signal.sosfilt(pred_sos.numpy(), x.view(-1).numpy())
    x_filt /= np.max(np.abs(x_filt))

    # plr response
    filename = os.path.join(f'docs', 'audio', f'response_static_match.png')
    plot_responses(pred_sos.detach(), filt_dB, filename=filename)


y /= y.abs().max()
outfile_iirnet = os.path.join('docs', 'audio', 'x_filt_iirnet.wav')
torchaudio.save(outfile_iirnet, torch.tensor(y).view(1,-1), sr)


