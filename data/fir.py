import os
import sys
import glob
import torch
import torchaudio
import scipy.signal

def get_ir_magnitude(filename=None, x=None, num_points=512, n_fft=65536, eps=1e-8):
    
    if filename is not None:
        h, sr = torchaudio.load(filename, normalize=True)
    else:
        h = x
        x /= x.abs().max()

    if h.shape[0] > 1:
        # left channel
        h = h[0,:]

    # zero pad or crop
    if h.shape[-1] < n_fft:
        padsize = (n_fft - h.shape[-1])//2
        h = torch.nn.functional.pad(h, (padsize,padsize))
    else:
        h = h[:n_fft]

    # apply Hann window
    wn = torch.hann_window(h.shape[-1])
    h *= wn

    H = torch.fft.rfft(h, n=n_fft)

    # compute dB magnitude
    mag = 20 * torch.log10(H.abs() + eps)

    mag = torch.nn.functional.interpolate(
        mag.view(1,1,-1), 
        size=num_points+1
    ).view(-1)

    # omit nyquist bin
    mag = mag[:num_points]

    # smooth
    mag = scipy.signal.savgol_filter(mag.numpy(), 41, 2)
    mag = torch.tensor(mag)

    # zero mean
    mag = mag - torch.mean(mag)

    return mag

class FIRFilterDataset(torch.utils.data.Dataset):
    """ Dataset class to load FIR filters. 
    
    """
    def __init__(self, 
            data_dir,
            num_points = 512,
            n_fft = 65536,
            eps = 1e-8
        ):
        super(FIRFilterDataset, self).__init__()

        self.num_points = num_points
        self.n_fft = n_fft
        self.eps = eps

        # get a list of all files 
        self.files = glob.glob(
            os.path.join(data_dir, "*.wav"), 
            recursive=False
        )

        print(f"Located {len(self.files)} IRs.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        mag = get_ir_magnitude(
                filename, 
                num_points=self.num_points, 
                n_fft=self.n_fft,
                eps=self.eps
            )

        return mag, None, None, None, None
