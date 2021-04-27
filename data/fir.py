import os
import sys
import glob
import torch
import torchaudio
import scipy.signal

class FIRFilterDataset(torch.utils.data.Dataset):
    """ Dataset class to load FIR filters. 
    
    """
    def __init__(self, 
            data_dir,
            num_points = 512,
            eps = 1e-8
        ):
        super(FIRFilterDataset, self).__init__()

        self.num_points = num_points
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

        h, sr = torchaudio.load(filename, normalize=True)

        H = torch.fft.rfft(h, n=self.num_points*2)

        # left channel
        H = H[0,:]

        # omit nyquist bin
        H = H[:self.num_points]

        # compute dB magnitude
        mag = 20 * torch.log10(H.abs() + self.eps)

        # smooth
        mag = scipy.signal.savgol_filter(mag.numpy(), 41, 2)
        mag = torch.tensor(mag)

        # zero mean
        mag = mag - torch.mean(mag)

        return mag, None, None, None, None
