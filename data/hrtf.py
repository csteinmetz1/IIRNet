import os
import sys
import glob
import torch
import torchaudio

class HRTFDataset(torch.utils.data.Dataset):
    """ Dataset class to load HRTF filters. 
    
    """
    def __init__(self, 
            data_dir,
            num_points = 512,
            eps = 1e-8
        ):
        super(HRTFDataset, self).__init__()

        self.num_points = num_points
        self.eps = eps

        # get a list of all HRTF files (compensated)
        self.hrtf_files = glob.glob(
            os.path.join(data_dir, "**", "**", "*C*.wav_16.wav"), 
            recursive=True
        )

        print(f"Located {len(self.hrtf_files)} HRTFs.")

    def __len__(self):
        return len(self.hrtf_files)

    def __getitem__(self, idx):

        filename = self.hrtf_files[idx]

        h, sr = torchaudio.load(filename, normalize=True)

        H = torch.fft.rfft(h, n=self.num_points*2)

        # left channel
        H = H[0,:]

        # compute dB magnitude
        #mag = 20 * torch.log10(H.abs() + self.eps)

        # omit nyquist bin
        H = H[:self.num_points]

        return H
