import time
import glob
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from iirnet.signal import mag
from iirnet.mlp import MLPModel
from iirnet.loss import LogMagFrequencyLoss, LogMagTargetFrequencyLoss
from iirnet.filter import generate_characteristic_poly_filter

magloss = LogMagFrequencyLoss()
magtarget = LogMagTargetFrequencyLoss()

def sgd_filter_design(n_iters=10000, sos_init=None):

    # create the biquad coefficients we will optimize
    pred_sos = torch.rand(1,12,6, requires_grad=True)

    if sos_init is not None:
        with torch.no_grad():
            pred_sos.data = sos_init.data

    optimizer = torch.optim.SGD([pred_sos], lr=1e-1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #                                                optimizer, 
    #                                                n_iters)

    tic = time.perf_counter()
    for n in range(n_iters):
        loss = magloss(pred_sos, sos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print(f"{n+1}/{n_iters} MSE: {loss:0.3f} dB")
    
    toc = time.perf_counter()
    #print(f"SGD {n+1}/{n_iters} MSE: {loss:0.3f} dB in {toc-tic:0.2f} sec")
    sgd_elapsed = toc - tic

    return loss.item(), sgd_elapsed

if __name__ == '__main__':

    n_tests = 10
    gpu = False
    sgd_losses = []
    sgd_times = []
    model_losses = []
    model_times = []

    for n in range(n_tests):
        print(n)
        # generate a random filter as the target
        mag, phs, _, _, sos = generate_characteristic_poly_filter(512, 24)
        sos = torch.tensor(sos).unsqueeze(0) # add batch dim
        mag = torch.tensor(mag.astype('float32')).unsqueeze(0)

        # use SGD to fit a cascade of biquads
        sgd_loss, sgd_elapsed = sgd_filter_design()
        sgd_losses.append(sgd_loss)
        sgd_times.append(sgd_elapsed)

        # use a pre-trained IIRNet model to predict

        ckpt_path = glob.glob('lightning_logs/version_11/checkpoints/*.ckpt')[0]
        model = MLPModel.load_from_checkpoint(ckpt_path)
        model.eval()
        
        if gpu:
            mag = mag.to('cuda')
            sos = sos.to('cuda')
            model.to('cuda')
            
        tic = time.perf_counter()
        pred_sos = model(mag)
        toc = time.perf_counter()
        model_elapsed = toc - tic

        loss = magloss(pred_sos, sos)
        model_loss = loss.item()

        model_losses.append(model_loss)
        model_times.append(model_elapsed)
        #print(f"IIRNet MSE: {loss:0.3f} dB in {(toc-tic)*1000:0.1f} ms ({sgd_elapsed/model_elapsed:0.2f}x faster)")
        #sgd_filter_design(sos_init=pred_sos)   

    print(f"Mean    SGD MSE: {np.mean(sgd_losses):0.2f} dB  in {np.mean(sgd_times)*1000:0.1f} ms")
    print(f"Mean IIRNet MSE: {np.mean(model_losses):0.2f} dB  in {np.mean(model_times)*1000:0.1f} ms ({np.mean(sgd_times)/np.mean(model_times):0.2f}x faster)")
