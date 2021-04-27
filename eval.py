import sys
import time
import torch
import pickle
import scipy.signal
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from collections import defaultdict

from data.hrtf import HRTFDataset
from data.fir import FIRFilterDataset
from iirnet.mlp import MLPModel
from iirnet.sgd import SGDFilterDesign
from iirnet.data import IIRFilterDataset
from iirnet.loss import LogMagTargetFrequencyLoss
from iirnet.plotting import plot_responses
import iirnet.signal as signal

pl.seed_everything(12)

# fixed evaluation parameters
eps = 1e-8
gpu = False
num_points = 512
max_eval_order = 32
examples_per_method = 50
precompute = True
shuffle = False

# error metrics
mag_loss = LogMagTargetFrequencyLoss(priority=False)

# prepare testing datasets
val_datasetA = IIRFilterDataset(method="normal_poly",
                               num_points=num_points, 
                               max_order=max_eval_order, 
                               num_examples=examples_per_method,
                               precompute=precompute)

val_datasetB = IIRFilterDataset(method="normal_biquad",
                               num_points=num_points, 
                               max_order=max_eval_order, 
                               num_examples=examples_per_method,
                               precompute=precompute)

val_datasetC = IIRFilterDataset(method="uniform_disk",
                               num_points=num_points, 
                               max_order=max_eval_order, 
                               num_examples=examples_per_method,
                               precompute=precompute)

val_datasetD = IIRFilterDataset(method="uniform_mag_disk",
                               num_points=num_points, 
                               max_order=max_eval_order, 
                               num_examples=examples_per_method,
                               precompute=precompute)

val_datasetE = IIRFilterDataset(method="char_poly",
                               num_points=num_points, 
                               max_order=max_eval_order, 
                               num_examples=examples_per_method,
                               precompute=precompute)

val_datasetF = IIRFilterDataset(method="uniform_parametric",
                               num_points=num_points, 
                               max_order=max_eval_order, 
                               num_examples=examples_per_method,
                               precompute=precompute)

val_dataset = torch.utils.data.ConcatDataset([
  val_datasetA,
  val_datasetB, 
  val_datasetC, 
  val_datasetD,
  val_datasetE, 
  val_datasetF]
  )

val_guitar_cab_datatset = FIRFilterDataset("data/KCIRs_16bit")
val_hrtf_datatset = FIRFilterDataset("data/HRTF/IRC_1059/COMPENSATED/WAV/IRC_1059_C")

datasets = {
    "normal_poly"       : val_datasetA,
    "normal_biquad"     : val_datasetB,
    "uniform_disk"      : val_datasetC,
    "uniform_mag_disk"  : val_datasetD,
    "char_poly"         : val_datasetE,
    "uniform_parametric": val_datasetF,
    "guitar_cab"        : val_guitar_cab_datatset,
    "hrtf"              : val_hrtf_datatset
}

# model checkpoint paths
normal_poly_ckpt        = 'lightning_logs/normal_poly/lightning_logs/version_0/checkpoints/epoch=81-step=64123.ckpt'
normal_biquad_ckpt      = 'lightning_logs/normal_biquad/lightning_logs/version_0/checkpoints/epoch=70-step=55521.ckpt'
uniform_disk_ckpt       = 'lightning_logs/uniform_disk/lightning_logs/version_0/checkpoints/epoch=89-step=70379.ckpt'
uniform_mag_disk_ckpt   = 'lightning_logs/uniform_mag_disk/lightning_logs/version_0/checkpoints/epoch=28-step=22677.ckpt'
char_poly_ckpt          = 'lightning_logs/char_poly/lightning_logs/version_0/checkpoints/epoch=46-step=36753.ckpt'
uniform_parametric_ckpt = 'lightning_logs/normal_poly/lightning_logs/version_0/checkpoints/epoch=81-step=64123.ckpt'
all_ckpt                = 'lightning_logs/normal_poly/lightning_logs/version_0/checkpoints/epoch=81-step=64123.ckpt'

# load models from disk
models = {
    "SGD (1)"           : SGDFilterDesign(n_iters=1),
    "SGD (10)"          : SGDFilterDesign(n_iters=10),
    "SGD (100)"         : SGDFilterDesign(n_iters=100),
    #"SGD (1000)"        : SGDFilterDesign(n_iters=1000),
    "normal_poly"       : MLPModel.load_from_checkpoint(normal_poly_ckpt),
    "normal_biquad"     : MLPModel.load_from_checkpoint(normal_biquad_ckpt),
    "uniform_disk"      : MLPModel.load_from_checkpoint(uniform_disk_ckpt),
    "uniform_mag_disk"  : MLPModel.load_from_checkpoint(uniform_mag_disk_ckpt),
    "char_poly"         : MLPModel.load_from_checkpoint(char_poly_ckpt),
    "uniform_parametric": MLPModel.load_from_checkpoint(uniform_parametric_ckpt),
    "all"               : MLPModel.load_from_checkpoint(all_ckpt),
}

if gpu:
    model.to("cuda")

def evaluate_on_dataset(
                model, 
                dataset, 
                model_name=None, 
                dataset_name=None, 
                plot=True
            ):

    errors = []
    timings = []
    for idx, example in enumerate(dataset, 0):

        target_dB, phs, real, imag, sos = example

        if gpu: target_dB = target_dB.to("cuda")

        # predict filter coeffieicnts (do timings here)
        tic = time.perf_counter()
        with torch.no_grad():
            pred_sos = model(target_dB.view(1,1,-1))
        toc = time.perf_counter()
        elapsed = toc - tic

        # compute response of the predicted filter
        w, input_h = signal.sosfreqz(pred_sos, worN=target_dB.shape[-1])
        input_dB  = 20 * torch.log10(signal.mag(input_h) + eps)
        input_dB = input_dB.squeeze()
        input_dB = input_dB.cpu().squeeze()
        target_dB = target_dB.cpu().squeeze()

        error = torch.nn.functional.mse_loss(input_dB, target_dB)
        errors.append(error.item())
        timings.append(elapsed)

        if plot:
            filename = f"{model_name}-{dataset_name}-{idx}"
            plot_responses(pred_sos.detach(), target_dB, filename=filename)

        sys.stdout.write(f"* {idx+1}/{len(dataset)}: MSE: {np.mean(errors):0.2f} dB  Time: {np.mean(elapsed)*1e3:0.2f} ms\r")
        sys.stdout.flush()

    print()
    return errors, elapsed

results = defaultdict(dict)

for model_name, model in models.items():

    print("-" * 32)
    model.eval()
    all_errors, all_elapsed = [], []

    # evaluate on synthetic datasets
    for dataset_name, dataset in datasets.items():
        print(f"Evaluating {model_name} model on {dataset_name} dataset...")
        errors, elapsed = evaluate_on_dataset(
                                    model, 
                                    dataset,
                                    model_name=model_name, 
                                    dataset_name=dataset_name)
        results[model_name][dataset_name] = {
            "errors" : errors,
            "mean_error" : np.mean(errors),
            "std_error" : np.std(errors),
            "elapsed" : elapsed,
            "mean_elaped" : np.mean(elapsed),
            "std_elapsed" : np.std(elapsed)
        }
        all_errors.append(errors)
        all_elapsed.append(elapsed)
        
    results[model_name]["all"] = {
        "mean_errors" : np.mean(all_errors),
        "mean_elapsed" : np.mean(all_elapsed)
    }

    print(f"""All MSE: {np.mean(all_errors):0.2f} dB  Time: {np.mean(all_elapsed)*1e3:0.2f} ms""")
    
    print()

with open(f'results/results.pkl', 'wb') as handle:
    pickle.dump(
            results, 
            handle, 
            protocol=pickle.HIGHEST_PROTOCOL
        )
