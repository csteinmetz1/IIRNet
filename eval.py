import os
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
from iirnet.data import IIRFilterDataset
from iirnet.loss import LogMagTargetFrequencyLoss
from iirnet.plotting import plot_responses
import iirnet.signal as signal

from baselines.sgd import SGDFilterDesign
from baselines.yw import YuleWalkerFilterDesign

pl.seed_everything(32)

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
normal_poly_ckpt        = 'lightning_logs/400/normal_poly/lightning_logs/version_0/checkpoints/last.ckpt'
normal_biquad_ckpt      = 'lightning_logs/400/normal_biquad/lightning_logs/version_0/checkpoints/last.ckpt'
uniform_disk_ckpt       = 'lightning_logs/400/uniform_disk/lightning_logs/version_0/checkpoints/last.ckpt'
uniform_mag_disk_ckpt   = 'lightning_logs/400/uniform_mag_disk/lightning_logs/version_0/checkpoints/last.ckpt'
char_poly_ckpt          = 'lightning_logs/400/char_poly/lightning_logs/version_4/checkpoints/char_poly-epoch=06-step=5473.ckpt' #'lightning_logs/400/char_poly/lightning_logs/version_0/checkpoints/last.ckpt'
uniform_parametric_ckpt = 'lightning_logs/400/uniform_parametric/lightning_logs/version_0/checkpoints/uniform_parametric-epoch=133-step=104787.ckpt'
all_ckpt                = 'lightning_logs/400/all/lightning_logs/version_5/checkpoints/last.ckpt' #'lightning_logs/400/all/lightning_logs/version_0/checkpoints/all-epoch=351-step=275263.ckpt'

# load models from disk
models = {
    #"Yule-Walker"       : YuleWalkerFilterDesign(N=16),
    #"SGD (1)"           : SGDFilterDesign(n_iters=1),
    #"SGD (10)"          : SGDFilterDesign(n_iters=10),
    #"SGD (100)"         : SGDFilterDesign(n_iters=100),
    #"SGD (1000)"        : SGDFilterDesign(n_iters=1000),
    #"normal_poly"       : MLPModel.load_from_checkpoint(normal_poly_ckpt),
    #"normal_biquad"     : MLPModel.load_from_checkpoint(normal_biquad_ckpt),
    #"uniform_disk"      : MLPModel.load_from_checkpoint(uniform_disk_ckpt),
    #"uniform_mag_disk"  : MLPModel.load_from_checkpoint(uniform_mag_disk_ckpt),
    #"char_poly"         : MLPModel.load_from_checkpoint(char_poly_ckpt),
    #"uniform_parametric": MLPModel.load_from_checkpoint(uniform_parametric_ckpt),
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

    pl.seed_everything(32)

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

        # zero mean
        input_dB = input_dB - torch.mean(input_dB)

        error = torch.nn.functional.mse_loss(input_dB, target_dB)
        errors.append(error.item())
        timings.append(elapsed)

        if plot:
            model_plot_dir = os.path.join("data", "plots", model_name)
            if not os.path.isdir(model_plot_dir):
                os.makedirs(model_plot_dir)
            filename = os.path.join(model_plot_dir, f"{model_name}-{dataset_name}-{idx}.png")
            plot_responses(pred_sos.detach(), target_dB, filename=filename)

        sys.stdout.write(f"* {idx+1}/{len(dataset)}: MSE: {np.mean(errors):0.2f} dB  Time: {np.mean(timings)*1e3:0.2f} ms\r")
        sys.stdout.flush()

    print()
    return errors, timings

results = defaultdict(dict)

for model_name, model in models.items():

    print("-" * 32)
    model.eval()
    synthetic_errors, synthetic_elapsed = [], []

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
            "mean_elapsed" : np.mean(elapsed),
            "std_elapsed" : np.std(elapsed)
        }
        if dataset_name not in ["hrtf", "guitar_cab"]:
            synthetic_errors += errors
            synthetic_elapsed += elapsed
        
    results[model_name]["all"] = {
        "mean_error" : np.mean(synthetic_errors),
        "mean_elapsed" : np.mean(synthetic_elapsed)
    }

    print(f"""Synthetic MSE: {np.mean(synthetic_errors):0.2f} dB  Time: {np.mean(synthetic_elapsed)*1e3:0.2f} ms""")
    print()

with open(f'results/results.pkl', 'wb') as handle:
    pickle.dump(
            results, 
            handle, 
            protocol=pickle.HIGHEST_PROTOCOL
        )
