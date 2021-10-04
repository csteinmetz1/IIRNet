import os
import sys
import time
import glob
import torch
import random
import pickle
import argparse
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from collections import defaultdict

from data.fir import FIRFilterDataset
from iirnet.mlp import MLPModel
from iirnet.data import IIRFilterDataset
from iirnet.plotting import plot_responses
import iirnet.signal as signal

from baselines.sgd import SGDFilterDesign
from baselines.yw import YuleWalkerFilterDesign


def count_parameters(model):
    if len(list(model.parameters())) > 0:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        params = 0
    return params


def evaluate_on_dataset(
    model: pl.LightningModule,
    dataset: torch.utils.data.Dataset,
    model_name: str,
    dataset_name: str,
    plot: bool = True,
    eps: float = 1e-8,
    gpu: bool = False,
    examples: int = 100,
    seed: int = 42,
):

    pl.seed_everything(seed)

    errors = []
    timings = []

    if dataset_name != "all":
        runs = min(examples, len(dataset))
    else:
        runs = examples

    for idx in range(runs):

        if dataset_name == "all":
            d = random.choice(dataset)
            example = d[idx]
        else:
            example = dataset[idx]

        mag_dB, mag_dB_norm, phs, real, imag, sos = example

        if model_name != "Yule-Walker":
            mag_dB = mag_dB.to(map_loc)
            mag_dB_norm = mag_dB_norm.to(map_loc)

        # predict filter coeffieicnts (do timings here)
        tic = time.perf_counter()
        with torch.no_grad():
            if model_name in [
                "Yule-Walker",
                "SGD (1)",
                "SGD (10)",
                "SGD (100)",
                "SGD (1000)",
            ]:
                pred_sos = model(mag_dB.view(1, 1, -1))
            else:
                pred_sos, _ = model(mag_dB_norm.view(1, 1, -1))
        toc = time.perf_counter()
        elapsed = toc - tic

        # compute response of the predicted and target filter
        _, pred_dB = signal.sosfreqz(pred_sos.squeeze(), worN=mag_dB.shape[-1])
        pred_dB = 20 * torch.log10(signal.mag(pred_dB) + eps)

        if dataset_name in ["guitar_cab", "hrtf"] or model_name in [
            "Yule-Walker",
            "SGD (1000)",
        ]:
            target_dB = mag_dB.squeeze().to(pred_dB.device)
        else:  # use SOS
            _, target_dB = signal.sosfreqz(sos.squeeze(), worN=mag_dB.shape[-1])
            target_dB = 20 * torch.log10(signal.mag(target_dB) + eps)

        error = torch.nn.functional.mse_loss(pred_dB.squeeze(), target_dB.squeeze())
        errors.append(error.item())
        timings.append(elapsed)

        if plot:
            model_plot_dir = os.path.join("data", "plots", model_name)
            if not os.path.isdir(model_plot_dir):
                os.makedirs(model_plot_dir)
            filename = os.path.join(
                model_plot_dir, f"{model_name}-{dataset_name}-{idx}.png"
            )
            plot_responses(
                pred_sos.detach().cpu().squeeze(),
                target_dB.cpu().squeeze(),
                filename=filename,
            )

        # sys.stdout.write(
        #    f"* {idx+1}/{len(dataset)}: MSE: {np.mean(errors):0.2f} dB  Time: {np.mean(timings)*1e3:0.2f} ms\r"
        # )
        # sys.stdout.flush()

    # print()
    return errors, timings


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_dir",
        help="Path to experiment directory.",
    )
    parser.add_argument(
        "--gpu",
        help="Run models on GPU.",
        action="store_true",
    )
    parser.add_argument(
        "--plot",
        help="Save plots for all examples.",
        action="store_true",
    )
    parser.add_argument(
        "--eps",
        help="Epsilon value for stability.",
        default=1e-8,
        type=np.float32,
    )
    parser.add_argument(
        "--seed",
        help="Dataset generation seed.",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--examples",
        help="Number of examples for each filter method.",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--yw",
        help="Evaluate with Yule-Walker.",
        action="store_true",
    )
    parser.add_argument(
        "--sgd",
        help="Evaluate with SGD method.",
        action="store_true",
    )
    parser.add_argument(
        "--guitar_cab",
        help="Evaluate using Guitar cabinent dataset.",
        action="store_true",
    )
    parser.add_argument(
        "--hrtf",
        help="Evaluate using HRTF dataset.",
        action="store_true",
    )
    parser.add_argument(
        "--filter_order",
        help="Filter order to use for Yule-Walker and SGD",
        default=16,
        type=int,
    )

    args = parser.parse_args()

    # use a different seed than the one used during training
    if args.seed == 13:
        print(
            f"Warning! Seed = {args.seed} was used during training.",
            "Try using a different seed.",
        )

    pl.seed_everything(args.seed)

    # fixed evaluation parameters
    num_points = 512
    max_eval_order = args.filter_order
    examples_per_method = args.examples
    precompute = True
    shuffle = False

    # prepare testing datasets
    val_datasetA = IIRFilterDataset(
        method="normal_poly",
        num_points=num_points,
        max_order=max_eval_order,
        num_examples=examples_per_method,
        precompute=precompute,
    )

    val_datasetB = IIRFilterDataset(
        method="normal_biquad",
        num_points=num_points,
        max_order=max_eval_order,
        num_examples=examples_per_method,
        precompute=precompute,
    )

    val_datasetC = IIRFilterDataset(
        method="uniform_disk",
        num_points=num_points,
        max_order=max_eval_order,
        num_examples=examples_per_method,
        precompute=precompute,
    )

    val_datasetD = IIRFilterDataset(
        method="uniform_mag_disk",
        num_points=num_points,
        max_order=max_eval_order,
        num_examples=examples_per_method,
        precompute=precompute,
    )

    val_datasetE = IIRFilterDataset(
        method="char_poly",
        num_points=num_points,
        max_order=max_eval_order,
        num_examples=examples_per_method,
        precompute=precompute,
    )

    val_datasetF = IIRFilterDataset(
        method="uniform_parametric",
        num_points=num_points,
        max_order=max_eval_order,
        num_examples=examples_per_method,
        precompute=precompute,
    )

    val_guitar_cab_datatset = FIRFilterDataset("data/GtrCab")
    val_hrtf_datatset = FIRFilterDataset("data/HRTF/IRC_1059_C")

    datasets = {
        # "normal_poly": val_datasetA,
        # "normal_biquad": val_datasetB,
        # "uniform_disk": val_datasetC,
        # "uniform_mag_disk": val_datasetD,
        # "char_poly": val_datasetE,
        # "uniform_parametric": val_datasetF,
        "all": [
            val_datasetA,
            val_datasetB,
            val_datasetC,
            val_datasetD,
            val_datasetE,
            val_datasetF,
        ],
    }

    if args.hrtf:
        datasets["hrtf"] = val_hrtf_datatset
    if args.guitar_cab:
        datasets["guitar_cab"] = val_guitar_cab_datatset

    # load models from disk
    models = {
        # "Yule-Walker": YuleWalkerFilterDesign(N=16),
        # "SGD (1)": SGDFilterDesign(n_iters=1).to("cuda"),
        # "SGD (10)": SGDFilterDesign(n_iters=10).to("cuda"),
        # "SGD (100)": SGDFilterDesign(n_iters=100).to("cuda"),
        # "SGD (1000)": SGDFilterDesign(n_iters=1000).to("cuda"),
    }

    # there are three different experiment modes
    # hidden_dim, filter_method, and filter_order
    experiment_name = os.path.basename(args.experiment_dir)
    print(experiment_name)

    if args.yw:
        models["Yule-Walker"] = YuleWalkerFilterDesign(N=args.filter_order)
    if args.sgd:
        models["SGD (1)"] = SGDFilterDesign(n_iters=1, order=args.filter_order)
        models["SGD (10)"] = SGDFilterDesign(n_iters=10, order=args.filter_order)
        models["SGD (100)"] = SGDFilterDesign(n_iters=100, order=args.filter_order)
        models["SGD (1000)"] = SGDFilterDesign(n_iters=1000, order=args.filter_order)

    # get all models from the experiment
    model_dirs = glob.glob(os.path.join(args.experiment_dir, "*"))

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        model_ckpts = glob.glob(
            os.path.join(
                model_dir,
                "lightning_logs",
                "version_0",
                "checkpoints",
                "*.ckpt",
            )
        )
        if len(model_ckpts) < 1:
            raise RuntimeError(f"No checkpoints found in {model_dir}.")
        model_ckpt = model_ckpts[0]

        if args.gpu:
            map_loc = "cuda"
        else:
            map_loc = "cpu"

        model = MLPModel.load_from_checkpoint(model_ckpt, map_location=map_loc)
        model.to(map_loc)
        models[model_name] = model

    results = defaultdict(dict)

    for model_name, model in models.items():

        print("-" * 32)
        model.eval()
        synthetic_errors, synthetic_elapsed = [], []

        print(
            f"Evaluating {model_name} model  {count_parameters(model)/1e6:0.2f} M parameters"
        )
        # evaluate on synthetic datasets
        avg = []
        for dataset_name, dataset in datasets.items():
            sys.stdout.write(f"{dataset_name} dataset ")
            errors, elapsed = evaluate_on_dataset(
                model,
                dataset,
                model_name,
                dataset_name,
                gpu=map_loc,
                examples=examples_per_method,
                plot=args.plot,
                seed=args.seed,
            )
            results[model_name][dataset_name] = {
                "errors": errors,
                "mean_error": np.mean(errors),
                "std_error": np.std(errors),
                "elapsed": elapsed,
                "mean_elapsed": np.mean(elapsed),
                "std_elapsed": np.std(elapsed),
            }
            avg.append(np.mean(errors))
            sys.stdout.write(f"{np.mean(errors):0.2f} \n")
            if dataset_name not in ["hrtf", "guitar_cab"]:
                synthetic_errors += errors
                synthetic_elapsed += elapsed

        results[model_name]["all"] = {
            "mean_error": np.mean(synthetic_errors),
            "mean_elapsed": np.mean(synthetic_elapsed),
        }

        results[model_name]["avg"] = np.mean(avg)

        # print(
        #    f"""MSE: {np.mean(synthetic_errors):0.2f} dB  Time: {np.mean(synthetic_elapsed)*1e3:0.2f} ms"""
        # )
        print(
            f"Avg MSE: {np.mean(avg):0.2f} | Time: {np.mean(synthetic_elapsed)*1e3:0.2f} ms"
        )

    with open(f"results/results_{experiment_name}.pkl", "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
