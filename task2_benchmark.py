from data import get_cifar10
from model import get_big_resnet, get_small_resnet, replace_bottleneck_with_quantizable_bottleneck

import torch
import wandb

from argparse import ArgumentParser
from tqdm import tqdm
from functools import partial
from time import perf_counter

import numpy as np
import random


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--wandb", default=None)
    parser.add_argument("--file", required=True)
    parser.add_argument("--quantize", action="store_true")
    return parser.parse_args()


def evaluate(model, test_dataloader, device, dry_run=False, max_batches=-1):
    model.eval()
    sum_time = 0
    n_guesses = n_samples = 0
    for i, (x, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            start = perf_counter()
            y_pred = model(x).argmax(dim=1)
            if i >= 5:
                sum_time += perf_counter() - start

        n_samples += y.size(0)
        n_guesses += (y_pred == y).sum()

        if i == max_batches:
            break

    acc = n_guesses / n_samples
    mean_time = sum_time / (len(test_dataloader) - 5)
    if not dry_run:
        wandb.log({"test_accuracy": acc, "average_time": mean_time})
    return acc, mean_time


def main(args):
    wandb.init(
        entity="broccoliman",
        project="efficient-dl-week9",
        name=args.wandb
    )
    _, test_loader = get_cifar10(256)
    model = get_small_resnet()
    # replace_bottleneck_with_quantizable_bottleneck(model)

    if args.quantize:
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.ao.quantization.prepare_qat(model, inplace=True)
        model.load_state_dict(torch.load(args.file))

        quantized_model = torch.ao.quantization.convert(model.eval(), inplace=False)
        quantized_model.eval()
        evaluate(quantized_model, test_loader, "cpu")
    else:
        model.load_state_dict(torch.load(args.file))
        evaluate(model, test_loader, "cpu")

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(42)
    main(args)