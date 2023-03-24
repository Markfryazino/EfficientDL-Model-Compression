from data import get_cifar10
from model import get_big_resnet, get_small_resnet

import torch
import wandb

from argparse import ArgumentParser
from tqdm import tqdm
from functools import partial

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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def time_to_stop(accs):
    if len(accs) < 3:
        return False

    return (torch.abs(accs[-1] - accs[-2]) < 0.01) and (torch.abs(accs[-2] - accs[-3]) < 0.01)


def evaluate(model, test_dataloader, device):
    model.eval()
    n_guesses = n_samples = 0
    for x, y in tqdm(test_dataloader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = model(x).argmax(dim=1)
        n_samples += y.size(0)
        n_guesses += (y_pred == y).sum()

    acc = n_guesses / n_samples
    wandb.log({"test_accuracy": acc}, step=wandb.run.step)
    return acc


teacher_activations = {}
student_activations = {}
def hook(module, input, output, activations):
    activations[module] = output


def distillation(model: torch.nn.Module, teacher: torch.nn.Module, train_dataloader, test_dataloader, device):
    model.to(device)
    teacher.to(device)
    teacher.eval()
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    test_accs = []
    epoch = step = logging_loss = logging_acc = logging_ce = logging_mse = logging_cls = 0

    global hook
    hook_handles = [
        model.layer1.register_forward_hook(partial(hook, activations=student_activations)),
        model.layer2.register_forward_hook(partial(hook, activations=student_activations)),
        model.layer4.register_forward_hook(partial(hook, activations=student_activations)),
        teacher.layer1.register_forward_hook(partial(hook, activations=teacher_activations)),
        teacher.layer2.register_forward_hook(partial(hook, activations=teacher_activations)),
        teacher.layer4.register_forward_hook(partial(hook, activations=teacher_activations)),
    ]

    while not time_to_stop(test_accs):
        model.train()
        print(f"Epoch {epoch}: ", end="")
        for x, y in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            with torch.no_grad():
                pred_teacher = teacher(x)

            cls_loss = loss(pred, y)
            ce_loss = loss(
                torch.nn.functional.softmax(pred, dim=1),
                torch.nn.functional.softmax(pred_teacher, dim=1)
            )
            logging_cls += cls_loss / 10
            logging_ce += ce_loss / 10
            logging_acc += (pred.argmax(dim=1) == y).sum() / y.size(0) / 10

            student_features = torch.cat([
                student_activations[model.layer1].flatten(),
                student_activations[model.layer2].flatten(),
                student_activations[model.layer4].flatten(),
            ])
            teacher_features = torch.cat([
                teacher_activations[teacher.layer1].flatten(),
                teacher_activations[teacher.layer2].flatten(),
                teacher_activations[teacher.layer4].flatten(),                    
            ])

            mse_loss = mse(student_features, teacher_features)
            logging_mse += mse_loss / 10
            logging_loss += (cls_loss + ce_loss + mse_loss) / 30
            bloss = (cls_loss + ce_loss + mse_loss) / 3

            bloss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if step % 10 == 0:
                metrics = {
                    "training_accuracy": logging_acc.item(),
                    "training_loss": logging_loss.item(),
                    "training_classification_loss": logging_cls.item(),
                    "training_cross_entropy_loss": logging_ce.item(),
                    "epoch": epoch,
                }
                metrics["training_mse_loss"] = logging_mse.item()

                wandb.log(metrics, step=step)
                logging_acc = logging_loss = logging_ce = logging_mse = logging_cls = 0.

        epoch += 1
        if epoch > 3:
            model.apply(torch.ao.quantization.disable_observer)
        if epoch > 2:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        torch.save(model.state_dict(), f"models/test.pt")

        test_accs.append(
            evaluate(
                model, 
                test_dataloader, device
            )
        )

    for hook in hook_handles:
        hook.remove()

    return model


def main(args):
    wandb.init(
        entity="broccoliman",
        project="efficient-dl-week9",
        name=args.wandb
    )
    train_loader, test_loader = get_cifar10(256)

    model = get_small_resnet()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    torch.ao.quantization.prepare_qat(model, inplace=True)

    teacher = get_big_resnet()
    teacher.load_state_dict(torch.load("models/a100-full-ft.pt"))

    distillation(
        model=model,
        teacher=teacher,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        device=args.device
    )

    if args.save_model:
        torch.save(model.state_dict(), f"models/{args.wandb}.pt")

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(42)
    main(args)