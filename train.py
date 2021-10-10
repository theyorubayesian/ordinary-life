import logging
import time
from math import exp

import numpy as np
import psutil
import torch
from datasets import load_dataset

from dataset import DialogDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def prepare_training_data(tokenizer, args):
    data = load_dataset(args.dataset_name)
    train, val = data["train"], data["validation"]

    train_dataset = DialogDataset(
        tokenizer, train["dialog"], "train", args
    )
    val_dataset = DialogDataset(
        tokenizer, val["dialog"], "val", args
    )
    
    return train_dataset, val_dataset


def log_neptune(logging_client, model, optimizer, iters, log_params=False):
    logging_client["losses/cum_avg_loss_epoch"].log(
        iters.total_loss_epoch / iters.epoch_step)
    logging_client["losses/loss"].log(iters.last_loss)
    logging_client["metric/ppl"].log(iters.last_ppl)
    
    if log_params:
        for param_name, param in model.named_parameters():
            logging_client[f"parameter_mean/{param_name}"].log(
                param.data.mean())
            logging_client[f"parameter_std/{param_name}"].log(
                param.data.std())
            if param.grad is None:
                continue
            logging_client[f"grad_mean/{param_name}"].log(
                param.grad.data.mean())
            logging_client[f"grad_std/{param_name}"].log(
                param.grad.data.std())

    logging_client["global/lr"].log(optimizer.param_groups[0]["lr"])
    logging_client["global/memory_usage"].log(psutil.virtual_memory()._asdict()["used"] / 1_000_000)
    logging_client["global/speed"].log(time.time() - iters.last_log)


def validate(
    model,
    dataloader,
    iters,
    args
):
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []

    with torch.no_grad():
        for batch in dataloader:
            if args.n_gpu > 0:
                batch = tuple(t.to(args.device) for t in batch)
            input_ids, label_ids = batch
            n_sample = input_ids.shape[0]

            outputs = model(
                input_ids=input_ids,
                labels=label_ids,
                return_dict=True
            )
            loss = outputs.loss
            ppl = exp(loss.mean().item())

            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl * n_sample)
            tot_sample.append(n_sample)
    val_loss = np.sum(tot_loss) / np.sum(tot_sample)
    val_ppl = np.sum(tot_ppl) / np.sum(tot_sample)
    print(
        f"Epoch {iters.epoch}: \nVal loss: {val_loss}\n" \
        f"Val ppl: {val_ppl}"
    )
    return val_loss, val_ppl
