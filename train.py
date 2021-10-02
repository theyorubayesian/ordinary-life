import logging
import os
import shutil
import time
from bisect import bisect_left
from math import exp

import numpy as np
import psutil
import torch
from datasets import load_dataset
from torch import nn

from dataset import DialogDataset
from dataset import DialogDataLoader
# from generate import generate_responses

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def prepare_training_data(args, tokenizer):
    data = load_dataset(args.dataset_name)
    train, val = data["train"], data["validation"]
    train_data = DialogDataset(tokenizer, train["dialog"], "train", args)
    val_data = DialogDataset(tokenizer, val["dialog"], "val", args)

    # train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_data, args.val_batch_size, shuffle=False)

    train_dataloader = DialogDataLoader(
        data=train_data,
        batch_size=args.train_batch_size,
        dtype=torch.long,
        repeat=True,
        shuffle=True,
        sort=False,
        sort_within_batch=True,
        args=args
    )
    val_dataloader = DialogDataLoader(
        data=val_data,
        batch_size=args.val_batch_size,
        dtype=torch.long,
        repeat=True,
        shuffle=True,
        sort=False,
        sort_within_batch=True,
        args=args
    )

    return train_dataloader, val_dataloader


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


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    output_dir,
    to_bucket=False,
    bucket=None,
    bucket_dir=None,
    cleanup=True
):
    os.makedirs(output_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)
    torch.save(
        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
    )
    torch.save(
        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
    )
    logger.info(f"Saving model, optimizer and scheduler states to {output_dir}")
    
    if to_bucket:
        for f in os.listdir(output_dir):
            dest = f"{bucket_dir}/{f}"
            blob = bucket.blob(dest)
            blob.upload_from_filename(os.path.join(output_dir, f))

        logger.info(f"Uploaded checkpoints to bucket: {bucket_dir}")
        if cleanup:
            shutil.rmtree(output_dir)


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


def check_val_rank(existing_ckpts, ppl, args):
    ckpt_rank = bisect_left([v[1] for v in existing_ckpts], ppl)

    if (ckpt_rank < len(existing_ckpts)) or (len(existing_ckpts) < args.num_val_checkpoints):
        return ckpt_rank
    return -1


def train_epoch(
    train_dataloader, 
    val_dataloader,
    model, 
    tokenizer,
    optimizer,
    scheduler,
    iters,
    logging_client,
    bucket,
    args
):
    iters.epoch_step = 0
    iters.n_sequences_epoch = 0
    iters.total_loss_epoch = 0

    for batch in train_dataloader:
        # TODO: Consider Gradient Accumulation Steps
        if iters.steps_trained_in_epoch > 0:
            iters.step_trained_in_epoch -= 1
            continue

        if args.n_gpu > 0:
            batch = tuple(t.to(args.device) for t in batch)
        input_ids, label_ids  = batch

        outputs = model(
            input_ids=input_ids,
            labels=label_ids,
            return_dict=True
        )
        loss= outputs.loss

        if (loss != loss).data.any():
            logger.error("NaN detected in loss")
            exit()
        
        if args.multi_gpu:
            loss = loss.mean()
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        ppl = exp(loss.item())
        loss.backward()
        
        iters.last_ppl = ppl
        iters.last_loss = loss.item()
        iters.total_loss_epoch += loss.item()
        iters.step += 1
        iters.total_training_time += time.time() - iters.last_iter_time
        iters.last_iter_time = time.time()
        
        if iters.step % args.gradient_accumulation_steps == 0:
            iters.global_step += 1
            iters.epoch_step += 1

            nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )

            optimizer.step()
            scheduler.step()

            if args.is_master:
                log_neptune(logging_client, model, optimizer, iters, log_params=args.log_params)
                iters.last_log = time.time()
            
            optimizer.zero_grad()
            
            if args.is_master:
                if iters.global_step % args.checkpoint_interval == 0:
                    output_dir = os.path.join(
                        args.output_dir, f"checkpoint-{iters.global_step}")
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        output_dir=output_dir,
                        to_bucket=args.upload_to_bucket,
                        bucket=bucket,
                        bucket_dir=args.bucket_dir,
                        cleanup=args.cleanup
                    )

                if iters.global_step % args.validation_interval == 0:
                    val_loss, val_ppl = validate(
                        model=model,
                        dataloader=val_dataloader,
                        iters=iters,
                        args=args
                    )
                    logging_client["val/loss"].log(val_loss)
                    logging_client["val/ppl"].log(val_ppl)

                    # TODO
                    checkpoint_val_rank = check_val_rank(iters.validation_ckpts, val_ppl, args)

                    # TODO: Checkpoint during validation?
                    if (checkpoint_val_rank  != -1):
                        if args.validation_interval != args.checkpoint_interval:
                            output_dir = os.path.join(
                                args.output_dir, f"checkpoint-validation-{iters.global_step}")
                            save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                output_dir=args.output_dir,
                                to_bucket=args.upload_to_bucket,
                                bucket=bucket,
                                bucket_dir=args.bucket_dir,
                                cleanup=args.cleanup
                            )
                        else:
                            # shutil.move(output_dir, )
                            os.rename(output_dir, "-validation-".join(output_dir.split("-")))

        iters.n_sequences_epoch += input_ids.size(0)
    # iters.n_toten_total += input_ids.shape[0] * input_ids.shape[1]
    # iters.n_token_real += (input_ids != 0).sum().item()

    print(f"Epoch: {iters.epoch} Epoch Step: {iters.epoch_step}")

    if args.is_master:
        logger.info(f"Ending epoch {iters.epoch}/{args.n_epochs-1}")
        output_dir = os.path.join(
            args.output_dir, f"epoch-checkpoint-{iters.epoch}"
        )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            to_bucket=args.upload_to_bucket,
            bucket=bucket,
            bucket_dir=args.bucket_dir,
            cleanup=args.cleanup
        )
        logging_client["epoch/loss"].log(iters.total_loss_epoch / iters.epoch_step)

        if args.generate_responses:
            # model.eval()

            # TODO
            # generate_responses(
            #    model=model,
            #    tokenizer=tokenizer,
            #    batch_size=64,
            #    tokenizer_max_len=128,
            #    model_max_len=128,
            #    beam=1,
            #    context_file="compression/distillation/msft/dstc/data/test.source",
            #    output_file=f"{args.dump_path}/model.epoch_{iters.epoch}.6k.resp.txt"
            # )   # TODO
            
            # model.train()
            pass
