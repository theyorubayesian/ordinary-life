import argparse
import json
import logging
import os
import time
from math import ceil

import torch
import neptune.new as neptune
from dotenv import load_dotenv
from google.cloud import storage
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2Config
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

from train import prepare_training_data
from train import save_checkpoint
from train import train_epoch
from train import validate
from utils import init_gpu_params
from utils import set_seed
from utils import Iters

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv()


def get_model_and_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.special_tokens: # TODO: Check
        assert isinstance(args.special_tokens, dict), "Check that special_tokens is dict"
        tokenizer.add_special_tokens(args.special_tokens)

    config = GPT2Config.from_pretrained(
        args.config_name,
        cache_dir=args.cache_dir,
        output_hidden_states=args.output_hidden_states
    )
    model = GPT2LMHeadModel.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        config=config
    )
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def get_parser():
    parser = argparse.ArgumentParser()
    
    # -----------------
    # Model & Tokenizer
    # -----------------
    parser.add_argument(
        "--tokenizer_name"
    )
    parser.add_argument(
        "--config_name"
    )
    parser.add_argument(
        "--model_name_or_path"
    )
    parser.add_argument(
        "--special_tokens",
        type=json.loads,
    )

    # ----
    # Data
    # ----
    parser.add_argument(
        "--dataset_name"
    )
    parser.add_argument(
        "--max_seq_len",
        default=128,
        type=int
    )
    parser.add_argument(
        "-min_seq_len",
        default=7,
        type=int
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true"
    )
    parser.add_argument(
        "--cache_dir"
    )

    # ---------
    # Modelling
    # ---------
    parser.add_argument(
        "--fp16",
        action="store_true"
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1"
    )
    parser.add_argument(
        "--continue_from_step",
        default=0,
        help="Step at which training should begin"
    )
    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int
    )
    parser.add_argument(
        "--val_batch_size",
        default=8,
        type=int
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=5,
        help="Number of epochs to train for"
    )
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4000
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        help="" # TODO
    )

    # -----------------------
    # Validation & Checkpoint
    # -----------------------
    parser.add_argument(
        "--checkpoint_interval",
        default=5000,
        type=int
    )
    parser.add_argument(
        "--validation_interval",
        default=5000,
        type=int
    )
    parser.add_argument(
        "--num_val_checkpoints",
        default=5,
        type=int,
        help="Number of Validation checkpoints to keep"
    )
    parser.add_argument(
        "--generate_responses",
        action="store_true"
    )
    parser.add_argument(
        "--upload_to_bucket",
        action="store_true"
    )
    parser.add_argument(
        "--bucket_dir",
        type=str,
        help=""
    )
    parser.add_argument(
        "--cleanup",
        action="store_true"
    )

    # ---
    # GPU
    # ---
    parser.add_argument(
        "--n_gpu",
        default=0,  # TODO
        type=int
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    # --------------
    # Initialization
    # --------------
    args = get_parser()
    # TODO
    # run_sanity_check(args)
    init_gpu_params(args)
    set_seed(args)
    args.device = torch.device(f"cuda:{args.local_rank}" if args.n_gpu else "cpu")

    model, tokenizer = get_model_and_tokenizer(args)
    train_dataloader, val_dataloader = prepare_training_data(args, tokenizer)

    if args.num_training_steps:
        num_training_steps = args.num_training_steps
        args.n_epochs = ceil(args.num_training_steps / len(train_dataloader) / args.gradient_accumulation_steps)
    else:
        num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.n_epochs
    
    # -----------------
    # Logging & Storage
    # -----------------
    logging_client = None
    storage_client = None

    if args.is_master:
        logging_client = neptune.init(
            project=os.getenv("NEPTUNE_PROJECT_NAME"),
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
            mode=os.getenv("NEPTUNE_CONNECTION_MODE"),
            source_files=[] # TODO
        )
        args_copy = vars(args)
        for a in args_copy:
            logging_client[a] = args_copy[a]
            logger.info(f"{a}:  {args_copy[a]}")

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(
            os.getenv("GOOGLE_CLOUD_BUCKET_NAME")
        )
    
    # --------
    # Trackers
    # --------
    iters = Iters(
        {
            "days":  0,
            "epoch": 0,
            "steps_trained_in_epoch": 0,
            "epoch_step": 0,
            "global_step": 0,
            "step": 0,
            "total_training_time": 0,
            "last_loss": 0,
            "last_ppl": 0,
            "total_loss_epoch": 0,
            "n_sequences_epoch": 0,
            "last_log": time.time(),
            "last_iter_time": time.time(),
            "validation_ckpts": []
        }
    )

    # ------------------------
    # Optimizer Initialization
    # ------------------------
    no_decay = ["bias", "ln"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        betas=(0.9, 0.999),
    )
    optimizer_dump_path = os.path.join(args.output_dir, "optimizer.pt")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    scheduler_dump_path = os.path.join(args.output_dir, "scheduler.pt")

    if (
        args.model_name_or_path and
        os.path.isfile(optimizer_dump_path) and
        os.path.isfile(scheduler_dump_path)
    ):
        optimizer.load_state_dict(torch.load(optimizer_dump_path))
        scheduler.load_state_dict(torch.load(scheduler_dump_path))    

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex: https://www.github.com/nvidia/apex"
            " for fp16 training")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_op_level)
    
    # --------
    # Training
    # --------
    if args.continue_from_step:
        assert os.path.exists(args.model_name_or_path)
        iters.global_step = int(args.model_name_or_path.split("-")[-1])
        iters.epoch = iters.global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        iters.steps_trained_in_epoch = iters.global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    while iters.epoch < args.n_epochs:
        if args.multi_gpu:
            torch.distributed.barrier()

        train_epoch(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            iters=iters,
            logging_client=logging_client,
            bucket=bucket,
        )
    
    if args.is_master:
        val_loss, val_ppl = validate(
            model=model,
            dataloader=val_dataloader,
            iters=iters,
            args=args
        )
        logging_client["val/loss"].log(val_loss)
        logging_client["val/ppl"].log(val_ppl)

        logger.info("Saving final checkpoint as `pytorch_model.bin`")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=args.output_dir,
            upload_to_bucket=args.upload_to_bucket,
            bucket=bucket,
            bucket_dir=args.bucket_dir,
            cleanup=args.cleanup
        )        
        logger.info("Training is finished")


if __name__ == "__main__":
    main()
