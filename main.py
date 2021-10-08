import argparse
import json
import logging
import os
import time
from math import ceil

import torch
import neptune.new as neptune
from dotenv import load_dotenv
from transformers import GPT2Config
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

from train import prepare_training_data
from trainer import DailyDialogTrainer as Trainer
from trainer import CustomTrainingArguments as TrainingArguments
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
        output_hidden_states=False
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
        "--output_dir",
        default="outputs"
    )
    parser.add_argument(
        "--seed",
        default=42
    )
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
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="" # TODO
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

    # -----------------------
    # Validation & Checkpoint
    # -----------------------
    parser.add_argument(
        "--log_params",
        action="store_true"
    )
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

    train_dataset, val_dataset = prepare_training_data(args, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to="neptune",
        sampler_bucket_size=100,
        dataloader_drop_last=True,
        dataloader_shuffle=True,
        shuffle=True,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        evaluation_strategy="epoch",
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        save_total_limit=3, # TODO
        load_best_model_at_end=True # TODO
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()
