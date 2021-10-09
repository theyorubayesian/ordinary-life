import json
import logging

import torch
from transformers import GPT2Config
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from transformers import HfArgumentParser

from train import prepare_training_data
from trainer import DailyDialogTrainer as Trainer
from trainer import CustomTrainingArguments as TrainingArguments
from utils import init_gpu_params
from utils import set_seed

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
    # parser = argparse.ArgumentParser()
    parser = HfArgumentParser(TrainingArguments)
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

    # -----------------------
    # Validation & Checkpoint
    # -----------------------
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

    training_args, init_args = parser.parse_args_into_dataclasses()
    return training_args, init_args


def main():
    # --------------
    # Initialization
    # --------------
    training_args, init_args = get_parser()
    # TODO
    # run_sanity_check(args)
    # init_gpu_params(args) # TODO
    # set_seed(args)    # TODO
    training_args.device = torch.device(f"cuda:{training_args.local_rank}" if training_args.n_gpu else "cpu")

    model, tokenizer = get_model_and_tokenizer(init_args)

    train_dataset, val_dataset = prepare_training_data(init_args, tokenizer)

    # TODO: Include no_cuda argument 
    # training_args = TrainingArguments(
    #    output_dir=args.output_dir,
    #    report_to="neptune",
    #    sampler_bucket_size=100,
    #    dataloader_drop_last=True,
    #    dataloader_shuffle=True,
    #    # dataloader_pin_memory=False,
    #    # dataloader_num_workers=1,
    #    shuffle=True,
    #    num_train_epochs=args.n_epochs,
    #    per_device_train_batch_size=args.train_batch_size,
    #    per_device_eval_batch_size=args.val_batch_size,
    #    evaluation_strategy="epoch",
    #    fp16=args.fp16,
    #    fp16_opt_level=args.fp16_opt_level,
    #    warmup_steps=args.warmup_steps,
    #    learning_rate=args.learning_rate,
    #    lr_scheduler_type=args.lr_scheduler_type,
    #    adam_epsilon=args.adam_epsilon,
    #    weight_decay=args.weight_decay,
    #    save_total_limit=3, # TODO
    #    load_best_model_at_end=True # TODO
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()
