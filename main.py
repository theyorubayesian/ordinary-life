import json
import logging

from dotenv import load_dotenv
from transformers import GPT2Config
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from transformers import HfArgumentParser

from train import prepare_training_data
from trainer import DailyDialogTrainer as Trainer
from trainer import CustomTrainingArguments as TrainingArguments

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
        "--dataset_name",
        type=str,
        default="daily_log"
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
        "--overwrite_data_cache",
        action="store_true",
        help="Overwrite data cache"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache"
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
    load_dotenv()
    training_args, init_args = get_parser()
    # TODO
    # run_sanity_check(args)
    model, tokenizer = get_model_and_tokenizer(init_args)
    train_dataset, eval_dataset = prepare_training_data(tokenizer, init_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
