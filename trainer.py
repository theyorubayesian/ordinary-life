import logging
from dataclasses import dataclass
from dataclasses import field

from transformers import Trainer
from transformers import TrainingArguments

from dataset import BucketSampler
from dataset import DialogDataLoader
# from generate import generate_responses

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DailyDialogTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a training dataset")
        
        train_dataset = self.train_dataset
        train_sampler = BucketSampler(
            batch_size=self.args.per_device_train_batch_size,
            bucket_size=self.args.sampler_bucket_size,  # TODO
            lengths=train_dataset.lengths,
            droplast=self.args.dataloader_drop_last,
            shuffle=self.args.dataloader_shuffle   # TODO
        )

        return DialogDataLoader(
            dataset=train_dataset,
            sampler=train_sampler,
            num_workers=self.args.dataloader_num_workers, # TODO
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None and self.eval_dataset==None:
            raise ValueError("Trainer: evaluation requires an eval dataset")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_sampler = BucketSampler(
            batch_size=self.args.per_device_train_batch_size,
            bucket_size=self.args.sampler_bucket_size,  # TODO
            lengths=eval_dataset.lengths,
            droplast=self.args.dataloader_drop_last,
            shuffle=self.args.dataloader_shuffle   # TODO
        )

        return DialogDataLoader(
            dataset=eval_dataset,
            sampler=eval_sampler,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    sampler_bucket_size: int = field(
        default=100, init=True, metadata={"help": "Size of bucket used in dataloader sampler"}
    )
    dataloader_shuffle: bool = field(
        default=True, init=True, metadata={"help": "Shuffle indices and batches in dataloader sampler"}
    )
