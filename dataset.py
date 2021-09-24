import logging
import os
import pickle
from copy import deepcopy

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.data.iterator import BucketIterator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_conversation(dialog: list, tokenizer):
    """
    Concatenates multi-turn conversation, separated by EOS token
    """
    flatten = lambda conv: [token for turn in conv for token in turn]
    full_conv = [
        tokenizer.encode(turn)
        + [tokenizer.eos_token_id] for turn in dialog
        ]
    return flatten(full_conv)


class DialogDataset(Dataset):
    def __init__(self, tokenizer, conv_list, split_name, args,):
        os.makedirs(args.cache_dir, exist_ok=True)
        cached_features_file = os.path.join(
            args.cache_dir, f"cached_dataset_{split_name}"
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info(f"Loading features from cached file: {cached_features_file}")
            with open(cached_features_file, "rb") as f:
                cache = pickle.load(f)
                self.data = cache["data"]
                self.labels = cache["labels"]
        else:
            logger.info(f"Creating features from data. Dataset will be cached to {cached_features_file}")

            self.data = []
            for conv in conv_list:
                full_conv = create_conversation(conv, tokenizer)
                if len(full_conv) <= args.max_seq_len:
                    self.data.append(full_conv)
                else:
                    sub_convs = []
                    for sub_conv in self.make_chunks(full_conv, args.max_seq_len - 2):
                        if sub_conv[0] != tokenizer.eos_token_id:
                            sub_conv.insert(0, tokenizer.eos_token_id)
                        if sub_conv[-1] != tokenizer.eos_token_id:
                            sub_conv.insert(len(sub_conv), tokenizer.eos_token_id)
                        
                        assert len(sub_conv) <= args.max_seq_len
                        assert (sub_conv[0] == sub_conv[-1] == tokenizer.eos_token_id), \
                            "EOS token does not start or end sub sequence"
                        sub_convs.append(sub_conv)

                    self.data.extend(sub_convs)
            self.remove_short_sentences(args)
            self.labels = deepcopy(self.data)
            self.pad_to_multiple_of_eight()

            with open(cached_features_file, "wb") as f:
                pickle.dump(
                    {"data": self.data, "labels": self.labels}, 
                    f, protocol=pickle.HIGHEST_PROTOCOL
                )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"input": self.data[idx], "label": self.labels[idx]}

    @staticmethod
    def make_chunks(full_conv, chunk_size):
        return [full_conv[i: i+chunk_size] for i in range(0, len(full_conv), chunk_size)]

    def remove_short_sentences(self, args):
        start_size = len(self.data)
        self.data = [data for data in self.data if len(data) >= args.min_seq_len]
        logger.info(f"Removed {start_size - len(self.data)} sequences which were shortert than {args.min_seq_len}")
    
    def pad_to_multiple_of_eight(self):
        for data, label in zip(self.data, self.labels):
            while len(data) % 8 != 0:
                data.append(0)
                label.append(-100)
            assert len(data) == len(label)


class DialogDataLoader:
    def __init__(self, data, batch_size, dtype, repeat, shuffle, sort, sort_within_batch, args):
        self.dtype = dtype
        self.batch_size = batch_size
        self.bucket = BucketIterator(
            data,
            batch_size=batch_size,
            device=args.device,
            sort_key=lambda x: len(x["input"]),
            repeat=repeat,
            shuffle=shuffle,
            sort=sort,
            sort_within_batch=sort_within_batch
        )
        self.bucket.create_batches()
    
    def __len__(self):
        return len(self.bucket)
    
    def __iter__(self):
        for batch in self.bucket.batches:
            if len(batch) < self.batch_size:
                continue
            input_ids = pad_sequence(
                [torch.tensor(x["input"], dtype=self.dtype) for x in batch],
                batch_first=True,
                padding_value=0
            )
            labels = pad_sequence(
                [torch.tensor(x["label"], dtype=self.dtype) for x in batch],
                batch_first=True,
                padding_value=-100
            )
            yield input_ids, labels
