import logging
import math
import os
import pickle
import random
from copy import deepcopy

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler


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
            self.lengths = [len(x) for x in self.data]

            with open(cached_features_file, "wb") as f:
                pickle.dump(
                    {"data": self.data, "labels": self.labels, "lengths": self.lengths}, 
                    f, protocol=pickle.HIGHEST_PROTOCOL
                )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"input": self.data[idx], "label": self.labels[idx], "length": self.lengths[idx]}

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

    @staticmethod
    def collate(features):
        input_ids = pad_sequence(
            [torch.tensor(x["input"], dtype=torch.long) for x in features],
            batch_first=True,
            padding_value=0
        )
        labels = pad_sequence(
            [torch.tensor(x["label"], dtype=torch.long) for x in features],
            batch_first=True,
            padding_value=-100
        )
        lengths = torch.tensor([x["length"] for x in features])
        attn_mask = torch.arange(input_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn_mask 
        }


class BucketSampler(Sampler):
    """
    See microsoft/DialoGPT
    https://github.com/microsoft/DialoGPT/blob/master/data_loader.py
    """
    def __init__(self, batch_size, bucket_size, lengths, droplast=False, shuffle=True):
        self._lengths=lengths
        self._bucket_size=bucket_size
        self._batch_size=batch_size
        self._droplast=droplast
        self._shuffle = shuffle

    def __iter__(self):
        ids = list(range(len(self._lengths)))

        if self._shuffle:
            random.shuffle(ids)

        buckets = [
            sorted(ids[i:i+self._bucket_size], key=lambda i: self._lengths[i], reverse=True)
            for i in range(0, len(ids), self._bucket_size)
        ]
        batches = [
            bucket[i:i+self._batch_size] 
            for bucket in buckets 
            for i in range(0, len(bucket), self._batch_size)
        ]

        if self._droplast:
            batches = [batch for batch in batches if len(batch) == self._batch_size]
        
        if self._shuffle:
            random.shuffle(batches)
        
        return iter(batches)

    def __len__(self):
        bucket_sizes = (
            [self._bucket_size] * (len(self._lengths) // self._bucket_size) 
            + [len(self._lengths) % self._bucket_size]
        )
        if self._droplast:
            return sum(s // self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)


class DialogDataLoader:
    def __init__(
        self, 
        dataset,
        sampler,
        num_workers,
        pin_memory
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.num_workers=num_workers
        self.pin_memory=pin_memory
    
    def __len__(self):
        return len(self.sampler)
    
    def __iter__(self):
        loader = DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            collate_fn=DialogDataset.collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        yield from loader
