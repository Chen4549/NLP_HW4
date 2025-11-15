import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.
        '''
        # TODO
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.max_len = 512 # Define a max sequence length

        schema_path = os.path.join(data_folder, 'flight_database.schema')
        try:
            with open(schema_path, 'r') as f:
                # Read, strip whitespace from each line, filter empty lines, and join with a space
                lines = [line.strip() for line in f.readlines() if line.strip()]
                self.schema_string = " ".join(lines)
        except FileNotFoundError:
            print(f"ERROR: Schema file not found at {schema_path}")
            raise

        self.process_data(data_folder, split)
        

    def process_data(self, data_folder, split):
        # TODO
        nl_path = os.path.join(data_folder, f"{split}.nl")
        self.nl_queries = load_lines(nl_path)
        
        self.sql_queries = []
        if self.split != 'test':
            sql_path = os.path.join(data_folder, f"{split}.sql")
            self.sql_queries = load_lines(sql_path)
            assert len(self.nl_queries) == len(self.sql_queries), "NL and SQL files must have the same number of lines"
    
    def __len__(self):
        # TODO
        return len(self.nl_queries)

    def __getitem__(self, idx):
        # TODO
        nl_text = self.nl_queries[idx]
        
        input_text = f"Schema: {self.schema_string} | Query: {nl_text}"

        # Tokenize the encoder input (Natural Language)
        # We use pt (PyTorch Tensors) here to make collation easier
        encoder_encoding = self.tokenizer(
            nl_text,
            max_length=self.max_len,
            padding=False, # We'll pad in the collate_fn
            truncation=True,
            return_tensors="pt"
        )
        
        # .squeeze(0) removes the batch dimension (1, seq_len) -> (seq_len)
        encoder_ids = encoder_encoding['input_ids'].squeeze(0)
        encoder_mask = encoder_encoding['attention_mask'].squeeze(0)

        # For the test set, we only have NL queries
        if self.split == 'test':
            # We return the raw NL text to be used in test_collate_fn
            return encoder_ids, encoder_mask, nl_text

        # For train/dev sets, we also process the SQL query
        sql_text = self.sql_queries[idx]
        
        # Tokenize the decoder output (SQL Query)
        # This will be used as the "labels" in T5
        target_encoding = self.tokenizer(
            sql_text,
            max_length=self.max_len,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        decoder_targets = target_encoding['input_ids'].squeeze(0)

        # Create decoder_input_ids
        # T5's `decoder_input_ids` should be the `decoder_targets` "shifted right"
        # It starts with the `pad_token_id` (which T5 uses as `decoder_start_token_id`)
        # and ends one token before the `eos_token`
        decoder_inputs = [self.tokenizer.pad_token_id] + decoder_targets.tolist()[:-1]
        decoder_inputs = torch.tensor(decoder_inputs)

        # We return the raw SQL text for evaluation
        return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, sql_text

def normal_collate_fn(batch):
    '''
    Collation function for training and dev.
    '''
    # TODO
    # `batch` is a list of tuples: [(enc_ids, enc_mask, dec_in, dec_tgt, sql_text), ...]
    
    # Separate the tensors and the raw text
    encoder_ids_list = [item[0] for item in batch]
    encoder_mask_list = [item[1] for item in batch]
    decoder_inputs_list = [item[2] for item in batch]
    decoder_targets_list = [item[3] for item in batch]
    sql_texts = [item[4] for item in batch] # Keep raw text separate

    # Pad the tensor sequences
    # `batch_first=True` gives (Batch, Seq_Len)
    # `padding_value=PAD_IDX` (which is 0)
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0) # Mask is padded with 0
    decoder_inputs = pad_sequence(decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)

    # The 5th item is the raw SQL text, which `train_epoch` ignores
    # but `eval_epoch` will use.
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, sql_texts

def test_collate_fn(batch):
    '''
    Collation function for testing.
    '''
    # TODO
    # `batch` is a list of tuples: [(enc_ids, enc_mask, nl_text), ...]
    encoder_ids_list = [item[0] for item in batch]
    encoder_mask_list = [item[1] for item in batch]
    nl_texts = [item[2] for item in batch] # Raw NL text

    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0)

    # The 3rd item is the raw NL text, which `test_inference` will use
    return encoder_ids, encoder_mask, nl_texts

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader

def load_prompting_data(data_folder):
    # TODO
    return None, None, None, None, None