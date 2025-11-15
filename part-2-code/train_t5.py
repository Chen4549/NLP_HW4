import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler, tokenizer):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = 'ft_experiment'
    gt_sql_path = os.path.join(f'data/dev.sql')
    # MODIFICATION: This path is given in the PDF
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        # Pass tokenizer to eval_epoch
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader, tokenizer,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, tokenizer, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Implementation of the evaluation loop.
    '''
    # TODO
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # Ignore pad tokens in loss
    
    all_generated_sqls = []
    all_ground_truth_sqls = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, gt_sqls in tqdm(dev_loader, desc="Evaluating"):
            # Move data to device
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # --- 1. Calculate Loss ---
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']

            # Calculate loss (B, T, V) -> (B*T, V)
            loss = criterion(logits.view(-1, logits.size(-1)), decoder_targets.view(-1))
            
            non_pad = decoder_targets != PAD_IDX
            num_tokens = torch.sum(non_pad).item()
            # Note: T5's criterion does not work well like this with custom loop
            # Let's use the same as train_epoch for consistency
            total_loss += loss.item() * num_tokens # This is not perfect, but tracks progress
            total_tokens += num_tokens

            # --- 2. Perform Generation ---
            # Use model.generate for inference
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=512, # Max length of generated SQL
                num_beams=4, # Beam search
            )
            
            # Decode generated token IDs to strings
            generated_sqls = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            all_generated_sqls.extend(generated_sqls)
            all_ground_truth_sqls.extend(gt_sqls) # gt_sqls is the raw text from collate_fn

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

    save_queries_and_records(
        sql_queries=all_generated_sqls,
        sql_path=model_sql_path,
        record_path=model_record_path
    )

    # Step 2: Compute metrics by comparing the saved files to the ground truth.
    # This function loads the files we just saved and returns the 4-tuple.
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_path=gt_sql_pth,
        model_path=model_sql_path,
        gt_query_records=gt_record_path,
        model_query_records=model_record_path
    )

    # Step 3: Calculate the error rate (a float) from the list of error messages.
    # An error occurred if the error message string is not empty.
    num_errors = sum(1 for msg in model_error_msgs if msg)
    total_queries = len(model_error_msgs)
    error_rate = num_errors / total_queries if total_queries > 0 else 0

    # Step 4: Return the correct values in the correct order
    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, tokenizer, model_sql_path, model_record_path):
    '''
    Implementation of inference for the test set.
    '''
    # TODO
    model.eval()
    all_generated_sqls = []

    with torch.no_grad():
        # test_loader yields (encoder_input, encoder_mask, nl_texts)
        for encoder_input, encoder_mask, nl_texts in tqdm(test_loader, desc="Testing"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            # --- Perform Generation ---
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=512,
                num_beams=4,
            )
            
            generated_sqls = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_generated_sqls.extend(generated_sqls)

    # --- Save Results ---
    # We don't compute metrics, just save the outputs.
    # This is the corrected function call (no db_path).
    print(f"Saving test outputs to {model_sql_path} and {model_record_path}")
    save_queries_and_records(
        sql_queries=all_generated_sqls,
        sql_path=model_sql_path,
        record_path=model_record_path
    )

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    
    # ***MODIFICATION: Get tokenizer from dataset***
    tokenizer = train_loader.dataset.tokenizer 
    
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train (Pass tokenizer)
    train(args, model, train_loader, dev_loader, optimizer, scheduler, tokenizer)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    # MODIFICATION: This path is given in the PDF
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    
    # Pass tokenizer
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader, tokenizer,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"FINAL Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"FINAL Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    
    # Pass tokenizer
    test_inference(args, model, test_loader, tokenizer, model_sql_path, model_record_path)
if __name__ == "__main__":
    main()
