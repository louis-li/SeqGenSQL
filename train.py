import argparse
import glob
import os
import json
import time
import logging
import random
import re
import copy
from itertools import chain
from tqdm.auto import tqdm

from dataset import WikiSqlDataset
from model import LoggingCallback,SeqGenSQL
import multiprocessing
import torch
import numpy as np
import pytorch_lightning as pl

######################################################################
## Utilities
######################################################################
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def contruct_params(parser):
  parser.add_argument('--data_dir', default="data")
  parser.add_argument("--output_dir", default=".")
  parser.add_argument("--default_root_dir", default=".", help="checkpoint folder") 
  parser.add_argument("--model_name_or_path", default="t5-base")
  parser.add_argument("--max_seq_length", default=512)
  parser.add_argument("--max_output_length", default=200)
  parser.add_argument("--learning_rate", default=2e-4)
  parser.add_argument("--weight_decay", default=0.0)
  parser.add_argument("--adam_epsilon", default=1e-8)
  parser.add_argument("--warmup_steps", default=0)
  parser.add_argument("--train_batch_size", default=32)
  parser.add_argument("--eval_batch_size", default=32)
  parser.add_argument("--num_train_epochs", default=30)
  parser.add_argument("--gradient_accumulation_steps", default=16)
  parser.add_argument("--gpus", default=-1)
  parser.add_argument("--early_stop_callback", default=False)
  parser.add_argument("--max_grad_norm", default=1.0, help="if you enable 16-bit training then set this to a sensible value, 0.5 is a good default")
  parser.add_argument("--seed", default=42)
  parser.add_argument("--early_stop_callback", default=False)
  parser.add_argument("--deterministic", default=False)
  parser.add_argument("--auto_scale_batch_size", default=None, help="None|'power'|'binsearch'")
  parser.add_argument("--benchmark", default=True)
  parser.add_argument("--num_of_workers", default=4)
  parser.add_argument("--distributed_backend", default="dp")
  parser.add_argument("--resume_from_checkpoint", default=None)

  # Data Augmentation and model enhancement Options
  parser.add_argument("--include_data_type", default=True)
  parser.add_argument("--num_sample_rows", default=3)
  parser.add_argument("--data_aug", default=[], help="List, use one of these options: ['select_column', 'where_value']. Default is []")
  parser.add_argument("--use_modified_network", default=True, help="Use gated layer to decide whether to extract or to generate")
  parser.add_argument("--generated_data_files", default=[], help="List of the generated data files. Default is []")
  args = parser.parse_args()
  # calculate number of GPUs for later use
  if isinstance(args.gpus, list):
    args.n_gpu= len(args.gpus)
  elif args.gpus == -1:
    args.n_gpu = torch.cuda.device_count()

  return args

if __name__ == '__main__':
    ## Add parameters
    parser = argparse.ArgumentParser()
    args = contruct_params(parser)
    # main
    logger = logging.getLogger(__name__)

    # suppress warning - Lightning 0.8.4 introduces an issue that could generate overwhelming warning messages
    logging.basicConfig(level=logging.ERROR)

    if args.generated_data_files != []:
        args.data_aug = []

    if isinstance(args.gpus, list):
        args.n_gpu= len(args.gpus)
    elif args.gpus == -1:
        args.n_gpu = torch.cuda.device_count()

    args.train_batch_size= 2 * args.n_gpu
    args.eval_batch_size = 2 * args.n_gpu

    seed_everything(args.seed)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.output_dir, "base_gated_{epoch:02d}-{val_loss:.5f}"), prefix="", monitor="val_loss", mode="min", save_top_k=5
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.gpus,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
    )

    if args.n_gpu > 1:
        train_params["distributed_backend"] = "dp"

    #tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    # initialize model
    model = SeqGenSQL(args)

    # restore full training state
    # trainer = pl.Trainer(resume_from_checkpoint='t5_checkpoints/epoch=15.ckpt', gpus=1, )
    # multi GPUs: 
    #trainer = pl.Trainer(resume_from_checkpoint='t5_checkpoints/base_gated_e03_0.2470.ckpt', **train_params)

    trainer = pl.Trainer(**train_params)

    # Train
    trainer.fit(model) 