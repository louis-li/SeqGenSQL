import argparse
import glob
import os
import logging
import random
import re
import copy
from itertools import chain

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import multiprocessing
from dataset import WikiSqlDataset

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from tqdm.auto import tqdm

######################################################################
## Layer Norm
######################################################################
class LayerNorm(pl.LightningModule):
    def __init__(self, hidden_size, eps=1e-6):
        """ Construct a layernorm module in the T5 style
            No bias and no substraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # layer norm should always be calculated in float32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype == torch.float16:
            x = x.to(torch.float16)
        return self.weight * x


######################################################################
## T5 Model with modified layer for WikiSQL
######################################################################

class SeqGenSQL(pl.LightningModule):
  def __init__(self, hparams):
    super(SeqGenSQL, self).__init__()
  
    if not isinstance(hparams, argparse.Namespace):
      hparams = argparse.Namespace(**hparams)    

    self.hparams = hparams
    
    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
    
    if hparams.use_modified_network:
        #hparam.max_seq_lengt 
        self.inner_dim = self.model.config.num_heads * self.model.config.d_kv
        self.q =  nn.Linear(self.model.config.d_model, self.inner_dim, bias = False) 
        self.k =  nn.Linear(self.model.config.d_model, self.inner_dim, bias = False) 
        self.v =  nn.Linear(self.model.config.d_model, self.inner_dim, bias = False) 
        self.layer_norm_gen = LayerNorm(self.model.config.d_model, eps=self.model.config.layer_norm_epsilon)
        self.layer_norm_ext = LayerNorm(self.model.config.d_model, eps=self.model.config.layer_norm_epsilon)
     
        # Added gated layer with model.config.d_model
        self.ff_gate = nn.Linear(self.model.config.d_model * 2,1, bias = False) 
        self.o = nn.Linear(self.inner_dim, self.model.config.d_model, bias = False) 
    
  
  def is_logger(self):
    return self.trainer.global_rank <= 0
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
  ):
    # last-layer hidden state, (presents,) (all hidden states), (all attentions)
    # Interesting, hidden state is same as 'all hidden states'
    # When no output_hidden_states provide, last layer hidden state is outputs[1]
    # When output_hidden_states = True, all hidden state from transformer will be in outputs[3]
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        lm_labels=lm_labels,
        output_hidden_states=True, # to output all hidden states from T5Stack
    )
    

  def _step(self, batch, debug=False):
    lm_labels = batch["target_ids"]
    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        lm_labels=lm_labels,
        decoder_attention_mask=batch['target_mask']
    )
    if (self.hparams.use_modified_network):
      # This is implementation for gated generation/extraction
      # Add additional layer to decide whether to extract or to generate
      ######################################################
      # Generative Branch
      # Generative Branch is from original T5 pretrain hidden state of last decoder layer before LM_Header layer
      ######################################################
      # Get generated branch - the original output hidden state
      # also scale like the original T5
      output_hidden_state = outputs[3][-1] * (self.model.model_dim ** -0.5)    #[batch_size, output_len, d_model]
      #decoder_state_norm = self.layer_norm(output_hidden_state)
      #print(decoder_state_norm.shape)
 
      # Pass final LM head
      lm_logits_gen = self.model.lm_head(output_hidden_state)
      ######################################################    
      # Extractive Branch
      # Extractive Branch is a cross attention between input questions/column headers and generate sql sequence
      ######################################################    
      # Get hidden state for input
      # To get the output of encoder, use model.get_encoder()(batch["source_ids"])
      bs, qlen, dim = output_hidden_state.size()
      def shape(x):
        """  projection """
        return x.view(bs, -1, self.model.config.num_heads, self.model.config.d_kv).transpose(1, 2)
      def unshape(x):
        """  compute context """
        return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)
        
      input_hidden_state = self.model.get_encoder()(batch["source_ids"])[0] #[batch_size, input_len, d_model]
      q = shape(self.q (output_hidden_state)) #[batch_size, n_heads,  input_len, dim_per_head]
      #print("q shape:", q.shape)
      v = shape(self.v(input_hidden_state))  #[batch_size, n_heads, output_len, dim_per_head]
      #print("v shape:", v.shape)
      k = shape(self.k(input_hidden_state)) #[batch_size, n_heads,  output_len, dim_per_head]  
      #print("k shape:", k.shape)
      
      # Simplified CrossAttention
      scores = torch.einsum("bnqd,bnkd->bnqk", q ,k)         # (batch, n_heads, output_len, input_len)
      #print("scores shape:", scores.shape)
      
      # mask datatypes, data values from input
      #gate_masks = torch.unsqueeze(batch['gate_mask'], dim=1)
      #scores = scores * gate_masks
    
      weights = F.softmax(scores.float(), dim=-1 ).type_as(scores)    # (batch, n_heads, output_len, input_len)
      #print("weights shape:", weights.shape)
      weights = nn.Dropout(p=self.model.config.dropout_rate) (weights) # (batch, n_heads, output_len, input_len)

      # Context
      context = torch.matmul(weights, v)   # (batch, n_heads, output_len, dim_per_head)  
      #print("context shape:", context.shape)
      context = unshape(context)
      #print("context shape:", context.shape)
      
      # Feed Forward layer
      context = self.o(context)              # (batch, output_len, d_model)
      #print("context shape:", context.shape)

      # Scale like original T5
      # this step is required because both branches need to be at the same scale
      context =  context * (self.model.model_dim ** -0.5)
      #context_norm = self.layer_norm(context)
      lm_logits_ext = self.model.lm_head(context)
      ######################################################    
      # Use probability to decide whether generate or extract
      ######################################################  
      # Pass gate layer - Probablities of generation or extration
      #gate_layer = self.ff_gate(self.layer_norm_gen(output_hidden_state)+self.layer_norm_ext(context))  # [batch_size, output_len, input_len+d_model]
      gate_layer = self.ff_gate(torch.cat((self.layer_norm_gen(output_hidden_state), self.layer_norm_ext(context)), dim=2))  # [batch_size, output_len, input_len+d_model]
      gate_layer_output = torch.nn.Sigmoid()(gate_layer)    # [batch_size, output_len, 1]
      
      # Put everything together:
      # merge output_hidden_state (generative) and input position index (extractive)
      #print(gate_layer_output.shape, decoder_state_norm.shape, context_norm.shape)
      
      ######################################################    
      # Use gated output to pass LM_Head layer
      ######################################################  
      lm_logits = (1 - gate_layer_output) * lm_logits_gen + gate_layer_output * lm_logits_ext
      #merged_output_norm =  self.layer_norm(merged_output)

      
      # Calculate new loss for gated layer
      loss_fct = CrossEntropyLoss(ignore_index=-100)
      #print(lm_logits.size(-1))
      #print(lm_logits.view(-1, lm_logits.size(-1)))
      loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
      return (loss,lm_logits,context,output_hidden_state, input_hidden_state,self.model.get_encoder()(batch["source_ids"]), lm_labels.view(-1)) 
    else:
      #loss = outputs[0]
      return outputs
    
    

  def training_step(self, batch, batch_idx, debug=False):
    outputs = self._step(batch, debug)
    loss = outputs[0]

    tensorboard_logs = {"train_loss": loss}

    if debug:
      return outputs
    else:
      return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    tensorboard_logs = {"avg_train_loss": avg_train_loss, 
                        "avg_gate_value":torch.mean(torch.nn.Sigmoid()(self.ff_gate.weight))}
    return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def validation_step(self, batch, batch_idx):
    outputs = self._step(batch)
    loss = outputs[0]
    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #avg_loss = torch.stack([x["val_loss"] for x in torch.reshape(outputs, (-1,))]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"
    optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]

  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  def get_dataset(self, data_type):
    return WikiSqlDataset(tokenizer=self.tokenizer, 
                            data_dir=self.hparams.data_dir, 
                            dataset_type=data_type, 
                            include_data_type = self.hparams.include_data_type, 
                            include_sample_data = self.hparams.num_sample_rows, 
                            data_augmentation = self.hparams.data_aug,
                            generated_data = self.hparams.generated_data_files,
                            max_input_len=self.hparams.max_seq_length,  
                            max_output_len=self.hparams.max_output_length)

  def train_dataloader(self):
    train_dataset = self.get_dataset(data_type="train")
    dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, 
                            drop_last=True, shuffle=True, num_workers=self.hparams.num_of_workers)
    t_total = (
        (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
        // self.hparams.gradient_accumulation_steps
        * float(self.hparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    val_dataset = self.get_dataset(data_type="dev")
    return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, drop_last=True, num_workers=self.hparams.num_of_workers)

class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))

######################################################################
## Logging
######################################################################
class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))