# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : base_model.py
#   Last Modified : 2022-07-15 15:48
#   Describe      : 
#
# ====================================================
import os
import argparse
import torch
import wandb
import numpy as np
from transformers.optimization import AdamW, get_scheduler
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from typing import Dict


class BaseModel(pl.LightningModule):
    """
    Instantiate a base model based on PyTorch Lightning, defines initialization procedure, basic training/validation/test/predict steps.
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific args
        Returns:
            parent_parser
        """
        parser = parent_parser.add_argument_group('BaseModel')
        # * Args for model setting
        parser.add_argument('--seed', default=None, type=int)
        parser.add_argument('--lr', default=1e-5, type=float)
        parser.add_argument('--l2', default=0., type=float)
        parser.add_argument('--adam_beta1', default=0.9, type=float)
        parser.add_argument('--adam_beta2', default=0.999, type=float)
        parser.add_argument('--scheduler', default="linear", type=str)
        parser.add_argument('--warmup', default=0.1, type=float)
        parser.add_argument('--show_training_ex', default=-1, type=int, help="print the training examples for the batch idx. Set to -1 to disable")
        #TODO Add t5 type, encoder-decoder
        parser.add_argument('--model_type', default=None, choices=["gpt", "bert"], type=str, help="model type, transformer encoder or decoder")
        parser.add_argument('--model_name', default=None, type=str, help="path to directory containing huggingface model or huggingface model name")
        parser.add_argument('--continue_train_from_ckpt', default=None, type=str, help="load a saved lightning checkpoint and continue training")

        return parent_parser

    def __init__(self, args):
        super().__init__()

        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        self.args = args
        self.save_hyperparameters(args)
        self._consumed_samples = 0
        self._consumed_tokens = 0
        self.ts_logger = self.logger.experiment

    def setup(self, stage):
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.micro_batch_size * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches
                print(f"Training batch size: {tb_size * ab_size}")
                wandb.config.update({"train_batch_size": tb_size * ab_size})
                self.total_step = (len(train_loader.dataset) *
                                    self.trainer.max_epochs // tb_size) // ab_size
                wandb.config.update({"total_step": self.total_step})
            else:
                self.total_step = self.trainer.max_steps // self.trainer.accumulate_grad_batches

            print('Total training step:', self.total_step)
        else:
            self.total_step = 0
            print('Not in fit stage, set total training step = 0')

    def get_inputs(self, batch) -> Dict:
        """ process a batch of data to be fed into model, the returned dict should be able to sent to model directly. """
        raise NotImplementedError()

    def training_step(self, batch, batch_idx) -> Dict:
        """ a universal training step for all circumstances"""
        inputs = self.get_inputs(batch)
        input_ids = inputs["input_ids"]
        batch_size = input_ids.size(0)
        world_size = self.trainer.world_size
        self._consumed_samples += batch_size * max(world_size, 1)
        labels = inputs["labels"]
        if labels is not None:
            self._consumed_tokens += len(labels.flatten()) * max(world_size, 1)
        else:
            self._consumed_tokens += len(input_ids.flatten()) * max(world_size, 1)

        loss, logits = self(**inputs)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, batch_size=batch_size)
        wandb.log({"train_loss": loss})

        self.ts_logger.add_scalar("train_loss_vs_samples", loss.item(), self._consumed_samples)
        self.ts_logger.add_scalar("train_loss_vs_tokens", loss.item(), self._consumed_tokens)

        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        """ save a huggingface checkpoint for each epoch """
        self.avg_train_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(), 4)
        self.log("avg_train_loss", self.avg_train_loss, prog_bar=False, logger=True, on_epoch=True)
        wandb.log({"avg_train_loss": self.avg_train_loss})

        self.save_hf_checkpoint()

    def validation_step(self, batch, batch_idx) -> Dict:
        """ validation step """
        inputs = self.get_inputs(batch)
        batch_size = inputs["input_ids"].size(0)
        loss, logits = self(**inputs)

        self.log("val_loss", loss, prog_bar=False, logger=True, on_step=True, batch_size=batch_size)
        wandb.log({"val_loss": loss})

        return {"loss": loss}

    def validation_epoch_end(self, validation_step_outputs):
        self.avg_val_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in validation_step_outputs])).item(), 4)
        self.log("avg_val_loss", self.avg_val_loss, prog_bar=False, logger=True, on_epoch=True)
        wandb.log({"avg_val_loss": self.avg_val_loss})

        self.ts_logger.add_scalar("val_loss_vs_samples", self.avg_val_loss, self._consumed_samples)

    def test_step(self, batch, batch_idx) -> Dict:
        """ test step """
        inputs = self.get_inputs(batch)
        batch_size = inputs["input_ids"].size(0)
        loss, logits = self(**inputs)

        self.log("test_loss", loss, prog_bar=False, logger=True, on_step=True, batch_size=batch_size)
        wandb.log({"test_loss": loss})

        return {"loss": loss}

    def test_epoch_end(self, test_step_outputs):
        self.avg_test_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in test_step_outputs])).item(), 4)
        self.log("avg_test_loss", self.avg_test_loss, prog_bar=False, logger=True, on_epoch=True)
        wandb.log({"avg_test_loss": self.avg_test_loss})

        self.ts_logger.add_scalar("test_loss_vs_samples", self.avg_test_loss, self._consumed_samples)

    def configure_optimizers(self) -> Dict:
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {'params': [p for n, p in self.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.hparams.l2},
            {'params': [p for n, p in self.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # Configure optimizer.
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            if 'offload_optimizer' in self.trainer.training_type_plugin.config['zero_optimization']:
                optimizer = DeepSpeedCPUAdam(
                    optimizer_grouped_params, adamw_mode=True,
                    lr=self.hparams.lr,
                    betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                    #  eps=self.hparams.adam_epsilon,
                    )
            else:
                optimizer = FusedAdam(
                    optimizer_grouped_params, adam_w_mode=True,
                    lr=self.hparams.lr,
                    betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                    #  eps=self.hparams.adam_epsilon,
                    )
        else:
            optimizer = AdamW(optimizer_grouped_params, lr=self.hparams.lr,
                              betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                              #  eps=self.hparams.adam_epsilon,
                              )
        # Configure learning rate scheduler.
        warmup_steps = self.hparams.warmup * self.total_step
        scheduler = get_scheduler(name=self.hparams.scheduler, optimizer=optimizer,
                                  num_warmup_steps=warmup_steps, num_training_steps=self.total_step)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [{
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }]

    def save_hf_checkpoint(self) -> None:
        #  if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
        """Save huggingface model checkpoint and tokenizer"""
        if self.global_rank == 0:
            save_path = os.path.join(
                self.trainer.checkpoint_callback.dirpath if self.trainer else self.hparams.save_dir,
                'hf_pretrained_epoch{}_step{}'.format(self.current_epoch, self.global_step))
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
