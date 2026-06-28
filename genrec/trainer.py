import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
from logging import getLogger
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_scheduler

from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.evaluator import Evaluator
from genrec.utils import get_file_name, get_total_steps, config_for_log, log


class Trainer:
    
    # A class that handles the training process for a model.

    # Args:
    #     config (dict): The configuration parameters for training.
    #     model (AbstractModel): The model to be trained.
    #     tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the data.

    # Methods:
    #     fit(train_dataloader, val_dataloader): Trains the model using the provided training and validation dataloaders.
    #     -> Returns: best_epoch, best_val_score
    #     evaluate(dataloader, split='test'): Evaluate the model on the given dataloader.
    #     -> Returns: OrderedDict: A dictionary containing the evaluation results.
    #     end(): Ends the training process and releases any used resources.
    

    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='.pth')
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)
        
        self.debug_flag = True

    def _get_wandb_init_kwargs(self) -> dict:
        wandb_kwargs = {
            "name": self.config.get("run_id") or get_file_name(self.config, suffix=''),
            "group": f"{self.config['dataset']}/{self.config['model']}",
        }
        if self.config.get("wandb_entity"):
            wandb_kwargs["entity"] = self.config["wandb_entity"]
        if self.config.get("wandb_tags"):
            wandb_kwargs["tags"] = self.config["wandb_tags"]
        if self.config.get("wandb_notes"):
            wandb_kwargs["notes"] = self.config["wandb_notes"]
        return wandb_kwargs

    def init_wandb(self):
        project_name = self.config.get("wandb_project") or get_file_name(self.config, suffix='')
        self.accelerator.init_trackers(
            project_name=project_name,
            config=config_for_log(self.config),
            init_kwargs={"wandb": self._get_wandb_init_kwargs()},
        )

    def fit(self, train_dataloader, val_dataloader):
        """
        Train the model using the provided training and validation dataloaders.
        Args:
            train_dataloader (DataLoader): The training dataloader.
            val_dataloader (DataLoader): The validation dataloader.
        Returns:
            best_epoch (int): The epoch number of the best model.
            best_val_score (float): The validation score of the best model.
        """
        # -------------------- Initialize the optimizer --------------------
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return None, None

        # -------------------- Initialize the learning rate scheduler --------------------
        # Cosine annealing with warmup: learning rate gradually increases during warmup, then gradually decreases following a cosine curve
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_n_steps,
        )
        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        
        # -------------------- Initialize Weights & Biases logging --------------------
        self.init_wandb()

        # If config specify total steps, use it to calculate the number of epochs
        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1

        # LOGGING
        n_train_samples = len(train_dataloader.dataset)
        n_batches_per_epoch = len(train_dataloader)
        n_steps_per_epoch = n_batches_per_epoch  # 1 GPU: one optimizer.step() per batch
        self.log('[TRAINER] ======================= Training started... ========================')
        self.log(f'[TRAINER] Number of training epochs: {n_epochs}')
        self.log(f'[TRAINER] Total training steps (LR schedule): {total_n_steps}')
        self.log(f'[TRAINER] Training batch size: {self.config["train_batch_size"]}')
        self.log(f'[TRAINER] Number of training samples: {n_train_samples}')
        self.log(f'[TRAINER] Number of batches per epoch: {n_batches_per_epoch}')
        self.log(f'[TRAINER] Number of steps per epoch: {n_steps_per_epoch}')
        
        # -------------------- Training Loop --------------------
        for epoch in range(n_epochs):
            self.log(f'[TRAINER] ==================== TRAINING ====================')
            # ===== Training Phase =====
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )
            # ---------- Batch Loop ----------
            for batch in train_progress_bar:
                optimizer.zero_grad()               # Reset gradients
                outputs = self.model(batch)         # Forward pass
                loss = outputs.loss                 # Compute loss
                self.accelerator.backward(loss)     # Backward pass
                # Clip gradients to prevent exploding gradients
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()                    # Update weights
                scheduler.step()                    # Update learning rate
                # Accumulate loss for the epoch
                total_loss = total_loss + loss.item()

            # Logging training loss 
            self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)
            self.log(f'[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_dataloader)}')

            # On-train-epoch-end hooks
            model_for_hooks = (
                self.accelerator.unwrap_model(self.model)
                if self.config.get('use_ddp')
                else self.model
            )
            for metric_name, metric_value in model_for_hooks.on_train_epoch_end(epoch).items():
                self.accelerator.log({metric_name: metric_value}, step=epoch + 1)
                self.log(f'[Epoch {epoch + 1}] {metric_name} → {metric_value:.4f}')

            # -------------------- Validation Phase --------------------
            self.log(f'[TRAINER] ======================= VALIDATION ========================')
            if (epoch + 1) % self.config['eval_interval'] == 0: # Evaluate depend on the interval
                
                all_results = self.evaluate(val_dataloader, split='val')
                
                # --- Logging validation results ---
                if self.accelerator.is_main_process:
                    for key in all_results:
                        if key != 'val_loss':
                            self.accelerator.log({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                        if key == 'val_loss':
                            self.accelerator.log({f"Loss/val_loss": all_results[key]}, step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')
                    
                # --- Early stopping and model saving ---
                val_score = all_results[self.config['val_metric']]
                
                # Save model if validation score improves
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        if self.config['use_ddp']: # unwrap model for saving
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        self.log(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')

                # Early  stop if no improvement for 'patience' epochs
                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    self.log(f'EARLY STOPPING AT EPOCH {epoch + 1}')
                    break
                
        # Final loggings
        self.log(f'BEST EPOCH: {best_epoch}, BEST VAL SCORE ({self.config["val_metric"]}): {best_val_score}')
        return best_epoch, best_val_score

    def evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.
        Args:
            dataloader (DataLoader): The dataloader to evaluate the model on.
            split (str): The split to evaluate the model on.
        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        
        self.model.eval()  # Set model to evaluation mode
        loss_key = f'{split}_loss'
        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        
        # -------------------- Batch Loop --------------------
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                
                # 1. Use model.generate() to get predictions and loss
                # 2. Use evaluator.calculate_metrics() to calculate the metrics
                if self.config['use_ddp']:
                    preds, loss = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk, return_loss=True)
                    if isinstance(preds, tuple):
                        preds, n_visited_items = preds
                        all_preds, all_labels, all_n_visited_items = self.accelerator.gather_for_metrics((preds, batch['labels'], n_visited_items))
                        all_preds = (all_preds, all_n_visited_items)
                    else:
                        all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds, loss = self.model.generate(batch, n_return_sequences=self.evaluator.maxk, return_loss=True)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])

                # Accumulate results
                for key, value in results.items():
                    all_results[key].append(value)
                # Store validation loss (unsqueeze to make it 1-dimensional for concatenation)
                all_results[loss_key].append(loss.detach().cpu().unsqueeze(0))
                

        # ---------------- Aggregate Results Across All Batches ----------------
        # Compute mean metrics over all evaluation samples
        final_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                final_results[key] = torch.cat(all_results[key]).mean().item()
        final_results[loss_key] = torch.cat(all_results[loss_key]).mean().item()
        
        return final_results


    def end(self):
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
