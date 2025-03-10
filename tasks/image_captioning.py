import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.logging_utils import setup_logger
from data_utils.utils import collate_fn
from .base_task import BaseTask
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
import evaluation
from evaluation import Cider

import os
import numpy as np
from tqdm import tqdm
import itertools
from shutil import copyfile
import json

logger = setup_logger()

@META_TASK.register()
class ImageCaptioningTask(BaseTask):
    """
        Task được thiết kế cho bài toán Image Captioning trên dataset OpenViLC.
        Các phương thức load_datasets, create_dataloaders, evaluate_loss, evaluate_metrics,
        train, start và get_predictions được điều chỉnh phù hợp cho captioning.
    """
    def __init__(self, config):
        super().__init__(config)
    def load_datasets(self, config):
        self.train_dataset = build_dataset(config.JSON_PATH.TRAIN, self.vocab, config.FEATURE_DATASET)
        self.dev_dataset = build_dataset(config.JSON_PATH.DEV, self.vocab, config.FEATURE_DATASET)
        self.test_dataset = build_dataset(config.JSON_PATH.TEST, self.vocab, config.FEATURE_DATASET)

    def create_dataloaders(self, config):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=config.DATASET.FEATURE_DATASET.BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.DATASET.FEATURE_DATASET.WORKERS,
            collate_fn=collate_fn
        )

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.warmup = config.TRAINING.WARMUP
        self.score = config.TRAINING.SCORE  # ví dụ: CIDEr
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.training_beam_size = config.TRAINING.TRAINING_BEAM_SIZE
        self.evaluating_beam_size = config.TRAINING.EVALUATING_BEAM_SIZE
        self.patience = config.TRAINING.PATIENCE
        # Xây dựng hàm tính điểm CIDEr dựa trên caption training set
        self.train_cider = Cider({f"{idx}": cap for idx, cap in enumerate(self.train_dataset.captions)})

    def evaluate_loss(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        with tqdm(desc=f'Epoch {self.epoch} - Validation', unit='it', total=len(dataloader)) as pbar:
            with torch.no_grad():
                for it, items in enumerate(dataloader):
                    items = items.to(self.device)
                    # Model trả về log probabilities cho mỗi từ trong caption
                    out = self.model(items).contiguous()
                    # Giả sử đối tượng items có thuộc tính shifted_right_caption_tokens chứa các token dịch sang phải (target cho teacher forcing)
                    shifted_right_caption_tokens = items.shifted_right_caption_tokens
                    loss = self.loss_fn(out.view(-1, len(self.vocab)), shifted_right_caption_tokens.view(-1))
                    running_loss += loss.item()
                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()
        val_loss = running_loss / len(dataloader)
        return val_loss

    def evaluate_metrics(self, dataloader):
        self.model.eval()
        gens = {}
        gts = {}
        with tqdm(desc=f'Epoch {self.epoch} - Evaluation', unit='it', total=len(dataloader)) as pbar:
            for it, items in enumerate(dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    outs, _ = self.model.beam_search(items, batch_size=items.batch_size, beam_size=self.evaluating_beam_size, out_size=1)

                captions_gt = items.caption
                captions_gen = self.vocab.decode_answer(outs.contiguous().view(-1, self.vocab.max_answer_length), 
                                                         join_words=False)
                for i, (gt, gen) in enumerate(zip(captions_gt, captions_gen)):
                    # Xử lý để loại bỏ từ lặp liên tiếp nếu cần
                    gen = ' '.join([k for k, g in itertools.groupby(gen)])
                    gens[f'{it}_{i}'] = [gen, ]
                    gts[f'{it}_{i}'] = gt
                pbar.update()
        scores, _ = evaluation.compute_scores(gts, gens)
        return scores

    def train(self):
        self.model.train()
        running_loss = 0.0
        with tqdm(desc=f'Epoch {self.epoch} - Training with cross-entropy loss', unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                self.optim.zero_grad()
                out = self.model(items).contiguous()
                shifted_right_caption_tokens = items.shifted_right_caption_tokens
                loss = self.loss_fn(out.view(-1, len(self.vocab)), shifted_right_caption_tokens.view(-1))
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()
        logger.info(f"Training loss: {running_loss / len(self.train_dataloader)}")

    def start(self):
        # Nếu tồn tại checkpoint, tải checkpoint
        last_checkpoint = os.path.join(self.checkpoint_path, "last_model.pth")
        if os.path.isfile(last_checkpoint):
            checkpoint = self.load_checkpoint(last_checkpoint)
            best_val_score = checkpoint["best_val_score"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"] + 1
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            best_val_score = 0.0
            patience = 0

        while True:
            self.train()
            val_loss = self.evaluate_loss(self.dev_dataloader)
            scores = self.evaluate_metrics(self.dev_dataloader)
            logger.info(f"Validation scores: {scores}")
            val_score = scores[self.score]

            best = False
            if val_score > best_val_score:
                best_val_score = val_score
                patience = 0
                best = True
            else:
                patience += 1

            exit_train = False
            if patience >= self.patience:
                logger.info("Patience reached.")
                exit_train = True

            self.save_checkpoint({
                'best_val_score': best_val_score,
                'patience': patience,
            })

            if best:
                copyfile(os.path.join(self.checkpoint_path, "last_model.pth"), 
                         os.path.join(self.checkpoint_path, "best_model.pth"))

            if exit_train:
                break

            self.epoch += 1

    def get_predictions(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            logger.error("Prediction requires a trained model. No best_model.pth found in the checkpoint path!")
            raise FileNotFoundError("Make sure your checkpoint path is correct and best_model.pth is available.")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))
        self.model.eval()

        results = []
        overall_gens = {}
        overall_gts = {}
        with tqdm(desc='Getting predictions on test set: ', unit='it', total=len(self.test_dataloader)) as pbar:
            for it, items in enumerate(self.test_dataloader):
                items = items.to(self.device)
                with torch.no_grad():
                    outs = self.model.generate(
                        input_ids=items.input_ids, 
                        attention_mask=items.attention_mask, 
                        max_length=self.vocab.max_caption_length, 
                        num_beams=self.evaluating_beam_size
                    )
                captions_gt = items.captions
                captions_gen = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
                gts = {}
                gens = {}
                for i, (gt, gen) in enumerate(zip(captions_gt, captions_gen)):
                    gen = ' '.join([k for k, g in itertools.groupby(gen)])
                    gens[f'{it}_{i}'] = gen
                    gts[f'{it}_{i}'] = gt
                    overall_gens[f'{it}_{i}'] = [gen, ]
                    overall_gts[f'{it}_{i}'] = gt
                pbar.update()

                results.append({
                    "id": items.image_id,  # giả sử items có thuộc tính image_id
                    "filename": items.filename,
                    "gens": gens,
                    "gts": gts
                })

        scores, _ = evaluation.compute_scores(overall_gts, overall_gens)
        logger.info(f"Evaluation score on test set: {scores}")

        with open(os.path.join(self.checkpoint_path, "test_results.json"), "w+", encoding="utf-8") as f:
            json.dump({
                "results": results,
                **scores
            }, f, ensure_ascii=False)
