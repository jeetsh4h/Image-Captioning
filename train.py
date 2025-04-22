import os
import json
import time
import torch
import argparse
import traceback
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Any
from datetime import timedelta
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from transformers.models.auto.tokenization_auto import AutoTokenizer

from models import ImageCaptionModel
from datasets import ImageCaptionDataset
from utils import transform, visualize_log


smoothie = SmoothingFunction()


def train_epoch(
    model: ImageCaptionModel,
    train_loader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    current_epoch: int,
    device: torch.device,
    scaler: GradScaler,
) -> tuple[float, Any, list[float]]:

    model.train()

    losses: list[float] = []
    bleu4_scores: list = []
    hypotheses: list[list[str]] = []
    references: list[list[list[str]]] = []

    bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Training epoch {current_epoch+1}",
    )
    for i, batch in bar:
        try:
            images = batch["image"].to(device, non_blocking=True)
            caption_ids = batch["caption"].to(device, non_blocking=True)

            all_captions_seq = batch["all_captions_seq"]
            decoder_input = caption_ids[:, :-1]

            optimizer.zero_grad()
            with autocast(str(device)):
                raw_logits = model(images, decoder_input)
                target_tokens = caption_ids[:, 1:].contiguous().view(-1)
                loss = loss_fn(raw_logits.view(-1, raw_logits.size(-1)), target_tokens)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

            # Calculate BLEU-4 score
            probs = F.softmax(raw_logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            decoded_texts = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids  # type: ignore
            ]
            hypo = [text.split() for text in decoded_texts]

            batch_size = len(hypo)
            ref = []
            for idx in range(batch_size):
                ri = [entry[idx].split() for entry in all_captions_seq if entry[idx]]
                ref.append(ri)

            bleu4_scores.append(
                corpus_bleu(ref, hypo, smoothing_function=smoothie.method4)
            )
            hypotheses += hypo
            references += ref

            bar.set_postfix(loss=losses[-1], bleu4=bleu4_scores[-1])

        except Exception as e:
            print(f"Batch {i} error: {e}")
            traceback.print_exc()
            continue

    train_bleu4 = corpus_bleu(
        references, hypotheses, smoothing_function=smoothie.method4
    )
    average_loss = sum(losses) / len(losses)

    return average_loss, train_bleu4, losses


def validate_epoch(
    model: ImageCaptionModel,
    valid_loader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    loss_fn: nn.CrossEntropyLoss,
    current_epoch: int,
    device: torch.device,
) -> tuple[float, Any, list[float]]:

    model.eval()

    losses: list[float] = []
    bleu4_scores: list = []
    hypotheses: list[list[str]] = []
    references: list[list[list[str]]] = []

    with torch.no_grad():
        bar = tqdm(
            enumerate(valid_loader),
            total=len(valid_loader),
            desc=f"Validating epoch {current_epoch+1}",
        )
        for i, batch in bar:
            images = batch["image"].to(device, non_blocking=True)
            caption_ids = batch["caption"].to(device, non_blocking=True)

            all_captions_seq = batch["all_captions_seq"]
            decoder_input = caption_ids[:, :-1]
            raw_logits = model(images, decoder_input)
            target_tokens = caption_ids[:, 1:].contiguous().view(-1)

            loss = loss_fn(raw_logits.view(-1, raw_logits.size(-1)), target_tokens)
            losses.append(loss.item())

            probs = F.softmax(raw_logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            decoded_texts = [
                tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids  # type: ignore
            ]
            hypo = [text.split() for text in decoded_texts]

            batch_size = len(hypo)
            ref = []
            for idx in range(batch_size):
                ri = [entry[idx].split() for entry in all_captions_seq if entry[idx]]
                ref.append(ri)

            bleu4_scores.append(
                corpus_bleu(ref, hypo, smoothing_function=smoothie.method4)
            )
            hypotheses += hypo
            references += ref

            bar.set_postfix(loss=losses[-1], bleu4=bleu4_scores[-1])

    val_loss = sum(losses) / len(losses)
    val_bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothie.method4)

    return val_loss, val_bleu4, losses


def train(
    model: ImageCaptionModel,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.CrossEntropyLoss,
    start_epoch: int,
    n_epochs: int,
    tokenizer: AutoTokenizer,
    device: torch.device,
    model_path: str,
    log_path: str,
    early_stopping: int = 5,
) -> dict:

    scaler = GradScaler("cuda")
    model.train()

    if start_epoch > 0:
        log = json.load(open(log_path, "r"))
        best_train_bleu4, best_val_bleu4, best_epoch = (
            log["best_train_bleu4"],
            log["best_val_bleu4"],
            log["best_epoch"],
        )
        print("Load model from epoch {}, and continue training.".format(best_epoch))
        model.load_state_dict(torch.load(model_path, map_location=device))

    else:
        log = {
            "train_loss": [],
            "train_bleu4": [],
            "train_loss_batch": [],
            "val_loss": [],
            "val_bleu4": [],
            "val_loss_batch": [],
        }
        best_train_bleu4, best_val_bleu4, best_epoch = -np.inf, -np.inf, 1

    count_early_stopping = 0
    start_time = time.time()

    for epoch in range(start_epoch, n_epochs):
        try:
            train_loss, train_bleu4, train_loss_batch = train_epoch(
                model=model,
                train_loader=train_loader,
                tokenizer=tokenizer,
                optimizer=optimizer,
                loss_fn=loss_fn,
                current_epoch=epoch,
                device=device,
                scaler=scaler,
            )
            val_loss, val_bleu4, val_loss_batch = validate_epoch(
                model=model,
                valid_loader=valid_loader,
                tokenizer=tokenizer,
                loss_fn=loss_fn,
                current_epoch=epoch,
                device=device,
            )
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            traceback.print_exc()
            continue

        best_train_bleu4 = (
            train_bleu4 if train_bleu4 > best_train_bleu4 else best_train_bleu4
        )

        # Detect improvement and save model or early stopping and break
        if val_bleu4 > best_val_bleu4:
            best_val_bleu4 = val_bleu4
            best_epoch = epoch + 1

            # Save Model with best validation bleu4
            torch.save(model.state_dict(), model_path)
            print("-------- Detect improvement and save the best model --------")
            count_early_stopping = 0

        else:
            count_early_stopping += 1
            if count_early_stopping >= early_stopping:
                print("-------- Early stopping --------")
                break

        # Logfile
        log["train_loss"].append(train_loss)
        log["train_bleu4"].append(train_bleu4)
        log["train_loss_batch"].append(train_loss_batch)
        log["val_loss"].append(val_loss)
        log["val_bleu4"].append(val_bleu4)
        log["val_loss_batch"].append(val_loss_batch)
        log["best_train_bleu4"] = best_train_bleu4  # type: ignore
        log["best_val_bleu4"] = best_val_bleu4  # type: ignore
        log["best_epoch"] = best_epoch  # type: ignore
        log["last_epoch"] = epoch + 1  # type: ignore

        # Save log
        with open(log_path, "w") as f:
            json.dump(log, f, indent=4)

        torch.cuda.empty_cache()

        elapsed = timedelta(seconds=int(time.time() - start_time))
        print(
            f"Epoch [{epoch+1}/{n_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train BLEU4: {train_bleu4:.4f} | Val BLEU4: {val_bleu4:.4f} | Best Val BLEU4: {best_val_bleu4:.4f} | Time: {elapsed}"
        )

    return log


def main() -> None:
    try:
        parser = argparse.ArgumentParser()

        # Model parameters
        parser.add_argument(
            "--embed-dim",
            "-ed",
            type=int,
            default=512,
            help="Dimensionality of embeddings",
        )
        parser.add_argument(
            "--tokenizer-name",
            "-t",
            type=str,
            default="bert-base-uncased",
            help="Name of pretrained tokenizer",
        )
        parser.add_argument(
            "--max-seq-len",
            "-msl",
            type=int,
            default=128,
            help="Maximum sequence length",
        )
        parser.add_argument(
            "--encoder-layers",
            "-el",
            type=int,
            default=6,
            help="Number of encoder layers",
        )
        parser.add_argument(
            "--decoder-layers",
            "-dl",
            type=int,
            default=12,
            help="Number of decoder layers",
        )
        parser.add_argument(
            "--num-heads", "-nh", type=int, default=8, help="Number of attention heads"
        )
        parser.add_argument(
            "--dropout-rate", "-dr", type=float, default=0.1, help="Dropout rate"
        )

        # Training parameters
        parser.add_argument(
            "--model-path",
            "-mp",
            type=str,
            default="./pretrained/model_image_captioning_eff_transfomer.pt",
            help="Path to model file",
        )
        parser.add_argument(
            "--device", "-dv", type=str, default="cuda:0", help="Device to use"
        )
        parser.add_argument(
            "--batch-size", "-bs", type=int, default=24, help="Batch size"
        )
        parser.add_argument(
            "--num-epochs",
            "-ne",
            type=int,
            default=25,
            help="Number of training epochs",
        )
        parser.add_argument(
            "--start-epoch", "-se", type=int, default=0, help="Starting epoch index"
        )
        parser.add_argument(
            "--learning-rate",
            "-lr",
            type=float,
            default=1e-4,
            help="Learning rate for optimizer",
        )
        parser.add_argument(
            "--adam-betas",
            "-bt",
            type=tuple,
            default=(0.9, 0.999),
            help="Adam optimizer betas",
        )
        parser.add_argument(
            "--adam-eps",
            "-eps",
            type=float,
            default=1e-9,
            help="Adam optimizer epsilon value",
        )
        parser.add_argument(
            "--early-stopping",
            "-es",
            type=int,
            default=5,
            help="Patience for early stopping",
        )

        # Data parameters
        parser.add_argument(
            "--image-dir",
            "-id",
            type=str,
            default="./coco/",
            help="Directory containing images",
        )
        parser.add_argument(
            "--karpathy-json",
            "-kjp",
            type=str,
            default="./coco/karpathy/dataset_coco.json",
            help="Path to Karpathy JSON file",
        )
        parser.add_argument(
            "--val-annotations",
            "-vap",
            type=str,
            default="./coco/annotations/captions_val2014.json",
            help="Validation annotations file path",
        )
        parser.add_argument(
            "--train-annotations",
            "-tap",
            type=str,
            default="./coco/annotations/captions_train2014.json",
            help="Training annotations file path",
        )

        # Log parameters
        parser.add_argument(
            "--log-path",
            "-lp",
            type=str,
            default="./images/log_training.json",
            help="Training log file path",
        )
        parser.add_argument(
            "--log-vis-dir",
            "-lvd",
            type=str,
            default="./images/",
            help="Directory for log visualizations",
        )

        args = parser.parse_args()

        print("------------ Training parameters ----------------")

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        print(f"---DEBUG---\n{args}\n---DEBUG---")

        if not os.path.exists(args.log_vis_dir):
            print(f"Create directory {args.log_vis_dir}")
            os.makedirs(args.log_vis_dir)

        model_path_dir = os.path.dirname(args.model_path)

        if not os.path.exists(model_path_dir):
            print(f"Create directory {model_path_dir}")
            os.makedirs(model_path_dir)

        if not os.path.exists(args.image_dir):
            print(f"Directory image_dir {args.image_dir} does not exist")
            return

        print("-------------------------------------------------")

        device = torch.device(args.device)
        print("Using device: {}".format(device))

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        model = ImageCaptionModel(
            embedding_dim=args.embed_dim,
            vocab_size=tokenizer.vocab_size,
            max_seq_len=args.max_seq_len,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            num_heads=args.num_heads,
            dropout=args.dropout_rate,
        )
        model.to(device)
        print("Model to {}".format(device))

        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        optim = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=args.adam_betas,
            eps=args.adam_eps,
        )

        # Dataset
        train_dataset = ImageCaptionDataset(
            karpathy_json_path=args.karpathy_json,
            image_dir=args.image_dir,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            transform_fn=transform,
            phase="train",
        )
        valid_dataset = ImageCaptionDataset(
            karpathy_json_path=args.karpathy_json,
            image_dir=args.image_dir,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            transform_fn=transform,
            phase="val",
        )

        # DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Train
        start_time = time.time()

        log = train(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optim,
            loss_fn=criterion,
            start_epoch=args.start_epoch,
            n_epochs=args.num_epochs,
            tokenizer=tokenizer,
            device=device,
            model_path=args.model_path,
            log_path=args.log_path,
            early_stopping=args.early_stopping,
        )

        print(
            f"======================== Training finished: {timedelta(seconds=int(time.time()-start_time))} ========================"
        )
        print(
            f"---- Training | Best BLEU-4: {log['best_train_bleu4']:.5f} | Best Loss: {min(log['train_loss']):.5f}"
        )
        print(
            f"---- Validation | Best BLEU-4: {log['best_val_bleu4']:.5f} | Best Loss: {min(log['val_loss']):.5f}"
        )
        print(f"---- Best epoch: {log['best_epoch']}")

        # Save log
        json.dump(log, open(args.log_path, "w"))

        # Visualize loss and save to log_visualize_dir
        visualize_log(log, args.log_vis_dir)

    except Exception as e:
        print(f"Fatal error in training setup: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
