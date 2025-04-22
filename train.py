import os
import json
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Any
from datetime import timedelta
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from transformers.models.auto.tokenization_auto import AutoTokenizer

from utils import transform, visualize_log
from datasets import ImageCaptionDataset
from models import ImageCaptionModel


smoothie = SmoothingFunction()


def train_epoch(
    model: ImageCaptionModel,
    train_loader: torch.utils.data.DataLoader,
    tokenizer: AutoTokenizer,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    current_epoch: int,
    device: torch.device,
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
        images = batch["image"].to(device)
        caption_ids = batch["caption"].to(device)

        all_captions_seq = batch["all_captions_seq"]
        decoder_input = caption_ids[:, :-1]
        raw_logits = model(images, decoder_input)

        optimizer.zero_grad()

        target_tokens = caption_ids[:, 1:].contiguous().view(-1)
        loss = loss_fn(raw_logits.view(-1, raw_logits.size(-1)), target_tokens)

        loss.backward()
        optimizer.step()

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

        bleu4_scores.append(corpus_bleu(ref, hypo, smoothing_function=smoothie.method4))
        hypotheses += hypo
        references += ref

        bar.set_postfix(loss=losses[-1], bleu4=bleu4_scores[-1])

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
            images = batch["image"].to(device)
            caption_ids = batch["caption"].to(device)

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
        train_loss, train_bleu4, train_loss_batch = train_epoch(
            model=model,
            train_loader=train_loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            loss_fn=loss_fn,
            current_epoch=epoch,
            device=device,
        )
        val_loss, val_bleu4, val_loss_batch = validate_epoch(
            model=model,
            valid_loader=valid_loader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            current_epoch=epoch,
            device=device,
        )

        best_train_bleu4 = (
            train_bleu4 if train_bleu4 > best_train_bleu4 else best_train_bleu4
        )

        # Detect improvement and save model or early stopping and break
        if val_bleu4 > best_val_bleu4:
            best_val_bleu4 = val_bleu4
            best_epoch = epoch + 1

            # Save Model with best validation bleu4
            torch.save(model.state_dict(), model_path)
            print("-------- Detect improment and save the best model --------")
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

        print(
            f"---- Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.5f} | Valid Loss: {val_loss:.5f} | Train BLEU-4: {train_bleu4:.5f} | Validation BLEU-4: {val_bleu4:.5f} | Best BLEU-4: {best_val_bleu4:.5f} | Best Epoch: {best_epoch} | Time taken: {timedelta(seconds=int(time.time()-start_time))}"
        )

    return log


def main() -> None:

    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument(
        "--embedding_dim", "-ed", type=int, default=512, help="Embedding dimension"
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        default="bert-base-uncased",
        help="Bert tokenizer",
    )
    parser.add_argument(
        "--max_seq_len",
        "-msl",
        type=int,
        default=128,
        help="Maximum sequence length for caption generation",
    )
    parser.add_argument(
        "--encoder_layers",
        "-ad",
        type=int,
        default=6,
        help="Number of layers in the transformer encoder",
    )
    parser.add_argument(
        "--decoder_layers",
        "-nl",
        type=int,
        default=12,
        help="Number of layers in the transformer decoder",
    )
    parser.add_argument(
        "--num_heads",
        "-nh",
        type=int,
        default=8,
        help="Number of heads in multi-head attention",
    )
    parser.add_argument(
        "--dropout", "-dr", type=float, default=0.1, help="Dropout probability"
    )

    # Training parameters
    parser.add_argument(
        "--model_path",
        "-md",
        type=str,
        default="./pretrained/model_image_captioning_eff_transfomer.pt",
        help="Path to save model",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        help="Device to use {cpu, cuda:0, cuda:1,...}",
    )
    parser.add_argument("--batch_size", "-bs", type=int, default=24, help="Batch size")
    parser.add_argument(
        "--n_epochs", "-ne", type=int, default=25, help="Number of epochs"
    )
    parser.add_argument(
        "--start_epoch",
        "-se",
        type=int,
        default=0,
        help="Start epoch. If start_epoch > 0, load model from model_path and continue training",
    )
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--betas", "-bt", type=tuple, default=(0.9, 0.999), help="Adam optimizer betas"
    )
    parser.add_argument(
        "--eps", "-eps", type=float, default=1e-9, help="Adam optimizer epsilon"
    )
    parser.add_argument(
        "--early_stopping", "-es", type=int, default=5, help="Early stopping"
    )

    # Data parameters
    parser.add_argument(
        "--image_dir",
        "-id",
        type=str,
        default="./coco/",
        help="Path to image directory, this contains train2014, val2014",
    )
    parser.add_argument(
        "--karpathy_json_path",
        "-kap",
        type=str,
        default="./coco/karpathy/dataset_coco.json",
        help="Path to karpathy json file",
    )
    parser.add_argument(
        "--val_annotation_path",
        "-vap",
        type=str,
        default="./coco/annotations/captions_val2014.json",
        help="Path to validation annotation file",
    )
    parser.add_argument(
        "--train_annotation_path",
        "-tap",
        type=str,
        default="./coco/annotations/captions_train2014.json",
        help="Path to training annotation file",
    )

    # Log parameters
    parser.add_argument(
        "--log_path",
        "-lp",
        type=str,
        default="./images/log_training.json",
        help="Path to log file for training",
    )
    parser.add_argument(
        "--log_visualize_dir",
        "-lvd",
        type=str,
        default="./images/",
        help="Directory to save log visualization",
    )

    args = parser.parse_args()

    print("------------ Training parameters ----------------")

    print(f"---DEBUG---\n{args}\n---DEBUG---")

    if not os.path.exists(args.log_visualize_dir):
        print(f"Create directory {args.log_visualize_dir}")
        os.makedirs(args.log_visualize_dir)

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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = ImageCaptionModel(
        embedding_dim=args.embedding_dim,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model.to(device)
    print("Model to {}".format(device))

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optim = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps
    )

    # Dataset
    train_dataset = ImageCaptionDataset(
        karpathy_json_path=args.karpathy_json_path,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        transform_fn=transform,
        phase="train",
    )
    valid_dataset = ImageCaptionDataset(
        karpathy_json_path=args.karpathy_json_path,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        transform_fn=transform,
        phase="val",
    )

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False
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
        n_epochs=args.n_epochs,
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
    visualize_log(log, args.log_visualize_dir)


if __name__ == "__main__":
    main()
