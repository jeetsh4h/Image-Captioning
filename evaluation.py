import os
import copy
import json
import time
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from datetime import timedelta
import torch.nn.functional as F
from typing import Any, Callable
from torch.amp.autocast_mode import autocast
from transformers.models.bert import BertTokenizer

from models import ImageCaptionModel
from utils import transform, metric_scores, convert_karpathy_to_coco_format


def generate_caption(
    model: ImageCaptionModel,
    image_path: str,
    transform_fn: Callable[[Image.Image], torch.Tensor],
    tokenizer: BertTokenizer,
    max_seq_len: int = 256,
    beam_width: int = 3,
    device: torch.device = torch.device("cpu"),
    print_process: bool = False,
) -> str:
    """
    Generate caption for an image.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return ""

    image = transform_fn(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    # Generate caption
    with torch.no_grad():
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True

            with autocast(str(device)):
                encoder_output = model.encoder(image)

            # Initialize beam list
            candidate_beams = [([tokenizer.cls_token_id], 0)]
            completed_beams = []

            # Start decoding
            for _ in range(max_seq_len):
                updated_beams = []
                for sequence, score in candidate_beams:
                    # Get input token
                    input_token = torch.tensor([sequence]).to(device)

                    # Create mask
                    target_mask = model.make_mask(input_token).to(device)

                    # Decoder forward pass with autocast
                    with autocast(str(device)):
                        pred = model.decoder(input_token, encoder_output, target_mask)

                    # Forward to linear classify token in vocab and Softmax
                    pred = F.softmax(model.fc(pred), dim=-1)

                    # Get tail predict token
                    pred = pred[:, -1, :].view(-1)

                    # Get top k tokens
                    top_k_scores, top_k_tokens = pred.topk(beam_width)

                    # Update beams
                    for i in range(beam_width):
                        updated_beams.append(
                            (
                                sequence + [top_k_tokens[i].item()],
                                score + top_k_scores[i].item(),
                            )
                        )

                candidate_beams = sorted(
                    copy.deepcopy(updated_beams), key=lambda x: x[1], reverse=True
                )[:beam_width]

                # Add completed beams to completed list and reduce beam size
                for sequence, score in list(candidate_beams):
                    if sequence[-1] == tokenizer.sep_token_id:
                        completed_beams.append((sequence, score))
                        candidate_beams.remove((sequence, score))
                        beam_width -= 1

                # Print screen progress
                if print_process:
                    print(f"Step {_+1}/{max_seq_len}")
                    print(f"Beam width: {beam_width}")
                    print(
                        f"Beams: {[tokenizer.decode(beam[0]) for beam in candidate_beams]}"
                    )
                    print(
                        f"Completed beams: {[tokenizer.decode(beam[0]) for beam in completed_beams]}"
                    )
                    print(f"Beams score: {[beam[1] for beam in candidate_beams]}")
                    print("-" * 100)

                if beam_width == 0:
                    break

            # If no completed beams produced, fall back to current candidates
            if not completed_beams:
                completed_beams = candidate_beams

            # Sort the completed beams
            completed_beams.sort(key=lambda x: x[1], reverse=True)

            # Get best sequence tokens
            target_tokens = completed_beams[0][0]

            # Convert target sentence from tokens to string
            caption = tokenizer.decode(target_tokens, skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error during caption generation for {image_path}: {e}")
            return ""


# Evaluate model on test dataset
def main() -> None:
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument(
        "--embed-dim", "-ed", type=int, default=512, help="Dimensionality of embeddings"
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
        help="Maximum sequence length for caption generation",
    )
    parser.add_argument(
        "--encoder-layers",
        "-el",
        type=int,
        default=3,
        help="Number of encoder layers",
    )
    parser.add_argument(
        "--decoder-layers",
        "-dl",
        type=int,
        default=6,
        help="Number of decoder layers",
    )
    parser.add_argument(
        "--num-heads",
        "-nh",
        type=int,
        default=8,
        help="Number of attention heads",
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
        "--device",
        "-dv",
        type=str,
        default="cuda:0",
        help="Device to use",
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

    # Evaluation parameters
    parser.add_argument(
        "--beam-size",
        "-bs",
        type=int,
        default=3,
        help="Beam size for beam search",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        "-od",
        type=str,
        default="./results/",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Speed optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Load tokenizer
    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)

    # Load model
    start_time = time.time()
    model_configs = {
        "embedding_dim": args.embed_dim,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": args.max_seq_len,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout_rate,
    }
    model = ImageCaptionModel(**model_configs)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device), strict=False
    )
    model.to(device)

    model.eval()

    time_load_model = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"Done load model on the {device} device in {time_load_model}")

    # Load test dataset
    try:
        kaparthy = json.load(open(args.karpathy_json, "r"))
    except Exception as e:
        print(f"Error loading Karpathy JSON {args.karpathy_json}: {e}")
        return

    image_paths = [
        os.path.join(args.image_dir, image["filepath"], image["filename"])
        for image in kaparthy["images"]
        if image["split"] == "test"
    ]
    image_ids = [
        image["cocoid"] for image in kaparthy["images"] if image["split"] == "test"
    ]

    # Evaluate model
    model.eval()

    # convert karpathy json to coco format for evaluation
    ann_path = os.path.join(args.output_dir, "coco_annotation_test.json")
    if not os.path.exists(ann_path):
        ann = convert_karpathy_to_coco_format(
            karpathy_path=args.karpathy_json, annotation_path=args.val_annotations
        )
        json.dump(ann, open(ann_path, "w"))
    else:
        print(f"Annotation file already exists at {ann_path}, skipping conversion.")

    # Evaluate model on test dataset with beam search and save results
    beam_widths: list[int] = [args.beam_size]
    metrics: dict[str, Any] = {}

    for b in beam_widths:
        try:
            predict_path = os.path.join(
                args.output_dir, f"prediction_beam_width_{b}.json"
            )
            if os.path.exists(predict_path):
                print(
                    f"Prediction file for beam width {b} exists at {predict_path}, skipping generation."
                )
            else:
                predictions: list[dict[str, Any]] = []
                for image_path, image_id in tqdm(
                    zip(image_paths, image_ids), total=len(image_paths)
                ):
                    caption = generate_caption(
                        model=model,
                        image_path=image_path,
                        transform_fn=transform,
                        tokenizer=tokenizer,
                        max_seq_len=args.max_seq_len,
                        beam_width=b,
                        device=device,
                    )
                    predictions.append({"image_id": image_id, "caption": caption})

                # Save predictions to JSON file
                with open(predict_path, "w") as f:
                    json.dump(predictions, f)

        except Exception as e:
            print(f"Error processing beam width {b}: {e}")
            continue

        # Compute metrics
        try:
            result = metric_scores(
                annotation_path=ann_path, prediction_path=predict_path
            )
            metrics[f"beam{b}"] = result
        except Exception as e:
            print(f"Error computing metrics for beam{b}: {e}")
            metrics[f"beam{b}"] = {}

    # print metrics scores and save scores
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    json.dump(metrics, open(metrics_path, "w"))

    print("\n====== Evaluation Metrics ======")
    for b in beam_widths:
        print(f"Beam Width {b}")
        for metric_name, metric_value in metrics[f"beam{b}"].items():
            print(f"  {metric_name:15}: {metric_value:.4f}")

        print("-" * 40)

    # Save summary to text file
    summary_path = os.path.join(args.output_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:

        f.write("Evaluation Summary\n")
        for b in beam_widths:

            f.write(f"Beam Width {b}\n")
            for metric_name, metric_value in metrics[f"beam{b}"].items():
                f.write(f"  {metric_name:15}: {metric_value:.4f}\n")

            f.write("-" * 40 + "\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
