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
    image = Image.open(image_path).convert("RGB")
    image = transform_fn(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    # Generate caption
    with torch.no_grad():
        # Feed forward Encoder
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

                # Decoder forward pass
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


# Evaluate model on test dataset
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
        default=3,
        help="Number of layers in the transformer encoder",
    )
    parser.add_argument(
        "--decoder_layers",
        "-nl",
        type=int,
        default=6,
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
        default="./pretrained/model_image_captioning_eff_transfomer_final.pt",
        help="Path to save model",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        help="Device to use {cpu, cuda:0, cuda:1,...}",
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
        default="./coco/annotations/annotations/captions_val2014.json",
        help="Path to validation annotation file",
    )
    parser.add_argument(
        "--train_annotation_path",
        "-tap",
        type=str,
        default="./coco/annotations/annotations/captions_train2014.json",
        help="Path to training annotation file",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="./results/",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Load tokenizer
    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Load model
    start_time = time.time()
    model_configs = {
        "embedding_dim": args.embedding_dim,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": args.max_seq_len,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
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
    kaparthy = json.load(open(args.karpathy_json_path, "r"))

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
    ann = convert_karpathy_to_coco_format(
        karpathy_path=args.karpathy_json_path, annotation_path=args.val_annotation_path
    )
    ann_path = os.path.join(args.output_dir, "coco_annotation_test.json")
    json.dump(ann, open(ann_path, "w"))

    # Evaluate model on test dataset with beam search and save results
    beam_widths: list[int] = [args.beam_size]
    metrics: dict[str, Any] = {}

    for b in beam_widths:
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

        # Save prediction
        predict_path = os.path.join(args.output_dir, f"prediction_beam_width_{b}.json")
        json.dump(predictions, open(predict_path, "w"))

        # Calculate metrics
        result = metric_scores(annotation_path=ann_path, prediction_path=predict_path)
        metrics[f"beam{b}"] = result

    # print metrics scores and save scores
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    json.dump(metrics, open(metrics_path, "w"))

    print("--------------------- Done ----------------------------")
    print(f"Metrics saved in {metrics_path}")
    for b in beam_widths:
        print(
            f"Beam width {b}, predictions saved in "
            f"{os.path.join(args.output_dir, f'prediction_beam_width_{b}.json')}"
        )
        result = metrics[f"beam{b}"]
        for metric_name, metric_value in result.items():
            print(f"----- {metric_name}: {metric_value}")
    print("-------------------------------------------------------")


if __name__ == "__main__":
    main()
