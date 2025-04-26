import os
import json
import time
import torch
import argparse
from datetime import timedelta
from transformers.models.bert import BertTokenizer

from utils import transform
from models import ImageCaptionModel
from evaluation import generate_caption


def main() -> None:
    parser = argparse.ArgumentParser()

    # Input options
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        default=None,
        help="Path to image file or directory",
    )
    parser.add_argument(
        "--captions-json",
        "-cj",
        type=str,
        default="./coco/karpathy/dataset_coco.json",
        help="Karpathy JSON for true captions",
    )

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
    parser.add_argument(
        "--beam-size", "-b", type=int, default=3, help="Beam size for beam search"
    )

    args = parser.parse_args()

    # Speed: cuDNN benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Load true captions mapping
    true_captions = {}
    if args.captions_json:
        try:
            data = json.load(open(args.captions_json, "r"))
        except Exception as e:
            print(f"Error loading captions JSON {args.captions_json}: {e}")
            data = {}

        for img in data.get("images", []):
            key = img.get("filename")
            caps = [s.get("raw") for s in img.get("sentences", [])]
            true_captions[key] = caps

    device = torch.device(args.device)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)

    # Load model ImageCaptionModel
    load_start_time = time.time()

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
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        return

    model.to(device)
    model.eval()
    model_load_duration = timedelta(seconds=int(time.time() - load_start_time))

    print(f"Model loaded on {device} in {model_load_duration}")

    def generate_and_display_caption(image_path):
        """Generate and display caption for a single image with timing information"""
        try:
            start = time.time()
            pred = generate_caption(
                model=model,
                image_path=image_path,
                transform_fn=transform,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
                beam_width=args.beam_size,
                device=device,
                print_process=False,
            )
            duration = time.time() - start

            print(f"Image: {image_path}")
            basename = os.path.basename(image_path)
            if basename in true_captions:
                for caption in true_captions[basename]:
                    print(f"  True   : {caption}")
            print(f"  Pred   : {pred}")
            print(f"  Time   : {duration:.3f}s")
            print("=" * 80)
            return True
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return False

    # Process input paths
    paths = []
    if args.input_path:
        if os.path.isdir(args.input_path):
            for root, _, files in os.walk(args.input_path):
                paths.extend(
                    [
                        os.path.join(root, f)
                        for f in files
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )
        elif os.path.exists(args.input_path):
            paths = [args.input_path]

    # Process all found images
    for image_path in paths:
        generate_and_display_caption(image_path)

    # Interactive mode if no paths provided or explicitly requested
    if not paths:
        print("Enter image path (or q to exit): ", end="")
        while True:
            user_input = input().strip()
            if user_input.lower() == "q":
                break

            generate_and_display_caption(user_input)


if __name__ == "__main__":
    main()
