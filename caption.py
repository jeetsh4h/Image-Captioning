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
        # default="./pretrained/model_image_captioning_eff_transfomer_final.pt",
        default="./pretrained/model_image_captioning_eff_transfomer_our.pt",
        help="Path to save model",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        help="Device to use {cpu, cuda:0, cuda:1,...}",
    )
    parser.add_argument(
        "--beam_size", "-b", type=int, default=3, help="Beam size for beam search"
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Load model ImageCaptionModel
    load_start_time = time.time()

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
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    model_load_duration = timedelta(seconds=int(time.time() - load_start_time))

    print(f"Model loaded on {device} in {model_load_duration}")

    # Generate captions
    # TODO: predict image for a directory of images
    while True:
        image_path: str = input("Enter image path (or q to exit): ")
        if image_path.lower() == "q":
            break
        loop_start_time = time.time()
        generated_caption: str = generate_caption(
            model=model,
            image_path=image_path,
            transform_fn=transform,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            beam_width=args.beam_size,
            device=device,
            print_process=False,
        )
        loop_end_time = time.time()
        print(f"--- Caption: {generated_caption}")
        print(f"--- Time: {loop_end_time - loop_start_time} s")


if __name__ == "__main__":
    main()
