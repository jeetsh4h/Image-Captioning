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
        default="./pretrained/model_image_captioning_eff_transfomer_our.pt",
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
