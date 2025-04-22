import os
import json
import torch
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers.models.bert import BertTokenizer

from utils import transform


def create_image_inputs(karpathy_json_path: str, image_dir: str, transform_fn) -> None:
    """
    This function preprocesses the image and saves it in the image_dir.
    It's faster for the training process to load the images from the torch file.
    """

    karpathy = json.load(open(karpathy_json_path, "r"))
    bar = tqdm(karpathy["images"])

    for image in bar:
        image_path = os.path.join(image_dir, image["filepath"], image["filename"])
        image = Image.open(image_path).convert("RGB")
        image = transform_fn(image)
        torch.save(image, image_path.replace(".jpg", ".pt"))


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        karpathy_json_path: str,
        image_dir: str,
        tokenizer: BertTokenizer,  # fixed for now.
        max_seq_len: int = 256,
        transform_fn=None,
        phase: str = "train",
    ) -> None:
        self.transform = transform_fn
        self.transform_fn = transform_fn
        self.tokenizer = tokenizer
        self.karpathy_json_path = karpathy_json_path
        self.image_dir = image_dir
        self.max_seq_len = max_seq_len
        self.phase = phase
        self.df = self.create_inputs()

    def create_inputs(self):
        data: dict | None = None
        with open(self.karpathy_json_path, "r") as f:
            data = json.load(f)
        if data is None:
            raise ValueError("Karpathy JSON file is empty or not found.")

        df: list[dict] = []
        for image in data["images"]:
            image_path = os.path.join(
                self.image_dir, image["filepath"], image["filename"]
            )
            captions = [" ".join(c["tokens"]) for c in image["sentences"]]

            for caption in captions:
                row = {
                    "image_id": image["cocoid"],
                    "image_path": image_path,
                    "caption": caption,
                    "all_captions": captions + [""] * (10 - len(captions)),
                }

                # Only include images based on the current phase and split
                valid_splits = {
                    "train": {"train", "restval"},
                    "val": {"val"},
                    "test": {"test"},
                }
                if image["split"] in valid_splits.get(self.phase, set()):
                    df.append(row)

        return pd.DataFrame(df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        image_path = self.df.iloc[index]["image_path"]
        image_torch_path = image_path.replace(".jpg", ".pt")

        if os.path.exists(image_torch_path):
            image = torch.load(image_torch_path)

        else:
            image = Image.open(image_path).convert("RGB")

            if self.transform_fn is not None:
                image = self.transform_fn(image)

            torch.save(image, image_torch_path)

        caption = self.df.loc[index, "caption"]
        caption_tokens = self.tokenizer(
            caption,  # type: ignore
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"][0]

        all_captions = self.df.loc[index, "all_captions"]
        all_captions_tokens = self.tokenizer(
            all_captions,  # type: ignore
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        return {
            "image_id": self.df.loc[index, "image_id"],
            "image_path": image_path,
            "image": image,
            "caption_seq": caption,
            "caption": caption_tokens,
            "all_captions_seq": all_captions,
            "all_captions": all_captions_tokens,
        }


def main() -> None:
    parser = argparse.ArgumentParser()

    # Model parameters
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
    parser.add_argument("--batch-size", "-bs", type=int, default=16, help="Batch size")

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
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
    dataset = ImageCaptionDataset(
        karpathy_json_path=args.karpathy_json,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        transform_fn=transform,
        phase="train",
    )
    print(dataset[0])

    create_image_inputs(args.karpathy_json, args.image_dir, transform)


# test the dataset class
if __name__ == "__main__":
    main()
