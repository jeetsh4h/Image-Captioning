import os
import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torchvision import transforms
from pycocoevalcap.eval import COCOEvalCap


transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def visualize_log(
    training_log: dict[str, list[float] | list[list[float]]], output_dir: str
) -> None:
    # Plot loss per epoch
    plt.figure()

    plt.plot(training_log["train_loss"], label="train")
    plt.plot(training_log["val_loss"], label="val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per epoch")

    plt.savefig(os.path.join(output_dir, "loss_epoch.png"))

    # Plot bleu4 per epoch
    plt.figure()

    plt.plot(training_log["train_bleu4"], label="train")
    plt.plot(training_log["val_bleu4"], label="val")

    plt.xlabel("Epoch")
    plt.ylabel("Bleu4")
    plt.legend()
    plt.title("BLEU-4 per epoch")

    plt.savefig(os.path.join(output_dir, "bleu4_epoch.png"))

    # Plot loss per batch
    plt.figure()

    train_loss_batch: list[float] = []
    for loss in training_log["train_loss_batch"]:
        train_loss_batch += loss  # type: ignore

    plt.plot(train_loss_batch, label="train")

    val_loss_batch: list[float] = []
    for loss in training_log["val_loss_batch"]:
        val_loss_batch += loss  # type: ignore

    plt.plot(val_loss_batch, label="val")

    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per batch")

    plt.savefig(os.path.join(output_dir, "loss_batch.png"))


def metric_scores(annotation_path: str, prediction_path: str) -> dict[str, float]:
    scores: dict[str, float] = {}

    annotation_coco = COCO(annotation_path)
    predictions_coco = annotation_coco.loadRes(prediction_path)

    evaluator = COCOEvalCap(annotation_coco, predictions_coco)
    evaluator.params["image_id"] = predictions_coco.getImgIds()
    evaluator.evaluate()

    for metric_name, score_value in evaluator.eval.items():
        print(f"{metric_name}: {score_value:.3f}")
        scores[metric_name] = score_value

    return scores


def convert_karpathy_to_coco_format(
    karpathy_path: str,
    annotation_path: str,
    split: str = "test",
) -> dict:
    assert split in {"train", "val", "test"}
    process_phases = {"train", "restval"} if split == "train" else {split}

    with open(karpathy_path, "r") as f:
        karpathy_data = json.load(f)

    selected_image_ids = {
        item["cocoid"]
        for item in karpathy_data["images"]
        if item["split"] in process_phases
    }

    with open(annotation_path, "r") as f:
        annotation_data = json.load(f)

    filtered_images = [
        image
        for image in annotation_data["images"]
        if image["id"] in selected_image_ids
    ]
    filtered_annotations = [
        annotation
        for annotation in annotation_data["annotations"]
        if annotation["image_id"] in selected_image_ids
    ]

    annotation_data["images"] = filtered_images
    annotation_data["annotations"] = filtered_annotations

    return annotation_data
