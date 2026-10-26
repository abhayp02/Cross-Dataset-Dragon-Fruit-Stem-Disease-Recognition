"""Train reproducible within-dataset and cross-dataset stem disease models."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SOURCES = {
    "a": "dragonfruit_multiorgan_stem_v1",
    "b": "common_dragon_fruit_stem_disease_v2",
    "c": "dragon_fruit_stem_disease_v1",
}
EXPERIMENTS = {
    "a_multiclass": {"train": ["a"], "test": "a", "labels": ["healthy", "rust", "scab", "stem_canker", "sunburn"]},
    "b_multiclass": {"train": ["b"], "test": "b", "labels": ["anthracnose", "cactus_virus_x", "healthy", "stem_canker"]},
    "c_multiclass": {"train": ["c"], "test": "c", "labels": ["anthracnose", "brown_stem_spot", "gray_blight", "healthy", "soft_rot", "stem_canker"]},
    "a_to_c_binary": {"train": ["a"], "test": "c", "labels": ["healthy", "stem_canker"]},
    "b_to_c_binary": {"train": ["b"], "test": "c", "labels": ["healthy", "stem_canker"]},
    "ab_to_c_binary": {"train": ["a", "b"], "test": "c", "labels": ["healthy", "stem_canker"]},
}
MODEL_CHOICES = ("mobilenet_v3_small", "efficientnet_b0", "resnet18")


class ImageRecords(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label = self.rows[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def load_rows(data_root: Path, source_key: str, splits: set[str], labels: dict[str, int]):
    root = data_root / "sources" / SOURCES[source_key]
    rows = []
    with (root / "manifest.csv").open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] in splits and row["standard_label"] in labels:
                rows.append((root / row["file"], labels[row["standard_label"]]))
    return rows


def evaluate(model, loader, device, label_names):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for images, labels in loader:
            predicted.extend(model(images.to(device)).argmax(1).cpu().tolist())
            actual.extend(labels.tolist())
    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "macro_precision": precision_score(actual, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(actual, predicted, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(len(label_names)))).tolist(),
        "classification_report": classification_report(actual, predicted, labels=list(range(len(label_names))), target_names=label_names, output_dict=True, zero_division=0),
    }
    return metrics


def create_model(model_name, class_count):
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, class_count)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_count)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    experiment = EXPERIMENTS[args.experiment]
    labels = {name: index for index, name in enumerate(experiment["labels"])}
    data_root = Path("data")
    train_rows = sum((load_rows(data_root, source, {"train"}, labels) for source in experiment["train"]), [])
    valid_rows = sum((load_rows(data_root, source, {"valid"}, labels) for source in experiment["train"]), [])
    test_splits = {"test"} if not args.experiment.endswith("_to_c_binary") else {"train", "valid", "test"}
    test_rows = load_rows(data_root, experiment["test"], test_splits, labels)
    if not train_rows or not valid_rows or not test_rows:
        raise RuntimeError("No usable records found. Check the source folders and manifests.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    eval_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    loader_args = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(ImageRecords(train_rows, train_transform), shuffle=True, **loader_args)
    valid_loader = DataLoader(ImageRecords(valid_rows, eval_transform), shuffle=False, **loader_args)
    test_loader = DataLoader(ImageRecords(test_rows, eval_transform), shuffle=False, **loader_args)

    model = create_model(args.model, len(labels))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    output = Path("model") / args.model / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    best_path, best_f1, history = output / "best_model.pt", -1.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * targets.size(0)
        validation = evaluate(model, valid_loader, device, experiment["labels"])
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_rows), **{key: validation[key] for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall")}}
        history.append(row)
        print(json.dumps(row))
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "model": args.model, "labels": labels, "validation": validation}, best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["state_dict"])
    test = evaluate(model, test_loader, device, experiment["labels"])
    result = {
        "experiment": args.experiment,
        "model": args.model,
        "device": str(device),
        "labels": labels,
        "record_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "external_test_uses_full_C": args.experiment.endswith("_to_c_binary"),
        "test_metrics": test,
        "history": history,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Final test metrics:", json.dumps({key: test[key] for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall")}, indent=2))
    print("Saved model:", best_path)


if __name__ == "__main__":
    main()
