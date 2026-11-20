"""Opt-in integration check against the repository's real checkpoint and test images."""

import json
import os
from pathlib import Path
import sys

import pytest
import torch
from PIL import Image
from torchvision import models, transforms

# Keep the backend package distinct from the repository-root app.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config
from app.services.classifier import ClassifierService


def test_real_checkpoint_matches_independent_resnet18_reference(monkeypatch) -> None:
    if os.getenv("RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("Set RUN_REAL_MODEL_TESTS=1 to load the real checkpoint and images.")

    project_root = Path(__file__).resolve().parents[2]
    classes = json.loads(config.MODEL_CLASSES_PATH.read_text(encoding="utf-8"))
    dataset_root = project_root / "datasets" / "split_v2" / "test"
    selected: list[tuple[str, Path]] = []
    counts: dict[str, int] = {}
    missing: list[str] = []
    for label in classes:
        class_dir = dataset_root / label
        if not class_dir.is_dir():
            print(f"SKIP CLASS {label}: directory not found: {class_dir}")
            missing.append(label)
            counts[label] = 0
            continue
        limit = 10 if label == "trash" else 5
        files = sorted((path for path in class_dir.iterdir() if path.is_file()), key=lambda path: path.name)
        selected_for_class = files[:limit]
        counts[label] = len(selected_for_class)
        selected.extend((label, path) for path in selected_for_class)

    print("IMAGE_COUNTS " + json.dumps(counts, ensure_ascii=False, sort_keys=True))
    print("MISSING_CLASS_DIRECTORIES " + json.dumps(missing, ensure_ascii=False))
    if not selected:
        pytest.skip("No real test images were found in any class directory.")

    monkeypatch.setattr(config, "MODEL_TOP_K", 6)
    service = ClassifierService()
    service.load()

    # Construct and load a separate model as the reference; do not use service.model.
    reference_classes = json.loads(config.MODEL_CLASSES_PATH.read_text(encoding="utf-8"))
    reference_model = models.resnet18(weights=None)
    reference_model.fc = torch.nn.Linear(reference_model.fc.in_features, len(reference_classes))
    reference_model.load_state_dict(
        torch.load(
            config.MODEL_CHECKPOINT_PATH,
            map_location=service.device,
            weights_only=True,
        )
    )
    reference_model.to(service.device)
    reference_model.eval()
    weight_transform = models.ResNet18_Weights.DEFAULT.transforms()
    reference_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=weight_transform.mean, std=weight_transform.std),
        ]
    )

    top1_different = 0
    order_different = 0
    max_confidence_difference = 0.0
    print("file\ttrue_label\ttop3\tmax_confidence_difference")
    for true_label, image_path in selected:
        actual = service.predict(image_path.read_bytes())
        with Image.open(image_path) as image:
            input_tensor = reference_transform(image.convert("RGB")).unsqueeze(0).to(service.device)
        with torch.inference_mode():
            probabilities = torch.softmax(reference_model(input_tensor), dim=1)[0]
            values, indices = torch.topk(probabilities, k=len(reference_classes))
        reference = [
            {"label": reference_classes[int(index)], "confidence": float(value)}
            for value, index in zip(values.tolist(), indices.tolist())
        ]

        assert len(actual) == 6
        assert len(reference) == 6
        actual_pairs = [(item["label"], item["confidence"]) for item in actual]
        reference_pairs = [(item["label"], item["confidence"]) for item in reference]
        assert [label for label, _ in actual_pairs] == [label for label, _ in reference_pairs]
        differences = [abs(left[1] - right[1]) for left, right in zip(actual_pairs, reference_pairs)]
        image_max_difference = max(differences)
        assert all(abs(left[1] - right[1]) <= 1e-6 for left, right in zip(actual_pairs, reference_pairs))
        max_confidence_difference = max(max_confidence_difference, image_max_difference)
        top1_different += actual[0]["label"] != true_label
        order_different += [item["label"] for item in actual] != [item["label"] for item in reference]
        top3 = "; ".join(f"{item['label']}={item['confidence']:.8f}" for item in actual[:3])
        print(f"{image_path.relative_to(project_root)}\t{true_label}\t{top3}\t{image_max_difference:.10g}")

    print(f"TOP1_DIFFERENT_IMAGES {top1_different}")
    print(f"ORDER_DIFFERENT_IMAGES {order_different}")
    print(f"MAX_CONFIDENCE_DIFFERENCE_OVER_IMAGE_X_6_CLASS_PAIRS {max_confidence_difference:.10g}")
