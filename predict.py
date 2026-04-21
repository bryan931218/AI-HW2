"""
DETR inference: run trained model on test images and output pred.json in COCO format.

Output format:
[
    {"image_id": int, "bbox": [x_min, y_min, w, h], "score": float, "category_id": int},
    ...
]
bbox is in absolute pixel coordinates (no normalization), category_id starts from 1.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from model import build_model
from dataset import Compose, ToTensor, Normalize, RandomResize


def get_val_transforms(config):
    short_edge = config.get('val_short_edge', 320)
    max_size = config.get('val_max_size', 640)
    return Compose([
        RandomResize([short_edge], max_size=max_size),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def main(config_path, checkpoint_path, test_dir=None, output_path="pred.json",
         score_threshold=0.01, batch_size=1):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    device = torch.device(config["device"])

    model, _, postprocessors = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    if test_dir is None:
        test_dir = os.path.join(config["data_path"], "test")

    transforms = get_val_transforms(config)

    image_files = sorted(Path(test_dir).glob("*.png"))
    if not image_files:
        image_files = sorted(Path(test_dir).glob("*.jpg"))
    print(f"Found {len(image_files)} test images in {test_dir}")

    predictions = []

    for batch_start in tqdm(range(0, len(image_files), batch_size), desc="Running inference"):
        batch_paths = image_files[batch_start: batch_start + batch_size]

        tensors, orig_sizes, image_ids = [], [], []
        for img_path in batch_paths:
            image_ids.append(int(img_path.stem))
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            orig_sizes.append((orig_h, orig_w))
            img_tensor, _ = transforms(img, None)
            tensors.append(img_tensor)

        # Pass as list of individual tensors so that
        # nested_tensor_from_tensor_list generates a correct padding mask.
        tensor_list = [t.to(device) for t in tensors]
        outputs = model(tensor_list)

        target_sizes = torch.tensor(orig_sizes, device=device)
        results = postprocessors['bbox'](outputs, target_sizes)

        for result, image_id in zip(results, image_ids):
            scores = result['scores']
            labels = result['labels']
            boxes = result['boxes']

            keep = (scores > score_threshold) & (labels >= 1) & (labels <= 10)
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            for s, l, b in zip(scores, labels, boxes):
                x0, y0, x1, y1 = b.tolist()
                predictions.append({
                    "image_id": image_id,
                    "bbox": [round(x0, 4), round(y0, 4),
                             round(x1 - x0, 4), round(y1 - y0, 4)],
                    "score": round(s.item(), 6),
                    "category_id": l.item(),
                })

    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    print(f"Saved {len(predictions)} predictions to {output_path}")




def find_latest_checkpoint(save_dir):
    save_dir = Path(save_dir)
    candidates = sorted(save_dir.glob("*/best.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(save_dir.glob("best.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No best.pth found under {save_dir}")
    return candidates[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Integrated DINO-style DETR inference')
    parser.add_argument('--data_root', default='./nycu-hw2-data', type=str)
    parser.add_argument('--save_dir', default='./outputs', type=str)
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--test_dir', default='', type=str)
    parser.add_argument('--output', default='', type=str)
    parser.add_argument('--score_threshold', default=0.01, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--zip', action='store_true')
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(args.save_dir)
    config = Path(args.config) if args.config else Path(args.save_dir) / 'config.json'
    if not config.exists():
        ckpt = torch.load(checkpoint, map_location='cpu')
        ckpt_config = ckpt.get('config')
        if ckpt_config is None:
            raise FileNotFoundError(f'Config not found: {config}')
        config.parent.mkdir(parents=True, exist_ok=True)
        with open(config, 'w', encoding='utf-8') as f:
            json.dump(ckpt_config, f, indent=2)
    test_dir = args.test_dir or str(Path(args.data_root) / 'test')
    output = args.output or str(Path(args.save_dir) / 'pred.json')
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    main(str(config), str(checkpoint), test_dir, output, args.score_threshold, args.batch_size)

    if args.zip:
        import subprocess
        zip_path = str(Path(output).with_suffix('.zip'))
        subprocess.run(['zip', '-j', zip_path, output], check=True)
        print(f'Zip saved to {zip_path}')
