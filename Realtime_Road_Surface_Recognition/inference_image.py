#!/usr/bin/env python
# coding: utf-8

# # Image as input

import os
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


# ===== HARD-CODED PARAMETERS =====
IMAGE_FOLDER = r"archive/dataset/dataset/test/images"
OUTPUT_FOLDER = r"results"
CHECKPOINT_PATH = r"ckpt/faster_rcnn_best.pth"
NUM_CLASSES = 4  # 包含 background
LABELS = ["background", "pothole", "cracks", "open_manhole"]
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================

# 支援的影像副檔名
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def get_model(num_classes, checkpoint_path=None, device='cpu'):
    # model = fasterrcnn_resnet50_fpn(pretrained=False)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    num_classes = 4  # 3 classes (pothole, cracks, open_manhole) + 1 background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model, image, device, threshold=0.5):
    img_tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        outputs = model([img_tensor])[0]

    keep = outputs['scores'] >= threshold
    boxes = outputs['boxes'][keep].cpu()
    labels = outputs['labels'][keep].cpu()
    scores = outputs['scores'][keep].cpu()
    return boxes, labels, scores


def draw_boxes(image, boxes, labels, scores, category_names=None, text_scale=3.0):
    draw = ImageDraw.Draw(image)
    # font = ImageFont.load_default()
    
    # 1. 定義一個 base_size（像素），再乘上 text_scale 得到實際字體大小
    base_size = 12
    font_size = int(base_size * text_scale)
    # 2. 載入一支 TrueType 字型 (這裡以 arial.ttf 為例，實際要改成你的字型路徑)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # 如果找不到 arial.ttf，就 fallback 回 load_default
        font = ImageFont.load_default()
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{category_names[label] if category_names else label}: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
        draw.text((x1, y1 - text_height), text, fill="white", font=font)
    return image


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    model = get_model(NUM_CLASSES, CHECKPOINT_PATH, DEVICE)

    for fname in os.listdir(IMAGE_FOLDER):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTENSIONS:
            continue
        image_path = os.path.join(IMAGE_FOLDER, fname)
        image = Image.open(image_path).convert("RGB")

        boxes, labels, scores = predict(model, image, DEVICE, THRESHOLD)
        result = draw_boxes(image.copy(), boxes, labels, scores, LABELS)

        out_path = os.path.join(OUTPUT_FOLDER, fname)
        result.save(out_path)
        print(f"Processed {fname} -> {out_path}")

if __name__ == "__main__":
    main()
