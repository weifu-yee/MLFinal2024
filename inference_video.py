#!/usr/bin/env python
# coding: utf-8

# # Video as input

# In[5]:


import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# ===== HARD-CODED PARAMETERS =====
VIDEO_FOLDER = r"archive/dataset/dataset/test/video"
OUTPUT_FOLDER = r"video_results"
CHECKPOINT_PATH = r"ckpt/faster_rcnn_best.pth"
NUM_CLASSES = 4  # 包含 background
LABELS = ["background", "pothole", "cracks", "open_manhole"]
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ================================

# 支援的影片副檔名
VID_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}


def get_model(num_classes, checkpoint_path=None, device='cpu'):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
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


def draw_boxes_pil(image, boxes, labels, scores, category_names=None):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{category_names[label] if category_names else label}: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill="red")
        draw.text((x1, y1 - th), text, fill="white", font=font)
    return image


def process_video(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        boxes, labels, scores = predict(model, pil_img, DEVICE, THRESHOLD)
        inf_img = draw_boxes_pil(pil_img.copy(), boxes, labels, scores, LABELS)
        inf_frame = cv2.cvtColor(np.array(inf_img), cv2.COLOR_RGB2BGR)
        out.write(inf_frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[{os.path.basename(video_path)}] Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"Finished {os.path.basename(video_path)}, saved to {output_path}")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    model = get_model(NUM_CLASSES, CHECKPOINT_PATH, DEVICE)

    video_files = [f for f in os.listdir(VIDEO_FOLDER)
                   if os.path.splitext(f)[1].lower() in VID_EXTENSIONS]
    if not video_files:
        print(f"No videos found in {VIDEO_FOLDER}")
        return
    for vf in video_files:
        in_path = os.path.join(VIDEO_FOLDER, vf)
        out_name = os.path.splitext(vf)[0] + '_inf.mp4'
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        process_video(model, in_path, out_path)

if __name__ == "__main__":
    main()

