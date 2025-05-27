#!/usr/bin/env python
# coding: utf-8

import os
import re
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

# 請將 train.py 放在同目錄，並確保裡面有以下定義：
#   - COCODataset
#   - get_transform()
#   - collate_fn
#   - validate_one_epoch(model, data_loader, device)
from tools.train_library import COCODataset, get_transform, collate_fn, validate_one_epoch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立模型函式
def build_model(num_classes=4):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return model

# 建立 validation DataLoader
val_dataset = COCODataset(
    annotation_file='archive/valid_annotations1.json',
    image_dir='archive/dataset/dataset/valid/images',
    transforms=get_transform()
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

# 指定存放 checkpoints 的資料夾
checkpoint_dir = 'ckpt'  # <- 改成你的資料夾路徑

best_acc = -1.0
best_epoch = None

# 依檔名找出 epoch number 並評估
for fn in os.listdir(checkpoint_dir):
    if fn.endswith('.pth'):
        m = re.search(r'faster_rcnn_epoch[_\-]?(\d+)\.pth$', fn)
        if not m:
            continue
        epoch_num = int(m.group(1))
        path = os.path.join(checkpoint_dir, fn)

        # 載入模型
        model = build_model()
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)

        # 驗證
        _, acc = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch_num}: Validation Accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch_num

if best_epoch is not None:
    print(f"\n>>> Best checkpoint: epoch {best_epoch}  (Accuracy = {best_acc:.4f})")
else:
    print("No valid checkpoint files found.")
