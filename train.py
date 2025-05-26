#!/usr/bin/env python
# coding: utf-8

## Declarate Dataset
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou # 用於計算 IoU
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
import os
from PIL import Image
from tqdm import tqdm
import json

# IO Parameter to control debug messages
VERBOSE = True # 設定為 True 來印出偵錯訊息，False 則關閉

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_dir, transforms=None):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        image_info = self.coco.loadImgs(image_id)[0]

        # Load image
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Process annotations
        boxes = []
        labels = []
        for ann in annotations:
            x, y, width, height = ann["bbox"]
            if width > 0 and height > 0:  # Only add valid boxes
                boxes.append([x, y, x + width, y + height])
                labels.append(ann["category_id"])

        # Convert to tensor
        if len(boxes) == 0:  # Handle no annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

## Declarate DataLoader
# Define paths
train_annotation_file = 'archive/train_annotations1.json'
val_annotation_file = 'archive/valid_annotations1.json'
train_image_dir = 'archive/dataset/dataset/train/images'
val_image_dir = 'archive/dataset/dataset/valid/images'

# Define transformations
def get_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

# Create datasets
train_dataset = COCODataset(train_annotation_file, train_image_dir, transforms=get_transform())
val_dataset = COCODataset(val_annotation_file, val_image_dir, transforms=get_transform())

# DataLoaders
def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

## Loading the pre-trained Faster R-CNN model
# Load a pre-trained Faster R-CNN model
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
num_classes = 4  # 3 classes (pothole, cracks, open_manhole) + 1 background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if VERBOSE:
    print(f"Using device: {device}")

## Validate data
def validate_data(image_dir, annotation_file):
    # Load annotations from the JSON file
    with open(annotation_file) as f:
        annotations = json.load(f)

    # Get image file names from the annotation data
    image_files = {image['file_name'] for image in annotations['images']}

    # Check if image files exist
    missing_images = []
    for image_file in image_files:
        if not os.path.exists(os.path.join(image_dir, image_file)):
            missing_images.append(image_file)

    if missing_images:
        if VERBOSE:
            print(f"Missing images: {missing_images}")
        return False

    # Check if all bounding boxes are valid
    invalid_bboxes = []
    for annotation in annotations['annotations']:
        x, y, w, h = annotation['bbox']
        if w <= 0 or h <= 0:
            invalid_bboxes.append(annotation['id'])

    if invalid_bboxes:
        if VERBOSE:
            print(f"Invalid bounding boxes for annotations: {invalid_bboxes}")
        return False
    
    if VERBOSE:
        print(f"Data validation successful for {annotation_file}!")
    return True

# Validate train and validation data
if not validate_data(train_image_dir, train_annotation_file):
    print("Training data validation failed. Please check your images and annotations.")
    exit() # Exit if validation fails
elif not validate_data(val_image_dir, val_annotation_file):
    print("Validation data validation failed. Please check your images and annotations.")
    exit() # Exit if validation fails
else:
    if VERBOSE:
        print("Data validation successful. Starting training...")

## Define one epoch training and validation functions
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(data_loader, desc="Training"):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()

    avg_epoch_loss = epoch_loss / len(data_loader)
    return avg_epoch_loss

def validate_one_epoch(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()  # 將模型設置為評估模式
    epoch_loss = 0.0
    total_correctly_predicted_images = 0
    total_images_processed = 0

    with torch.no_grad():  # 在評估過程中不計算梯度
        for batch_idx, (images, targets_batch) in tqdm(enumerate(data_loader), desc="Validation", total=len(data_loader)):
            # 將圖像和目標轉移到指定設備 (GPU/CPU)
            images_gpu = [img.to(device) for img in images]
            targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]

            # --- 開始: 損失計算 (整合使用者提供的片段邏輯) ---
            # 前向傳播以獲取損失 (此時傳遞了 targets)
            loss_dict_output = model(images_gpu, targets_gpu)

            current_batch_loss_tensor = None

            if VERBOSE:
                print(f"\n[Validation Batch {batch_idx}] loss_dict_output type: {type(loss_dict_output)}")

            if isinstance(loss_dict_output, dict):
                if VERBOSE:
                    print("  loss_dict_output is a dictionary. Losses:")
                    for name, tensor in loss_dict_output.items():
                        # tensor.sum().item() 用於確保張量被加總（如果它不是純量）然後取其 Python 純量值
                        print(f"    Key: {name}, Shape: {tensor.shape}, Value: {tensor.sum().item():.4f}")
                # 加總所有損失。對每個損失張量調用 .sum() 以處理其可能不是純量的情況。
                current_batch_loss_tensor = sum(t.sum() for t in loss_dict_output.values())
            
            elif isinstance(loss_dict_output, list): # 處理使用者片段中提到的 list of dicts 的情況
                if VERBOSE:
                    print("  loss_dict_output is a list of dictionaries.")
                
                temp_losses = []
                for i, d_item in enumerate(loss_dict_output):
                    if isinstance(d_item, dict):
                        if VERBOSE:
                            print(f"  Item {i} in list (is a dict):")
                            for name, tensor in d_item.items():
                                print(f"    Key: {name}, Shape: {tensor.shape}, Value: {tensor.sum().item():.4f}")
                        temp_losses.extend([t.sum() for t in d_item.values()])
                    else:
                        # 如果列表中的元素不是字典，則引發錯誤或進行其他處理
                        # 為了與使用者片段的結構保持一致，這裡假設列表中的元素總是字典
                        # 如果 torch 模型回傳 list of tensors (而不是 list of dicts of tensors)
                        # 這部分邏輯需要調整
                        if VERBOSE:
                            print(f"  Item {i} in list is NOT a dict, type: {type(d_item)}. This item will be skipped for loss summation unless handled.")
                        # 根據實際情況，您可能需要引發錯誤或以不同方式處理此情況
                        # For now, we'll only sum if it's a list of dicts of tensors
                
                if temp_losses: # 確保 temp_losses 非空
                    current_batch_loss_tensor = sum(temp_losses)
                else: # 如果列表為空或不包含可加總的損失
                    current_batch_loss_tensor = torch.tensor(0.0).to(device) # 設為0損失
                    if VERBOSE:
                        print("  Warning: loss_dict_output was a list, but no summable losses found.")

            else:
                # 對於非 dict 或非 list 的未知型態
                raise TypeError(f"Unexpected loss_dict_output type: {type(loss_dict_output)} in validation. Expected dict or list of dicts.")

            if VERBOSE and current_batch_loss_tensor is not None:
                print(f"  Summed batch loss for this item: {current_batch_loss_tensor.item():.4f}\n")
            
            if current_batch_loss_tensor is not None:
                epoch_loss += current_batch_loss_tensor.item()
            # --- 結束: 損失計算 ---


            # --- 開始: 準確率計算 (保留先前版本的功能) ---
            # 為了計算準確率，我們需要模型的預測輸出 (不傳遞 targets)
            predictions = model(images_gpu) # 注意：這會再次執行前向傳播

            for i in range(len(images_gpu)): # 迭代處理批次中的每張圖片
                pred_boxes = predictions[i]['boxes'].cpu()
                pred_scores = predictions[i]['scores'].cpu()
                pred_labels = predictions[i]['labels'].cpu()
                
                # 從 targets_batch (CPU上原始的) 或 targets_gpu (GPU上的，需轉回CPU) 獲取真實框
                # targets_batch[i] 是原始的，在 collate_fn 之前，可能還未轉換為張量或已在 dataset 中轉換
                # 這裡我們使用 targets_batch，假設它包含CPU上的張量或可以轉換的數據
                gt_boxes = targets_batch[i]['boxes'].cpu() # 確保在CPU上
                gt_labels = targets_batch[i]['labels'].cpu() # 確保在CPU上

                image_has_correct_detection = False
                
                if len(gt_boxes) == 0: # 如果圖片中沒有真實物件
                    if len(pred_boxes) == 0 : # 模型也沒有預測任何物件
                         image_has_correct_detection = True
                    # 否則 (有預測但無真實物件)，所有預測都是誤報，image_has_correct_detection 保持 False
                
                elif len(pred_boxes) > 0 : # 如果有預測物件且有真實物件
                    # 用於標記真實框是否已被匹配，以避免一個真實框匹配多個預測框 (對於此簡化準確率而言非必需，但良好實踐)
                    # matched_gt = [False] * len(gt_boxes) # 如果需要更複雜的匹配邏輯

                    for pred_idx in range(len(pred_boxes)):
                        if pred_scores[pred_idx] < score_threshold:
                            continue # 跳過低於分數閾值的預測

                        # 檢查此預測是否與任何真實物件匹配
                        # (一個簡化的檢查：只要有一個預測物件正確，就認為圖片檢測正確)
                        # (更複雜的 mAP 計算會逐個匹配)
                        found_match_for_this_pred = False
                        for gt_idx in range(len(gt_boxes)):
                            if pred_labels[pred_idx] == gt_labels[gt_idx]: # 檢查類別是否匹配
                                # 計算 IoU
                                iou = box_iou(pred_boxes[pred_idx].unsqueeze(0), gt_boxes[gt_idx].unsqueeze(0))
                                if iou.item() > iou_threshold:
                                    image_has_correct_detection = True # 標記圖片有正確檢測
                                    found_match_for_this_pred = True
                                    break # 此預測框已找到匹配的真實框，跳出內層迴圈
                        
                        if image_has_correct_detection and found_match_for_this_pred: 
                             # 如果圖片已被標記為正確檢測 (因為此預測框找到了匹配)
                             # 就可以跳出外層預測框迴圈，處理下一張圖片
                             break 
                
                if image_has_correct_detection:
                    total_correctly_predicted_images += 1
            
            total_images_processed += len(images_gpu) # 更新已處理的圖片總數
            # --- 結束: 準確率計算 ---

    avg_epoch_loss = epoch_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    accuracy = total_correctly_predicted_images / total_images_processed if total_images_processed > 0 else 0.0
    
    # 這裡的 VERBOSE 控制整體驗證摘要的打印
    if VERBOSE:
        print(f"\n--- Validation Epoch Summary ---")
        print(f"Average Validation Loss: {avg_epoch_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f} ({total_correctly_predicted_images}/{total_images_processed} images correctly predicted)")
        print(f"--- End of Validation Epoch Summary ---\n")
    else: # 即使不是 VERBOSE，也打印關鍵指標
        print(f"Validation Results: Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
        
    return avg_epoch_loss, accuracy

## Declarate optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

## Train the model
if VERBOSE:
    print(f"Using device for training: {device}")

num_epochs = 10 # 您可以調整 epoch 數量
best_val_accuracy = -1.0 # 初始化最佳準確率
best_model_path = "best_model.pth"

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 20)

    train_loss = train_one_epoch(model, optimizer, train_loader, device)
    val_loss, val_accuracy = validate_one_epoch(model, val_loader, device)

    lr_scheduler.step() # Step the learning rate scheduler

    print(f"Epoch {epoch + 1} Summary:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")

    # Save model checkpoint for the current epoch
    epoch_save_path = f"faster_rcnn_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), epoch_save_path)
    if VERBOSE:
        print(f"Model saved for epoch {epoch + 1} to {epoch_save_path}")

    # Check if this is the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        if VERBOSE:
            print(f"*** New best model saved with accuracy: {best_val_accuracy:.4f} to {best_model_path} ***")
    
print("Training finished.")
if best_val_accuracy != -1.0:
    print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")
    print(f"Best model saved to: {best_model_path}")
else:
    print("No best model was saved (perhaps validation accuracy did not improve).")

