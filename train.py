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
    model.eval()
    epoch_loss = 0.0
    total_correctly_predicted_images = 0
    total_images_processed = 0

    with torch.no_grad():
        for batch_idx, (images, targets_batch) in tqdm(enumerate(data_loader), desc="Validation", total=len(data_loader)):
            images_gpu = [img.to(device) for img in images]
            # For loss calculation, targets need to be on device
            targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]

            # Get losses
            loss_dict = model(images_gpu, targets_gpu)
            
            if VERBOSE and batch_idx == 0: # Print loss structure for the first batch if VERBOSE
                print(f"[Validation Batch {batch_idx}] loss_dict type: {type(loss_dict)}")
                if isinstance(loss_dict, dict):
                    for name, tensor in loss_dict.items():
                        print(f"  {name}: {tensor.shape if hasattr(tensor, 'shape') else type(tensor)}")
            
            current_batch_loss = sum(loss for loss in loss_dict.values())
            epoch_loss += current_batch_loss.item()

            # Get predictions for accuracy calculation
            # model(images_gpu) returns predictions when targets are not provided or in eval mode without targets
            # However, to keep it simple, we use the same call and then extract predictions if needed.
            # For FasterRCNN, model(images) in eval mode returns list of dicts (boxes, labels, scores)
            predictions = model(images_gpu) # Get predictions

            for i in range(len(images)):
                pred_boxes = predictions[i]['boxes'].cpu()
                pred_scores = predictions[i]['scores'].cpu()
                pred_labels = predictions[i]['labels'].cpu()
                
                gt_boxes = targets_batch[i]['boxes'].cpu()
                gt_labels = targets_batch[i]['labels'].cpu()

                image_has_correct_detection = False
                
                if len(gt_boxes) == 0: # No ground truth objects
                    if len(pred_boxes) == 0 : # No predictions either
                         image_has_correct_detection = True
                    # else: some predictions, all are false positives, so image_has_correct_detection remains False
                
                elif len(pred_boxes) > 0 : # There are predictions and ground truth objects
                    for pred_idx in range(len(pred_boxes)):
                        if pred_scores[pred_idx] < score_threshold:
                            continue

                        found_match_for_pred = False
                        for gt_idx in range(len(gt_boxes)):
                            if pred_labels[pred_idx] == gt_labels[gt_idx]:
                                iou = box_iou(pred_boxes[pred_idx].unsqueeze(0), gt_boxes[gt_idx].unsqueeze(0))
                                if iou.item() > iou_threshold:
                                    image_has_correct_detection = True
                                    found_match_for_pred = True
                                    break # Matched this gt box, move to next pred box or finish image
                        if image_has_correct_detection and found_match_for_pred: # Optimization: if one good pred found, image is correct
                             break 
                
                if image_has_correct_detection:
                    total_correctly_predicted_images += 1
            
            total_images_processed += len(images)


    avg_loss = epoch_loss / len(data_loader)
    accuracy = total_correctly_predicted_images / total_images_processed if total_images_processed > 0 else 0.0
    
    if VERBOSE:
        print(f"Average validation loss: {avg_loss:.6f}")
        print(f"Validation accuracy: {accuracy:.4f} ({total_correctly_predicted_images}/{total_images_processed})")
        
    return avg_loss, accuracy

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

