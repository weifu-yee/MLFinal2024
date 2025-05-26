#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Declarate Dataset
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
import os
from PIL import Image
from tqdm import tqdm

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


# In[ ]:


## Declarate DataLoader
from torch.utils.data import DataLoader

# Define paths
train_annotation_file = 'archive/train_annotations1.json'
val_annotation_file = 'archive/valid_annotations1.json'
train_image_dir = 'archive/dataset/dataset/train/images'
val_image_dir = 'archive/dataset/dataset/valid/images'

# Define transformations (Optional, can be expanded as needed)
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


# In[3]:


## Loading the pre-trained Faster R-CNN model
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# Load a pre-trained Faster R-CNN model
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
num_classes = 4  # 3 classes (pothole, cracks, open_manhole) + 1 background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")


# In[4]:


## Validata data
import os
import json
from PIL import Image

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
        print(f"Missing images: {missing_images}")
        return False

    # Check if all bounding boxes are valid
    invalid_bboxes = []
    for annotation in annotations['annotations']:
        x, y, w, h = annotation['bbox']
        if w <= 0 or h <= 0:
            invalid_bboxes.append(annotation['id'])

    if invalid_bboxes:
        print(f"Invalid bounding boxes for annotations: {invalid_bboxes}")
        return False

    print("Data validation successful!")
    return True

# Define paths for your data
train_annotation_file = 'archive/train_annotations1.json'
val_annotation_file = 'archive/valid_annotations1.json'
train_image_dir = 'archive/dataset/dataset/train/images'
val_image_dir = 'archive/dataset/dataset/valid/images'

# Validate train and validation data
if not validate_data(train_image_dir, train_annotation_file):
    print("Training data validation failed. Please check your images and annotations.")
elif not validate_data(val_image_dir, val_annotation_file):
    print("Validation data validation failed. Please check your images and annotations.")
else:
    print("Starting training...")
    # Proceed with training after successful data validation
    model.train()  # Make sure your model is in training mode


# In[ ]:


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

        # Sum all losses in the loss_dict
        if isinstance(loss_dict, list):
            losses = sum(loss for loss in loss_dict)  # Don't use .item(), keep it as a tensor
        else:
            losses = sum(loss for loss in loss_dict.values())  # Same here

        losses.backward()  # Now losses is a tensor, so backward() can be called
        optimizer.step()  # Update the weights
        epoch_loss += losses.item()  # Add the loss to the epoch's loss (convert to float here for reporting)

    avg_epoch_loss = epoch_loss / len(data_loader)
    return avg_epoch_loss

def validate_one_epoch(model, data_loader, device):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            # 移到 GPU
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward
            loss_dict = model(images, targets)

            # 列印 loss_dict 型態
            print(f"[Batch {batch_idx}] loss_dict type: {type(loss_dict)}")

            # 如果是 dict，就列出 key 和 shape；如果是 list，就對每個 dict 做同樣的事
            if isinstance(loss_dict, dict):
                for name, tensor in loss_dict.items():
                    print(f"  {name}: {tensor.shape}")
                # sum 所有 losses
                losses = sum(t.sum() for t in loss_dict.values())

            elif isinstance(loss_dict, list):
                for idx, d in enumerate(loss_dict):
                    print(f"  loss_dict[{idx}] keys: {list(d.keys())}")
                    for name, tensor in d.items():
                        print(f"    {name}: {tensor.shape}")
                losses = sum(t.sum() for d in loss_dict for t in d.values())

            else:
                raise TypeError(f"Unexpected loss_dict type: {type(loss_dict)}")

            # 確認最終 scalar loss
            print(f"  summed loss: {losses.item()}\n")

            epoch_loss += losses.item()

    avg_loss = epoch_loss / len(data_loader)
    print(f"Average validation loss: {avg_loss:.6f}")
    return avg_loss


# In[ ]:


## Declarate optimizer and learning rate scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
optimizer.zero_grad()


# In[ ]:


## Train the model
# check if using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    train_loss = train_one_epoch(model, optimizer, train_loader, device)
    val_loss = validate_one_epoch(model, val_loader, device)

    # Step the parameters
    optimizer.step()
    # Zero the gradients
    optimizer.zero_grad()
    # Step the learning rate scheduler
    lr_scheduler.step()

    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), f"faster_rcnn_epoch_{epoch + 1}.pth")
    print(f"Model saved for epoch {epoch + 1}")

