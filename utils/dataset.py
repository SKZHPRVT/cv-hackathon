"""
Датасеты для классификации с поддержкой:
- Структуры папок (class/folder)
- CSV-разметки
- Стратифицированного разделения
- Конфигурируемых аугментаций
- Единого препроцессинга
"""
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# Константы препроцессинга (ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 224

def get_transforms(image_size=224, is_train=True, augmentation_config=None):
    """Создает трансформации с конфигурируемыми аугментациями."""
    if is_train:
        transforms = [
            A.Resize(image_size, image_size),
        ]
        
        # Добавляем аугментации из конфига
        if augmentation_config:
            if augmentation_config.get('horizontal_flip', False):
                transforms.append(A.HorizontalFlip(p=augmentation_config.get('horizontal_flip_prob', 0.5)))
            if augmentation_config.get('vertical_flip', False):
                transforms.append(A.VerticalFlip(p=augmentation_config.get('vertical_flip_prob', 0.5)))
            if augmentation_config.get('random_rotate', False):
                transforms.append(A.Rotate(limit=augmentation_config.get('rotate_limit', 15), p=0.5))
            if augmentation_config.get('random_crop', False):
                # Сначала увеличиваем, потом crop до нужного размера
                crop_scale = augmentation_config.get('crop_scale', 1.2)
                larger_size = int(image_size * crop_scale)
                transforms.append(A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), p=0.5))
            if augmentation_config.get('brightness_contrast', False):
                transforms.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
            if augmentation_config.get('hue_saturation', False):
                transforms.append(A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3))
            if augmentation_config.get('coarse_dropout', False):
                transforms.append(A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25), hole_width_range=(0.1, 0.25), p=0.3))
        
        # Безопасные по умолчанию аугментации (не меняют семантику)
        transforms.extend([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    else:
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    
    return A.Compose(transforms)


class ImageDataset(Dataset):
    """Датасет для классификации из структуры папок."""
    
    def __init__(self, root_path, transform=None, class_to_idx=None):
        self.root_path = Path(root_path)
        self.transform = transform
        
        if class_to_idx is None:
            self.classes = sorted([d.name for d in self.root_path.iterdir() if d.is_dir()])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_path = self.root_path / class_name
            if not class_path.exists():
                continue
            for img_path in class_path.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = cv2.imread(img_path)
        if image is None:
            # Заглушка для битых файлов
            image = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


class CSVImageDataset(Dataset):
    """Датасет из CSV-разметки (path, label)."""
    
    def __init__(self, csv_path, transform=None, class_to_idx=None, image_column='path', label_column='label'):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column
        
        if class_to_idx is None:
            self.classes = sorted(self.df[label_column].unique().tolist())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())
        
        self.images = self.df[image_column].tolist()
        self.labels = [self.class_to_idx[label] for label in self.df[label_column].tolist()]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


def create_dataloaders(train_path, val_path, batch_size=32, num_workers=4, 
                       image_size=224, augmentation_config=None, train_csv=None, val_csv=None):
    """Создает DataLoader'ы с общим class_to_idx."""
    
    transform_train = get_transforms(image_size, is_train=True, augmentation_config=augmentation_config)
    transform_val = get_transforms(image_size, is_train=False)
    
    if train_csv:
        train_dataset = CSVImageDataset(train_csv, transform=transform_train)
    else:
        train_dataset = ImageDataset(train_path, transform=transform_train)
    
    # Общий class_to_idx из train
    class_to_idx = train_dataset.class_to_idx
    classes = train_dataset.classes
    
    if val_csv:
        val_dataset = CSVImageDataset(val_csv, transform=transform_val, class_to_idx=class_to_idx)
    else:
        val_dataset = ImageDataset(val_path, transform=transform_val, class_to_idx=class_to_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, classes, class_to_idx


def stratified_split(source_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                     seed=42, group_column=None, csv_path=None):
    """Стратифицированное разделение данных."""
    if csv_path:
        df = pd.read_csv(csv_path)
        labels = df['label'] if 'label' in df.columns else None
        groups = df[group_column] if group_column and group_column in df.columns else None
        
        train_df, temp_df = train_test_split(
            df, test_size=val_ratio + test_ratio, 
            stratify=labels, random_state=seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_df['label'] if labels is not None else None,
            random_state=seed
        )
        
        return train_df, val_df, test_df
    else:
        # Для структуры папок
        images = []
        labels = []
        for class_name in sorted([d.name for d in Path(source_path).iterdir() if d.is_dir()]):
            class_path = Path(source_path) / class_name
            for img in class_path.glob("*"):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    images.append(str(img))
                    labels.append(class_name)
        
        train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
            images, labels, test_size=val_ratio + test_ratio,
            stratify=labels, random_state=seed
        )
        val_imgs, test_imgs, val_labels, test_labels = train_test_split(
            temp_imgs, temp_labels, test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_labels, random_state=seed
        )
        
        return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels)
