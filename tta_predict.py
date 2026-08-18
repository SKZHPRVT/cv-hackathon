"""
Настраиваемое Test Time Augmentation (TTA).
Варианты: none, hflip, vflip, rotate, all
"""
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils.dataset import MEAN, STD

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model_name = config['model']['name']
    class_names = checkpoint['class_names']
    image_size = checkpoint.get('image_size', 224)
    
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, image_size

def get_tta_transforms(image_size, tta_type='hflip'):
    """Создание TTA трансформаций."""
    base_transforms = []
    
    if tta_type == 'none':
        base_transforms = [A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])]
    elif tta_type == 'hflip':
        base_transforms = [
            A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]),
        ]
    elif tta_type == 'vflip':
        base_transforms = [
            A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(image_size, image_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]),
        ]
    elif tta_type == 'all':
        base_transforms = [
            A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(image_size, image_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Resize(image_size, image_size),
                A.Rotate(limit=90, p=1.0),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]),
        ]
    
    return base_transforms

def predict_with_tta(model, image_path, class_names, device, image_size=224, tta_type='hflip'):
    """Предсказание с настраиваемым TTA."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Не удалось загрузить {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transforms = get_tta_transforms(image_size, tta_type)
    
    all_probs = []
    with torch.no_grad():
        for transform in transforms:
            augmented = transform(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs)
    
    avg_probs = torch.stack(all_probs).mean(dim=0)
    top_probs, top_indices = torch.topk(avg_probs, min(3, len(class_names)))
    
    print(f"📸 {image_path} (TTA: {tta_type}, {len(transforms)} вариантов)")
    for prob, idx in zip(top_probs[0], top_indices[0]):
        print(f"  {class_names[idx]}: {prob.item()*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TTA Prediction')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--tta', choices=['none', 'hflip', 'vflip', 'all'], default='hflip',
                       help='Тип TTA (default: hflip)')
    parser.add_argument('--image_size', type=int, default=None)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, class_names, checkpoint_image_size = load_model(args.checkpoint, device)
    
    image_size = args.image_size or checkpoint_image_size
    predict_with_tta(model, args.image, class_names, device, image_size, args.tta)
