"""
Предсказание с Test Time Augmentation (TTA).
Повышает точность за счет усреднения предсказаний на аугментированных версиях.
"""
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import argparse

def load_model(checkpoint_path, device):
    """Загрузка модели."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model_name = config['model']['name']
    class_names = checkpoint['class_names']
    
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names

def get_tta_transforms(image_size=224):
    """Создание TTA трансформаций."""
    return [
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(image_size, image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]

def predict_with_tta(model, image_path, class_names, device, image_size=224):
    """Предсказание с TTA."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transforms = get_tta_transforms(image_size)
    all_probs = []
    
    with torch.no_grad():
        for transform in transforms:
            augmented = transform(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs)
    
    # Усредняем предсказания
    avg_probs = torch.stack(all_probs).mean(dim=0)
    
    # Получаем результаты
    top_probs, top_indices = torch.topk(avg_probs, min(3, len(class_names)))
    
    print(f"📸 {image_path}")
    for prob, idx in zip(top_probs[0], top_indices[0]):
        print(f"  {class_names[idx]}: {prob.item()*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict with TTA')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--image_size', type=int, default=224)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, class_names = load_model(args.checkpoint, device)
    predict_with_tta(model, args.image, class_names, device, args.image_size)
