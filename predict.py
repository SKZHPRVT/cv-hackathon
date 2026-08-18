import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import argparse
import os
from pathlib import Path

def load_model(checkpoint_path, device):
    """Загружаем модель из чекпоинта"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Получаем имя модели и классы из конфига
    config = checkpoint['config']
    model_name = config['model']['name']
    class_names = checkpoint['class_names']
    
    # Создаем модель
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names

def preprocess_image(image_path, image_size=224):
    """Предобработка изображения"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    augmented = transform(image=image)
    return augmented['image'].unsqueeze(0)

def predict_image(model, image_tensor, class_names, device):
    """Предсказание для одного изображения"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    # Получаем топ-3 предсказания
    top_probs, top_indices = torch.topk(probabilities, min(3, len(class_names)))
    
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append({
            'class': class_names[idx],
            'probability': prob.item() * 100
        })
    
    return results

def predict_folder(model, folder_path, class_names, device, image_size=224):
    """Предсказание для всей папки"""
    results = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for img_path in Path(folder_path).iterdir():
        if img_path.suffix.lower() in image_extensions:
            try:
                image_tensor = preprocess_image(img_path, image_size)
                predictions = predict_image(model, image_tensor, class_names, device)
                results[img_path.name] = predictions
            except Exception as e:
                print(f"⚠️ Ошибка при обработке {img_path.name}: {e}")
                results[img_path.name] = [{'class': 'ERROR', 'probability': 0}]
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Predict with trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Загружаем модель
    model, class_names = load_model(args.checkpoint, device)
    print(f"✅ Model loaded. Classes: {class_names}")
    
    if args.image:
        # Предсказание для одного изображения
        image_tensor = preprocess_image(args.image, args.image_size)
        results = predict_image(model, image_tensor, class_names, device)
        
        print(f"\n📸 Image: {args.image}")
        for i, pred in enumerate(results, 1):
            print(f"  {i}. {pred['class']}: {pred['probability']:.2f}%")
    
    elif args.folder:
        # Предсказание для папки
        results = predict_folder(model, args.folder, class_names, device, args.image_size)
        
        print(f"\n📁 Folder: {args.folder}")
        for img_name, predictions in results.items():
            print(f"\n📸 {img_name}:")
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. {pred['class']}: {pred['probability']:.2f}%")
    
    else:
        print("❌ Please provide either --image or --folder argument")

if __name__ == "__main__":
    main()
