"""
Инференс классификационной модели.
Берет image_size, mean, std, class_names из чекпоинта.
"""
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import argparse
from pathlib import Path

def load_model(checkpoint_path, device):
    """Загрузка модели с полной информацией из чекпоинта."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_name = checkpoint['config']['model']['name']
    class_names = checkpoint['class_names']
    image_size = checkpoint.get('image_size', checkpoint['config']['training']['image_size'])
    mean = checkpoint.get('mean', [0.485, 0.456, 0.406])
    std = checkpoint.get('std', [0.229, 0.224, 0.225])
    
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, image_size, mean, std

def preprocess_image(image_path, image_size, mean, std):
    """Предобработка с параметрами из чекпоинта."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    augmented = transform(image=image)
    return augmented['image'].unsqueeze(0)

def predict_image(model, image_tensor, class_names, device):
    """Предсказание."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    top_probs, top_indices = torch.topk(probabilities, min(3, len(class_names)))
    
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append({
            'class': class_names[idx],
            'probability': prob.item() * 100
        })
    
    return results

def predict_folder(model, folder_path, class_names, device, image_size, mean, std):
    """Предсказание для папки."""
    results = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for img_path in Path(folder_path).iterdir():
        if img_path.suffix.lower() in image_extensions:
            try:
                image_tensor = preprocess_image(img_path, image_size, mean, std)
                if image_tensor is not None:
                    predictions = predict_image(model, image_tensor, class_names, device)
                    results[img_path.name] = predictions
            except Exception as e:
                results[img_path.name] = [{'class': 'ERROR', 'probability': 0}]
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Predict with trained model')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', help='Путь к изображению')
    parser.add_argument('--folder', help='Путь к папке')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Всё из чекпоинта
    model, class_names, image_size, mean, std = load_model(args.checkpoint, device)
    print(f"✅ Модель загружена")
    print(f"📋 Классы: {class_names}")
    print(f"📐 Image size: {image_size}")
    print(f"📊 Mean: {mean}")
    print(f"📊 Std: {std}")
    
    if args.image:
        image_tensor = preprocess_image(args.image, image_size, mean, std)
        if image_tensor is None:
            print(f"❌ Не удалось загрузить {args.image}")
            return
        
        results = predict_image(model, image_tensor, class_names, device)
        
        print(f"\n📸 {args.image}")
        for i, pred in enumerate(results, 1):
            print(f"  {i}. {pred['class']}: {pred['probability']:.2f}%")
    
    elif args.folder:
        results = predict_folder(model, args.folder, class_names, device, image_size, mean, std)
        
        print(f"\n📁 {args.folder}")
        for img_name, predictions in results.items():
            print(f"\n📸 {img_name}:")
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. {pred['class']}: {pred['probability']:.2f}%")
    
    else:
        print("❌ Укажите --image или --folder")

if __name__ == "__main__":
    main()
