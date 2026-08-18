"""
Генератор submission.csv для хакатона.
Форматы:
- Классификация: image,label или image,class1,class2,...
- Детекция: image,class,confidence,x1,y1,x2,y2
"""
import argparse
import pandas as pd
import numpy as np
import torch
import cv2
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from utils.dataset import MEAN, STD

def load_model(checkpoint_path, device):
    """Загрузка модели."""
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

def predict_image(model, image_path, class_names, image_size, device):
    """Предсказание для одного изображения."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    return probs[0].cpu().numpy()

def generate_classification_submission(checkpoint_path, test_folder, output_file='submission.csv', format_type='label'):
    """Генерация submission для классификации."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, class_names, image_size = load_model(checkpoint_path, device)
    
    test_folder = Path(test_folder)
    results = []
    
    # Рекурсивный поиск изображений
    image_paths = list(test_folder.rglob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    for img_path in image_paths:
            probs = predict_image(model, img_path, class_names, image_size, device)
            if probs is None:
                continue
            
            if format_type == 'label':
                predicted_idx = np.argmax(probs)
                results.append({
                    'image': img_path.name,
                    'label': class_names[predicted_idx]
                })
            elif format_type == 'probabilities':
                row = {'image': img_path.name}
                for i, cls in enumerate(class_names):
                    row[cls] = probs[i]
                results.append(row)
    
    if not results:
        print("⚠️ Не найдено изображений для предсказания")
        return None
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"✅ Submission сохранен: {output_file}")
    print(f"📊 Строк: {len(df)}")
    print(f"📋 Колонки: {list(df.columns)}")
    
    return df

def generate_yolo_submission(model_path, test_folder, output_file='submission.csv', conf=0.25):
    """Генерация submission для YOLO detection.
    Формат: image,class,confidence,x1,y1,x2,y2
    """
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    results_list = []
    
    for img_path in Path(test_folder).iterdir():
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        
        results = model.predict(str(img_path), conf=conf, verbose=False)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    
                    results_list.append({
                        'image': img_path.name,
                        'class': cls_name,
                        'confidence': conf_val,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    })
    
    df = pd.DataFrame(results_list)
    df.to_csv(output_file, index=False)
    print(f"✅ YOLO submission сохранен: {output_file}")
    print(f"📊 Строк: {len(df)}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Генерация submission')
    parser.add_argument('--checkpoint', required=True, help='Путь к чекпоинту (.pth или .pt)')
    parser.add_argument('--test_folder', required=True, help='Папка с тестовыми изображениями')
    parser.add_argument('--output', default='submission.csv', help='Выходной файл')
    parser.add_argument('--format', choices=['label', 'probabilities', 'yolo'], default='label',
                       help='label / probabilities / yolo (detection)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence для YOLO')
    
    args = parser.parse_args()
    
    if args.format == 'yolo':
        generate_yolo_submission(args.checkpoint, args.test_folder, args.output, args.conf)
    else:
        generate_classification_submission(args.checkpoint, args.test_folder, args.output, args.format)
