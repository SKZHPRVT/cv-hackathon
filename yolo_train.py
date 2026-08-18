"""
Скрипт для обучения YOLO на хакатоне.
Поддерживает детекцию, сегментацию, классификацию.
"""
from ultralytics import YOLO
import argparse
import os

def train_yolo(data_yaml, model_size='n', epochs=50, imgsz=640, task='detect'):
    """
    Обучение YOLO модели.
    
    Args:
        data_yaml: путь к YAML файлу с данными
        model_size: размер модели (n, s, m, l, x)
        epochs: количество эпох
        imgsz: размер изображения
        task: задача (detect, segment, classify, pose)
    """
    # Создаем имя модели
    model_name = f'yolov8{model_size}.pt'
    
    # Загружаем предобученную модель
    model = YOLO(model_name)
    
    # Обучаем
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        task=task,
        device='cuda' if os.system('nvidia-smi') == 0 else 'cpu',
        patience=10,
        save=True,
        plots=True,
    )
    
    return results

def predict_yolo(model_path, source, conf=0.25):
    """Инференс YOLO модели."""
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        show=False,
    )
    return results

def export_yolo(model_path, format='onnx'):
    """Экспорт модели в ONNX."""
    model = YOLO(model_path)
    model.export(format=format)
    print(f"✅ Модель экспортирована в {format}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO Training for Hackathon')
    parser.add_argument('--mode', choices=['train', 'predict', 'export'], required=True)
    parser.add_argument('--data', help='Path to data.yaml')
    parser.add_argument('--model', help='Path to model or size (n, s, m, l, x)')
    parser.add_argument('--source', help='Path to image/folder for prediction')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--task', default='detect', choices=['detect', 'segment', 'classify', 'pose'])
    parser.add_argument('--conf', type=float, default=0.25)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_yolo(args.data, args.model or 'n', args.epochs, args.imgsz, args.task)
    elif args.mode == 'predict':
        predict_yolo(args.model, args.source, args.conf)
    elif args.mode == 'export':
        export_yolo(args.model)
