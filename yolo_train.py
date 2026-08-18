"""
YOLO обучение/инференс/экспорт.
CUDA определяется через PyTorch, модель — произвольный путь или размер.
"""
from ultralytics import YOLO
import argparse
import torch

def get_device():
    """Определение устройства через PyTorch."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def train_yolo(data_yaml, model='yolov8n.pt', epochs=50, imgsz=640, task='detect'):
    """Обучение YOLO.
    
    Args:
        data_yaml: путь к YAML с данными
        model: путь к .pt файлу ИЛИ размер (n/s/m/l/x)
        epochs: количество эпох
        imgsz: размер изображения
        task: detect/segment/classify/pose
    """
    # Если передали размер (n/s/m/l/x), создаем имя модели
    if model in ['n', 's', 'm', 'l', 'x']:
        model_name = f'yolov8{model}.pt'
    else:
        model_name = model  # произвольный путь к .pt файлу
    
    print(f"📦 Модель: {model_name}")
    print(f"🔧 Device: {get_device()}")
    
    yolo_model = YOLO(model_name)
    
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        task=task,
        device=get_device(),
        patience=10,
        save=True,
        plots=True,
    )
    
    return results

def predict_yolo(model_path, source, conf=0.25):
    """Инференс YOLO."""
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        device=get_device(),
    )
    return results

def export_yolo(model_path, format='onnx'):
    """Экспорт YOLO."""
    model = YOLO(model_path)
    model.export(format=format)
    print(f"✅ Экспортировано в {format}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO для хакатона')
    parser.add_argument('--mode', choices=['train', 'predict', 'export'], required=True)
    parser.add_argument('--data', help='Путь к data.yaml')
    parser.add_argument('--model', default='yolov8n.pt', 
                       help='Размер (n/s/m/l/x) или путь к .pt файлу')
    parser.add_argument('--source', help='Изображение или папка')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--task', default='detect', choices=['detect', 'segment', 'classify', 'pose'])
    parser.add_argument('--conf', type=float, default=0.25)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_yolo(args.data, args.model, args.epochs, args.imgsz, args.task)
    elif args.mode == 'predict':
        predict_yolo(args.model, args.source, args.conf)
    elif args.mode == 'export':
        export_yolo(args.model)
