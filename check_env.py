#!/usr/bin/env python3
"""
Скрипт для быстрой проверки окружения перед хакатоном.
Проверяет все необходимые библиотеки, GPU, структуру проекта.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path

def print_status(message, status):
    """Выводит статус с цветом"""
    if status == "OK":
        print(f"✅ {message}")
    elif status == "WARN":
        print(f"⚠️  {message}")
    else:
        print(f"❌ {message}")

def check_python():
    """Проверка версии Python"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro}", "OK")
    else:
        print_status(f"Python {version.major}.{version.minor} (нужно 3.8+)", "ERROR")

def check_library(name, import_name=None):
    """Проверка библиотеки"""
    if import_name is None:
        import_name = name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print_status(f"{name} {version}", "OK")
        return True
    except ImportError:
        print_status(f"{name} не установлена", "ERROR")
        return False

def check_gpu():
    """Проверка GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print_status(f"GPU: {device_name} (x{device_count})", "OK")
            
            # Проверка памяти
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_status(f"GPU Memory: {memory:.1f} GB", "OK")
            return True
        else:
            print_status("GPU не найдена, будет использоваться CPU", "WARN")
            return False
    except ImportError:
        print_status("PyTorch не установлен", "ERROR")
        return False

def check_structure():
    """Проверка структуры проекта"""
    required_dirs = ['configs', 'utils', 'checkpoints', 'data']
    required_files = ['train.py', 'predict.py', 'app.py', 'requirements.txt']
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print_status(f"Папка {dir_name}/", "OK")
        else:
            print_status(f"Папка {dir_name}/ отсутствует", "WARN")
    
    for file_name in required_files:
        if Path(file_name).exists():
            print_status(f"Файл {file_name}", "OK")
        else:
            print_status(f"Файл {file_name} отсутствует", "ERROR")

def check_data():
    """Проверка данных"""
    data_path = Path('data')
    if not data_path.exists():
        print_status("Папка data/ не создана", "WARN")
        return
    
    train_path = data_path / 'train'
    val_path = data_path / 'val'
    
    if train_path.exists():
        classes = [d for d in train_path.iterdir() if d.is_dir()]
        if classes:
            total_images = sum(1 for c in classes for f in (train_path/c).glob('*') 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
            print_status(f"Train: {len(classes)} классов, {total_images} изображений", "OK")
        else:
            print_status("Train: нет классов", "WARN")
    else:
        print_status("Train: папка отсутствует", "WARN")
    
    if val_path.exists():
        classes = [d for d in val_path.iterdir() if d.is_dir()]
        if classes:
            total_images = sum(1 for c in classes for f in (val_path/c).glob('*') 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
            print_status(f"Val: {len(classes)} классов, {total_images} изображений", "OK")
        else:
            print_status("Val: нет классов", "WARN")
    else:
        print_status("Val: папка отсутствует", "WARN")

def check_checkpoints():
    """Проверка сохраненных моделей"""
    checkpoint_path = Path('checkpoints')
    if checkpoint_path.exists():
        models = list(checkpoint_path.glob('*.pth'))
        if models:
            for model in models:
                size_mb = model.stat().st_size / 1e6
                print_status(f"Модель {model.name} ({size_mb:.1f} MB)", "OK")
        else:
            print_status("Нет сохраненных моделей", "WARN")
    else:
        print_status("Папка checkpoints/ отсутствует", "WARN")

def check_config():
    """Проверка конфига"""
    config_path = Path('configs/config.yaml')
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Проверка ключевых параметров
            if 'model' in config and 'name' in config['model']:
                print_status(f"Модель: {config['model']['name']}", "OK")
            if 'training' in config:
                training = config['training']
                if 'epochs' in training:
                    print_status(f"Эпохи: {training['epochs']}", "OK")
                if 'batch_size' in training:
                    print_status(f"Batch size: {training['batch_size']}", "OK")
        except Exception as e:
            print_status(f"Ошибка чтения конфига: {e}", "ERROR")
    else:
        print_status("config.yaml отсутствует", "ERROR")

def check_disk_space():
    """Проверка свободного места"""
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / 1e9
    if free_gb > 10:
        print_status(f"Свободное место: {free_gb:.1f} GB", "OK")
    elif free_gb > 5:
        print_status(f"Свободное место: {free_gb:.1f} GB (маловато)", "WARN")
    else:
        print_status(f"Свободное место: {free_gb:.1f} GB (критически мало!)", "ERROR")

def main():
    print("="*70)
    print("🔍 ПРОВЕРКА ОКРУЖЕНИЯ CV HACKATHON TOOLKIT")
    print("="*70)
    
    print("\n📌 Python:")
    check_python()
    
    print("\n📌 Библиотеки:")
    libraries = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('timm', 'timm'),
        ('albumentations', 'albumentations'),
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pyyaml', 'yaml'),
        ('tqdm', 'tqdm'),
        ('gradio', 'gradio'),
        ('plotly', 'plotly'),
    ]
    
    all_ok = True
    for lib_name, import_name in libraries:
        if not check_library(lib_name, import_name):
            all_ok = False
    
    print("\n📌 GPU:")
    if not check_gpu():
        all_ok = False
    
    print("\n📌 Структура проекта:")
    check_structure()
    
    print("\n📌 Конфигурация:")
    check_config()
    
    print("\n📌 Данные:")
    check_data()
    
    print("\n📌 Модели:")
    check_checkpoints()
    
    print("\n📌 Диск:")
    check_disk_space()
    
    print("\n" + "="*70)
    if all_ok:
        print("🎉 Всё готово к хакатону!")
    else:
        print("⚠️  Есть проблемы. Установите недостающие компоненты:")
        print("   pip install -r requirements.txt")
    print("="*70)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
