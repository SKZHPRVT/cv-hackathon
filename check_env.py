#!/usr/bin/env python3
"""
Проверка окружения со smoke test.
Возвращает exit code 1 при провале любого теста.
"""
import sys
import os
import importlib
import subprocess
import tempfile
from pathlib import Path
import numpy as np

FAILED = False

def print_status(message, status):
    global FAILED
    if status == "OK":
        print(f"✅ {message}")
    elif status == "WARN":
        print(f"⚠️  {message}")
    else:
        print(f"❌ {message}")
        FAILED = True

def check_python():
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro}", "OK")
    else:
        print_status(f"Python {version.major}.{version.minor} (нужно 3.8+)", "ERROR")

def check_library(name, import_name=None):
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
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_status(f"GPU: {device_name} ({memory:.1f} GB)", "OK")
            return True
        else:
            print_status("GPU не найдена, используется CPU", "WARN")
            return False
    except ImportError:
        print_status("PyTorch не установлен", "ERROR")
        return False

def smoke_test_pytorch():
    """Smoke test: CUDA forward/backward, optimizer step, save/load."""
    print("\n🔬 Smoke test PyTorch...")
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Forward pass
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)).to(device)
        x = torch.randn(4, 10, device=device)
        y = model(x)
        assert y.shape == (4, 2), f"Shape mismatch: {y.shape}"
        print_status(f"Forward pass ({device})", "OK")
        
        # Backward pass
        loss = nn.CrossEntropyLoss()(y, torch.tensor([0, 1, 0, 1], device=device))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_status(f"Backward + optimizer step ({device})", "OK")
        
        # Save/load с tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(model.state_dict(), tmp_path)
        model2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)).to(device)
        model2.load_state_dict(torch.load(tmp_path))
        os.unlink(tmp_path)
        print_status("Save/load checkpoint (tempfile)", "OK")
        
        return True
    except Exception as e:
        print_status(f"Smoke test PyTorch: {e}", "ERROR")
        return False

def smoke_test_dataloader():
    print("\n🔬 Smoke test DataLoader...")
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        
        class DummyDataset(Dataset):
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return torch.randn(3, 32, 32), idx % 2
        
        loader = DataLoader(DummyDataset(), batch_size=4, shuffle=True)
        batch = next(iter(loader))
        assert len(batch) == 2
        assert batch[0].shape == (4, 3, 32, 32)
        print_status("DataLoader", "OK")
        return True
    except Exception as e:
        print_status(f"Smoke test DataLoader: {e}", "ERROR")
        return False

def smoke_test_timm():
    print("\n🔬 Smoke test timm...")
    try:
        import timm
        import torch
        model = timm.create_model('resnet18', pretrained=False, num_classes=2)
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        assert y.shape == (1, 2)
        print_status("timm ResNet18 forward", "OK")
        return True
    except Exception as e:
        print_status(f"Smoke test timm: {e}", "ERROR")
        return False

def smoke_test_onnx():
    print("\n🔬 Smoke test ONNX...")
    try:
        import onnx
        import onnxruntime as ort
        print_status("ONNX + ONNX Runtime", "OK")
        return True
    except ImportError as e:
        print_status(f"Smoke test ONNX: {e}", "ERROR")
        return False

def smoke_test_yolo():
    print("\n🔬 Smoke test YOLO...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print_status("YOLO загружен", "OK")
        return True
    except Exception as e:
        print_status(f"Smoke test YOLO: {e}", "ERROR")
        return False

def check_structure():
    print("\n📁 Структура проекта...")
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Проверка окружения')
    parser.add_argument('--no-yolo', action='store_true', help='Пропустить YOLO тест')
    parser.add_argument('--with-yolo', action='store_true', help='Запустить YOLO тест')
    args = parser.parse_args()
    
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
        ('ultralytics', 'ultralytics'),
        ('onnx', 'onnx'),
        ('onnxruntime', 'onnxruntime'),
        ('transformers', 'transformers'),
        ('diffusers', 'diffusers'),
        ('imagehash', 'imagehash'),
    ]
    
    for lib_name, import_name in libraries:
        check_library(lib_name, import_name)
    
    print("\n📌 GPU:")
    check_gpu()
    
    check_structure()
    
    # Smoke tests
    smoke_test_pytorch()
    smoke_test_dataloader()
    smoke_test_timm()
    smoke_test_onnx()
    
    # YOLO опционально (по умолчанию пропускаем)
    if args.with_yolo:
        smoke_test_yolo()
    elif not args.no_yolo:
        print("\n⚠️ YOLO тест пропущен (используйте --with-yolo для запуска)")
    
    print("\n" + "="*70)
    if FAILED:
        print("❌ ЕСТЬ ОШИБКИ! Исправьте и запустите снова.")
        print("="*70)
        sys.exit(1)
    else:
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print("="*70)
        sys.exit(0)

if __name__ == "__main__":
    main()
