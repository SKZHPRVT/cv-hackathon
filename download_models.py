"""
Скачивание моделей заранее для offline-режима.
Кеширует:
- timm модели (ResNet, EfficientNet)
- YOLO модели
- HF модели (опционально)
"""
import argparse
import os
import sys
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def download_timm_models():
    """Скачивание timm моделей."""
    print("\n📦 Скачивание timm моделей...")
    import timm
    
    models = [
        'resnet18',
        'resnet34', 
        'resnet50',
        'efficientnet_b0',
        'efficientnet_b3',
        'mobilenetv3_large_100',
    ]
    
    for model_name in models:
        try:
            print(f"  📥 {model_name}...")
            model = timm.create_model(model_name, pretrained=True)
            print(f"  ✅ {model_name}")
        except Exception as e:
            print(f"  ⚠️ {model_name}: {e}")

def download_yolo_models():
    """Скачивание YOLO моделей."""
    print("\n📦 Скачивание YOLO моделей...")
    try:
        from ultralytics import YOLO
        
        models = ['yolov8n.pt', 'yolov8s.pt']
        
        for model_name in models:
            try:
                print(f"  📥 {model_name}...")
                model = YOLO(model_name)
                # Копируем в models/
                import shutil
                src = Path.home() / '.cache' / 'ultralytics' / model_name
                if src.exists():
                    shutil.copy(src, MODELS_DIR / model_name)
                print(f"  ✅ {model_name}")
            except Exception as e:
                print(f"  ⚠️ {model_name}: {e}")
    except ImportError:
        print("  ⚠️ Ultralytics не установлен")

def download_hf_models():
    """Скачивание HF моделей."""
    print("\n📦 Скачивание HF моделей...")
    
    models = [
        ('google/vit-base-patch16-224', 'classification'),
        ('openai/clip-vit-base-patch32', 'clip'),
        ('facebook/sam-vit-base', 'sam'),
    ]
    
    for model_name, model_type in models:
        try:
            print(f"  📥 {model_name}...")
            if model_type == 'classification':
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                AutoImageProcessor.from_pretrained(model_name)
                AutoModelForImageClassification.from_pretrained(model_name)
            elif model_type == 'clip':
                from transformers import CLIPProcessor, CLIPModel
                CLIPProcessor.from_pretrained(model_name)
                CLIPModel.from_pretrained(model_name)
            elif model_type == 'sam':
                from transformers import SamModel, SamProcessor
                SamProcessor.from_pretrained(model_name)
                SamModel.from_pretrained(model_name)
            print(f"  ✅ {model_name}")
        except Exception as e:
            print(f"  ⚠️ {model_name}: {e}")

def check_offline():
    """Проверка offline-режима."""
    print("\n🔍 Проверка offline-режима...")
    
    # Проверяем кеш timm
    import timm
    for model_name in ['resnet18', 'efficientnet_b0']:
        try:
            # Пробуем загрузить без pretrained (должно работать offline)
            model = timm.create_model(model_name, pretrained=False)
            print(f"  ✅ {model_name}: создается без интернета")
        except:
            print(f"  ⚠️ {model_name}: не создается")
    
    # Проверяем HF кеш
    hf_cache = Path.home() / '.cache' / 'huggingface'
    if hf_cache.exists():
        cache_size = sum(f.stat().st_size for f in hf_cache.rglob('*') if f.is_file())
        print(f"  ✅ HF кеш: {cache_size / 1e9:.1f} GB")
    else:
        print(f"  ⚠️ HF кеш не найден")
    
    # Проверяем YOLO
    yolo_cache = Path.home() / '.cache' / 'ultralytics'
    if yolo_cache.exists():
        yolo_models = list(yolo_cache.glob('*.pt'))
        print(f"  ✅ YOLO кеш: {len(yolo_models)} моделей")
    else:
        print(f"  ⚠️ YOLO кеш не найден")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Скачивание моделей для offline-режима')
    parser.add_argument('--timm', action='store_true', help='Скачать timm модели')
    parser.add_argument('--yolo', action='store_true', help='Скачать YOLO модели')
    parser.add_argument('--hf', action='store_true', help='Скачать HF модели')
    parser.add_argument('--all', action='store_true', help='Скачать все модели')
    parser.add_argument('--check', action='store_true', help='Проверить offline-режим')
    
    args = parser.parse_args()
    
    if args.check:
        check_offline()
        exit(0)
    
    if args.timm or args.all:
        download_timm_models()
    
    if args.yolo or args.all:
        download_yolo_models()
    
    if args.hf or args.all:
        download_hf_models()
    
    print("\n✅ Готово!")
    print(f"📁 Модели сохранены в {MODELS_DIR}/")
    print("\n💡 Для offline-режима:")
    print("  export HF_HOME=./models/hf_cache")
    print("  export TORCH_HOME=./models/torch_cache")
