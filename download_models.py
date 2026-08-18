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

def check_offline(strict=False):
    """Проверка offline-режима. Возвращает True если всё загружается из кеша.
    
    Args:
        strict: если True, запрещает сетевой доступ
    Returns:
        bool: True если все модели загрузились offline
    """
    print("\n🔍 Проверка offline-режима (pretrained weights)...")
    
    all_ok = True
    
    if strict:
        # Запрещаем сетевой доступ
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
        os.environ['TORCH_HOME'] = str(MODELS_DIR / 'torch_cache')
        print("  🔒 Сетевой доступ запрещен (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)")
    
    # Проверяем timm модели
    import timm
    for model_name in ['resnet18', 'efficientnet_b0']:
        try:
            model = timm.create_model(model_name, pretrained=True)
            print(f"  ✅ {model_name}: pretrained загружается")
        except Exception as e:
            print(f"  ❌ {model_name}: pretrained НЕ загружается ({e})")
            all_ok = False
    
    # Проверяем HF модели с local_files_only=True
    try:
        from transformers import AutoModelForImageClassification, CLIPModel, SamModel
        
        hf_models_to_check = [
            ('google/vit-base-patch16-224', AutoModelForImageClassification),
            ('openai/clip-vit-base-patch32', CLIPModel),
            ('facebook/sam-vit-base', SamModel),
        ]
        
        for model_name, model_class in hf_models_to_check:
            try:
                model = model_class.from_pretrained(model_name, local_files_only=True)
                print(f"  ✅ {model_name}: загружается offline")
            except Exception as e:
                print(f"  ❌ {model_name}: НЕ загружается offline ({type(e).__name__}: {str(e)[:80]})")
                all_ok = False
    except ImportError:
        print("  ⚠️ Transformers не установлен, пропускаю HF проверку")
    
    # Проверяем YOLO (в кеше, корне и MODELS_DIR)
    yolo_found = False
    yolo_locations = [
        Path.home() / '.cache' / 'ultralytics',
        Path('.'),
        MODELS_DIR
    ]
    
    for loc in yolo_locations:
        if loc.exists():
            yolo_models = list(loc.glob('*.pt'))
            if yolo_models:
                print(f"  ✅ YOLO модели найдены в {loc}: {len(yolo_models)} шт")
                yolo_found = True
                break
    
    if not yolo_found:
        print(f"  ⚠️ YOLO модели не найдены")
        if strict:
            all_ok = False
    
    if all_ok:
        print("\n✅ OFFLINE РЕЖИМ ПОЛНОСТЬЮ ГОТОВ")
    else:
        print("\n❌ ЕСТЬ ПРОБЛЕМЫ С OFFLINE-РЕЖИМОМ")
        print("Запустите: python download_models.py --all")
    
    return all_ok

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Скачивание моделей для offline-режима')
    parser.add_argument('--timm', action='store_true', help='Скачать timm модели')
    parser.add_argument('--yolo', action='store_true', help='Скачать YOLO модели')
    parser.add_argument('--hf', action='store_true', help='Скачать HF модели')
    parser.add_argument('--all', action='store_true', help='Скачать все модели')
    parser.add_argument('--check', action='store_true', help='Проверить offline-режим')
    parser.add_argument('--strict', action='store_true', help='Строгий offline-тест (запрет сети)')
    
    args = parser.parse_args()
    
    if args.check:
        result = check_offline(strict=args.strict)
        exit(0 if result else 1)
    
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
