import os
import shutil
import random
from pathlib import Path
import argparse

def split_dataset(source_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Разделяет датасет из структуры:
    source/
        class1/
            img1.jpg
        class2/
            img2.jpg
    
    На train/val/test
    """
    random.seed(seed)
    
    source_path = Path(source_path)
    
    # Проверяем, что путь существует
    if not source_path.exists():
        print(f"❌ Путь {source_path} не существует")
        return
    
    classes = [d.name for d in source_path.iterdir() if d.is_dir()]
    
    if not classes:
        print(f"❌ В папке {source_path} нет подпапок с классами")
        return
    
    print(f"📁 Найдены классы: {classes}")
    
    # Создаем структуру для split
    for split, ratio in [('train', train_ratio), ('val', val_ratio), ('test', test_ratio)]:
        if ratio > 0:  # Создаем только если ratio > 0
            split_path = source_path.parent / split
            if split_path.exists():
                print(f"⚠️ {split_path} уже существует. Удаляю...")
                shutil.rmtree(split_path)
            split_path.mkdir(parents=True, exist_ok=True)
            
            for class_name in classes:
                (split_path / class_name).mkdir(exist_ok=True)
    
    total_images = 0
    
    for class_name in classes:
        class_path = source_path / class_name
        images = list(class_path.glob("*"))
        images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        # Перемешиваем
        random.shuffle(images)
        
        # Вычисляем границы
        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)
        
        # Распределяем
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # Копируем
        for img in train_images:
            shutil.copy(img, source_path.parent / 'train' / class_name / img.name)
        
        if val_ratio > 0:
            for img in val_images:
                shutil.copy(img, source_path.parent / 'val' / class_name / img.name)
        
        if test_ratio > 0:
            for img in test_images:
                shutil.copy(img, source_path.parent / 'test' / class_name / img.name)
        
        total_images += len(images)
        print(f"📦 {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    print(f"\n✅ Всего изображений: {total_images}")
    print(f"📁 Данные разделены!")
    print(f"   Train: {source_path.parent / 'train'}")
    print(f"   Val: {source_path.parent / 'val'}")
    print(f"   Test: {source_path.parent / 'test'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--source', type=str, required=True, help='Path to source dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Проверяем, что сумма ratios = 1
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"⚠️ Сумма ratios = {total_ratio}, должна быть 1.0")
        print("Нормализую...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    split_dataset(args.source, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
