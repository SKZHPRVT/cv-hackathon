"""
Стратифицированное разделение данных с защитой от пустых классов.
"""
import os
import shutil
import random
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from collections import Counter

def split_dataset(source_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, 
                 stratified=True, min_val_per_class=3, group_column=None, csv_path=None):
    """
    Стратифицированное разделение данных.
    
    Args:
        source_path: исходная папка
        train_ratio, val_ratio, test_ratio: пропорции
        seed: случайное зерно
        stratified: использовать стратификацию
        min_val_per_class: минимальное число изображений в val для каждого класса
    """
    random.seed(seed)
    source_path = Path(source_path)
    
    # Group-aware split для CSV
    if csv_path and group_column:
        import pandas as pd
        from sklearn.model_selection import GroupShuffleSplit
        
        df = pd.read_csv(csv_path)
        groups = df[group_column].unique()
        
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=seed)
        train_idx, temp_idx = next(gss.split(df, groups=df[group_column]))
        
        train_df = df.iloc[train_idx]
        temp_df = df.iloc[temp_idx]
        
        gss2 = GroupShuffleSplit(n_splits=1, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)
        val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_column]))
        
        val_df = temp_df.iloc[val_idx]
        test_df = temp_df.iloc[test_idx]
        
        print(f"✅ Group-aware split по колонке '{group_column}'")
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print(f"  Групп в train: {train_df[group_column].nunique()}")
        print(f"  Групп в val: {val_df[group_column].nunique()}")
        print(f"  Групп в test: {test_df[group_column].nunique()}")
        
        return train_df, val_df, test_df
    
    if not source_path.exists():
        print(f"❌ Путь {source_path} не существует")
        return
    
    classes = sorted([d.name for d in source_path.iterdir() if d.is_dir()])
    
    if not classes:
        print(f"❌ В папке {source_path} нет подпапок с классами")
        return
    
    print(f"📁 Классы: {classes}")
    
    # Собираем изображения
    class_images = {}
    total_images = 0
    
    for class_name in classes:
        class_path = source_path / class_name
        images = [str(img) for img in class_path.glob("*") 
                  if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if len(images) == 0:
            print(f"⚠️ Класс {class_name} пустой, пропускаю")
            continue
        
        if len(images) < 3:
            print(f"⚠️ Класс {class_name} имеет только {len(images)} изображений (мало для split)")
        
        class_images[class_name] = images
        total_images += len(images)
    
    if total_images == 0:
        print("❌ Нет изображений для разделения")
        return
    
    print(f"📊 Всего изображений: {total_images}")
    
    # Проверяем минимальные требования для КАЖДОГО класса
    for class_name, images in class_images.items():
        if len(images) < 3:
            print(f"⚠️ Класс {class_name}: только {len(images)} изображений. Все в train.")
            continue
        
        min_needed = min_val_per_class
        if len(images) < min_needed:
            min_needed = max(1, len(images) - 2)
            print(f"⚠️ Класс {class_name}: min_val_per_class={min_val_per_class} > {len(images)}. Использую {min_needed}.")
    
    # Создаем структуру
    for split_name, ratio in [('train', train_ratio), ('val', val_ratio), ('test', test_ratio)]:
        if ratio <= 0:
            continue
        split_path = source_path.parent / split_name
        if split_path.exists():
            shutil.rmtree(split_path)
        split_path.mkdir(parents=True, exist_ok=True)
        for cls in class_images.keys():
            (split_path / cls).mkdir(exist_ok=True)
    
    # Разделяем каждый класс отдельно
    stats = {}
    
    for class_name, images in class_images.items():
        n_total = len(images)
        
        # Вычисляем целевое число val с учетом min_val_per_class
        target_val = int(n_total * val_ratio)
        # min_val ограничен доступным количеством (оставляем 1 train и 1 test)
        min_val = min(min_val_per_class, max(1, n_total - 2))
        # Берем максимум из целевого и минимального
        n_val = max(target_val, min_val)
        # Гарантируем что n_val не превышает n_total - 2
        n_val = min(n_val, n_total - 2)
        
        n_test = max(1, int(n_total * test_ratio))
        n_train = n_total - n_val - n_test
        
        if n_train < 1:
            # Если слишком мало данных, все в train
            n_train = max(1, n_total - 2)
            n_val = 1 if n_total > 1 else 0
            n_test = 0
        
        random.shuffle(images)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        # Копируем
        for img in train_imgs:
            shutil.copy(img, source_path.parent / 'train' / class_name / Path(img).name)
        for img in val_imgs:
            shutil.copy(img, source_path.parent / 'val' / class_name / Path(img).name)
        for img in test_imgs:
            shutil.copy(img, source_path.parent / 'test' / class_name / Path(img).name)
        
        stats[class_name] = {'train': len(train_imgs), 'val': len(val_imgs), 'test': len(test_imgs)}
    
    # Выводим статистику
    print("\n📊 Статистика разделения:")
    print(f"{'Класс':<20} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 40)
    for cls, s in stats.items():
        print(f"{cls:<20} {s['train']:>6} {s['val']:>6} {s['test']:>6}")
    
    print(f"\n✅ Данные разделены!")
    print(f"   Train: {source_path.parent / 'train'}")
    print(f"   Val: {source_path.parent / 'val'}")
    print(f"   Test: {source_path.parent / 'test'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Стратифицированное разделение данных')
    parser.add_argument('--source', required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_val_per_class', type=int, default=3)
    parser.add_argument('--csv_path', type=str, default=None, help='Путь к CSV с разметкой')
    parser.add_argument('--group_column', type=str, default=None, help='Колонка для group-aware split')
    
    args = parser.parse_args()
    
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"⚠️ Сумма ratios = {total_ratio}, нормализую...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    split_dataset(args.source, args.train_ratio, args.val_ratio, args.test_ratio,
                 args.seed, min_val_per_class=args.min_val_per_class,
                 group_column=args.group_column, csv_path=args.csv_path)
