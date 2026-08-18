import cv2
import numpy as np
from pathlib import Path
import argparse
from collections import Counter

def analyze_dataset(data_path):
    """Анализ датасета: размеры, количество, баланс классов"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"❌ Путь {data_path} не существует")
        return
    
    classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    if not classes:
        print(f"❌ В папке {data_path} нет подпапок с классами")
        return
    
    print(f"📁 Classes found: {classes}")
    print("="*50)
    
    total_images = 0
    sizes = []
    class_stats = {}
    
    for class_name in classes:
        class_path = data_path / class_name
        images = list(class_path.glob("*"))
        images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        class_sizes = []
        for img in images:
            try:
                im = cv2.imread(str(img))
                if im is not None:
                    h, w, c = im.shape
                    class_sizes.append((w, h))
                    sizes.append((w, h))
            except Exception as e:
                print(f"⚠️ Error reading {img}: {e}")
        
        class_stats[class_name] = {
            'count': len(images),
            'avg_size': np.mean(class_sizes, axis=0) if class_sizes else (0, 0)
        }
        total_images += len(images)
    
    print(f"📊 Total images: {total_images}")
    print("\n📈 Class balance:")
    for class_name, stats in class_stats.items():
        print(f"  {class_name}: {stats['count']} images, avg size: {stats['avg_size'][0]:.0f}x{stats['avg_size'][1]:.0f}")
    
    # Проверка на дисбаланс классов
    if class_stats:
        counts = [stats['count'] for stats in class_stats.values()]
        min_count = min(counts)
        max_count = max(counts)
        if max_count > 0 and min_count / max_count < 0.5:
            print(f"\n⚠️ Внимание: сильный дисбаланс классов!")
            print(f"  Min: {min_count}, Max: {max_count}, Ratio: {min_count/max_count:.2f}")
    
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"\n📐 Image size range:")
        print(f"  Width: {min(widths)} - {max(widths)}")
        print(f"  Height: {min(heights)} - {max(heights)}")
        print(f"  Mean: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
    
    # Проверка на битые изображения
    print("\n🔍 Checking for corrupted images...")
    corrupted = []
    for class_name in classes:
        class_path = data_path / class_name
        for img in class_path.glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    im = cv2.imread(str(img))
                    if im is None:
                        corrupted.append(str(img))
                except:
                    corrupted.append(str(img))
    
    if corrupted:
        print(f"⚠️ Found {len(corrupted)} corrupted images:")
        for img in corrupted[:10]:
            print(f"  {img}")
        if len(corrupted) > 10:
            print(f"  ... and {len(corrupted)-10} more")
    else:
        print("✅ No corrupted images found")
    
    # Рекомендации
    print("\n💡 Рекомендации:")
    if total_images < 100:
        print("  • Мало данных! Используйте сильные аугментации")
        print("  • Рассмотрите transfer learning с заморозкой слоев")
    if len(classes) == 2 and total_images >= 100:
        print("  • Бинарная классификация - начните с ResNet18")
    if len(classes) > 10:
        print("  • Много классов - используйте EfficientNet")
    
    # Проверка форматов
    formats = Counter()
    for class_name in classes:
        class_path = data_path / class_name
        for img in class_path.glob("*"):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                formats[img.suffix.lower()] += 1
    
    print("\n📁 Форматы изображений:")
    for fmt, count in formats.most_common():
        print(f"  {fmt}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to dataset')
    args = parser.parse_args()
    analyze_dataset(args.path)
