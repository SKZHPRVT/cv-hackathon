"""
Поиск дубликатов и утечек (leakage) в датасете.
Ищет точные и почти точные дубликаты по хешам и эмбеддингам.
"""
import hashlib
import cv2
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
from PIL import Image
import imagehash

def get_md5(file_path):
    """MD5 хеш файла."""
    return hashlib.md5(open(file_path, 'rb').read()).hexdigest()

def get_phash(image_path, hash_size=8):
    """Perceptual hash изображения."""
    try:
        img = Image.open(image_path)
        return str(imagehash.phash(img, hash_size=hash_size))
    except:
        return None

def get_ahash(image_path, hash_size=8):
    """Average hash изображения."""
    try:
        img = Image.open(image_path)
        return str(imagehash.average_hash(img, hash_size=hash_size))
    except:
        return None

def find_exact_duplicates(data_path):
    """Поиск точных дубликатов по MD5."""
    print("\n🔍 Поиск точных дубликатов (MD5)...")
    
    hashes = defaultdict(list)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for img_path in Path(data_path).rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            md5 = get_md5(img_path)
            hashes[md5].append(str(img_path))
    
    duplicates = {k: v for k, v in hashes.items() if len(v) > 1}
    
    if duplicates:
        print(f"⚠️ Найдено {len(duplicates)} групп точных дубликатов:")
        for md5, paths in duplicates.items():
            print(f"\n  MD5: {md5[:16]}...")
            for p in paths:
                print(f"    - {p}")
    else:
        print("✅ Точных дубликатов не найдено")
    
    return duplicates

def find_near_duplicates(data_path, threshold=10):
    """Поиск почти дубликатов по perceptual hash."""
    print(f"\n🔍 Поиск почти дубликатов (pHash, порог {threshold})...")
    
    phashes = defaultdict(list)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for img_path in Path(data_path).rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            phash = get_phash(img_path)
            if phash:
                phashes[phash].append(str(img_path))
    
    # Группируем по похожести
    near_duplicates = []
    phash_keys = list(phashes.keys())
    
    for i in range(len(phash_keys)):
        for j in range(i+1, len(phash_keys)):
            distance = imagehash.hex_to_hash(phash_keys[i]) - imagehash.hex_to_hash(phash_keys[j])
            if distance <= threshold:
                near_duplicates.append((phash_keys[i], phash_keys[j], distance))
    
    if near_duplicates:
        print(f"⚠️ Найдено {len(near_duplicates)} пар почти дубликатов:")
        for hash1, hash2, dist in near_duplicates:
            print(f"\n  Расстояние: {dist}")
            print(f"    - {phashes[hash1][0]}")
            print(f"    - {phashes[hash2][0]}")
    else:
        print("✅ Почти дубликатов не найдено")
    
    return near_duplicates

def find_cross_split_leakage(train_path, val_path, test_path=None):
    """Поиск утечек между train/val/test."""
    print("\n🔍 Поиск утечек между выборками...")
    
    def get_hashes(path):
        hashes = set()
        if Path(path).exists():
            for img_path in Path(path).rglob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    hashes.add(get_md5(img_path))
        return hashes
    
    train_hashes = get_hashes(train_path)
    val_hashes = get_hashes(val_path)
    test_hashes = get_hashes(test_path) if test_path else set()
    
    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes
    
    if train_val_overlap:
        print(f"⚠️ Утечка train-val: {len(train_val_overlap)} изображений")
    else:
        print("✅ Train-Val: утечек нет")
    
    if train_test_overlap:
        print(f"⚠️ Утечка train-test: {len(train_test_overlap)} изображений")
    else:
        print("✅ Train-Test: утечек нет")
    
    if val_test_overlap:
        print(f"⚠️ Утечка val-test: {len(val_test_overlap)} изображений")
    else:
        print("✅ Val-Test: утечек нет")
    
    return train_val_overlap, train_test_overlap, val_test_overlap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Поиск дубликатов и утечек')
    parser.add_argument('--data', required=True, help='Путь к данным')
    parser.add_argument('--train', help='Путь к train')
    parser.add_argument('--val', help='Путь к val')
    parser.add_argument('--test', help='Путь к test')
    parser.add_argument('--exact', action='store_true', help='Поиск точных дубликатов')
    parser.add_argument('--near', action='store_true', help='Поиск почти дубликатов')
    parser.add_argument('--leakage', action='store_true', help='Поиск утечек между выборками')
    parser.add_argument('--threshold', type=int, default=10, help='Порог для почти дубликатов')
    
    args = parser.parse_args()
    
    if args.exact or not (args.near or args.leakage):
        find_exact_duplicates(args.data)
    
    if args.near:
        try:
            import imagehash
        except ImportError:
            print("❌ Установите: pip install imagehash")
            exit(1)
        find_near_duplicates(args.data, args.threshold)
    
    if args.leakage:
        find_cross_split_leakage(args.train, args.val, args.test)
