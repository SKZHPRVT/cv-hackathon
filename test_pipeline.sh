#!/bin/bash
# Полный тест пайплайна
# Запуск: bash test_pipeline.sh

set -e

echo "========================================="
echo "🧪 ПОЛНЫЙ ТЕСТ ПАЙПЛАЙНА"
echo "========================================="

# 1. Проверка окружения
echo -e "\n[1/8] Проверка окружения..."
python check_env.py

# 2. Создание тестовых данных
echo -e "\n[2/8] Создание тестовых данных..."
python -c "
import numpy as np
import cv2
from pathlib import Path

# Создаем тестовые данные
for split in ['train', 'val', 'test']:
    for cls in ['cat', 'dog']:
        path = Path(f'data/{split}/{cls}')
        path.mkdir(parents=True, exist_ok=True)
        for i in range(5 if split == 'train' else 3):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(path / f'{i}.jpg'), img)

print('✅ Тестовые данные созданы')
"

# 3. Анализ данных
echo -e "\n[3/8] Анализ данных..."
python utils/check_data.py --path data/train

# 4. Поиск дубликатов
echo -e "\n[4/8] Поиск дубликатов..."
python utils/find_duplicates.py --data data/train --exact

# 5. Обучение (fast mode)
echo -e "\n[5/8] Обучение (fast mode)..."
python train.py --fast --seed 42 --exp_name test_run

# 6. Проверка чекпоинта
echo -e "\n[6/8] Проверка чекпоинта..."
if [ -f "checkpoints/best_model.pth" ]; then
    echo "✅ Чекпоинт сохранен"
else
    echo "❌ Чекпоинт не найден"
    exit 1
fi

# 7. Инференс
echo -e "\n[7/8] Инференс..."
python predict.py --checkpoint checkpoints/best_model.pth --image data/val/cat/0.jpg

# 8. Генерация submission
echo -e "\n[8/8] Генерация submission..."
python generate_submission.py --checkpoint checkpoints/best_model.pth --test_folder data/test

echo -e "\n========================================="
echo "🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!"
echo "========================================="
