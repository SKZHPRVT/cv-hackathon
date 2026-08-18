# CV Hackathon Toolkit Pro 🚀

Готовый инструментарий для быстрого решения задач компьютерного зрения на хакатонах.

## 📁 Структура проекта

```
cv_hackathon/
├── configs/                    # Конфигурационные файлы
│   ├── config.yaml             # Основной конфиг (аугментации конфигурируемые)
│   └── yolo_data.yaml          # Конфиг для YOLO
│
├── utils/                      # Вспомогательные модули
│   ├── __init__.py             # Инициализация пакета
│   ├── dataset.py              # Датасеты (folder + CSV), аугментации
│   ├── metrics.py              # Метрики (число классов из class_names)
│   ├── split_data.py           # Стратифицированное разделение
│   ├── check_data.py           # Анализ датасета
│   └── find_duplicates.py      # Поиск дубликатов и утечек
│
├── checkpoints/                # Сохраненные модели
│   ├── best_model.pth          # Лучшая модель
│   ├── last_model.pth          # Последняя модель
│   └── experiment_*.json       # Журналы экспериментов
│
├── data/                       # Данные
│   ├── train/                  # Тренировочные
│   ├── val/                    # Валидационные
│   └── test/                   # Тестовые
│
├── models/                     # Кеш моделей (offline)
│
├── train.py                    # Обучение (seed, журнал, гарантия чекпоинта)
├── yolo_train.py               # YOLO (CUDA через PyTorch, произвольный путь)
├── predict.py                  # Инференс
├── tta_predict.py              # TTA (none/hflip/vflip/all)
├── export_onnx.py              # ONNX + проверка эквивалентности
├── embedding_search.py         # Поиск похожих объектов
├── hf_models.py                # Hugging Face модели
├── generate_submission.py      # Генерация submission.csv
├── download_models.py          # Скачивание моделей для offline
├── app.py                      # Веб-интерфейс (RGB/BGR fix, HF кеш)
├── check_env.py                # Проверка окружения + smoke test
├── test_pipeline.sh            # Полный тест пайплайна
│
├── MANUAL.txt                  # Полное руководство
├── CHEATSHEET.txt              # Шпаргалка
├── PRESENTATION_TEMPLATE.txt   # Шаблон презентации
├── HACKATHON_GUIDE.txt         # Гайд по хакатону
├── APP_GUIDE.txt               # Гайд по веб-интерфейсу
├── embedding_search_README.txt # Гайд по поиску
├── hf_models_README.txt        # Гайд по HF моделям
│
├── requirements.txt            # Зависимости (диапазоны)
├── requirements-lock.txt       # Точные версии (зафиксированные)
├── .gitignore                  # Игнорирование
└── README.md                   # Этот файл
```

## 🚀 Быстрый старт

### 1. Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Или точные версии (воспроизводимость):
pip install -r requirements-lock.txt
```

### 2. Проверка окружения (включая smoke test)

```bash
python check_env.py

# С YOLO тестом:
python check_env.py --with-yolo
```

### 3. Скачивание моделей для offline-режима

```bash
# Все модели:
python download_models.py --all

# Только timm:
python download_models.py --timm

# Только YOLO:
python download_models.py --yolo

# Только HF:
python download_models.py --hf

# Проверить кеш:
python download_models.py --check
```

### 4. Подготовка данных

```bash
# Стратифицированное разделение:
python utils/split_data.py --source data/all_data --seed 42

# Group-aware split (CSV с колонкой group):
python utils/split_data.py --source data/all_data --csv data.csv --group_column patient_id

# Проверить данные:
python utils/check_data.py --path data/train

# Поиск дубликатов:
python utils/find_duplicates.py --data data/train --exact

# Поиск почти дубликатов:
python utils/find_duplicates.py --data data/train --near

# Поиск утечек:
python utils/find_duplicates.py --data data/train --leakage --train data/train --val data/val --test data/test
```

### 5. Обучение классификации

```bash
# Быстрый тест:
python train.py --fast --seed 42

# Полное обучение:
python train.py --model resnet18 --epochs 50 --seed 42 --exp_name baseline

# С CSV-разметкой (укажите train_csv/val_csv в config.yaml)
```

### 6. Обучение YOLO

```bash
# Детекция:
python yolo_train.py --mode train --data configs/yolo_data.yaml --model n --epochs 50

# Сегментация:
python yolo_train.py --mode train --data configs/yolo_data.yaml --task segment
```

### 7. Инференс

```bash
# Обычное:
python predict.py --checkpoint checkpoints/best_model.pth --image test.jpg

# TTA:
python tta_predict.py --checkpoint checkpoints/best_model.pth --image test.jpg --tta hflip
python tta_predict.py --checkpoint checkpoints/best_model.pth --image test.jpg --tta all

# ONNX с проверкой:
python export_onnx.py --checkpoint checkpoints/best_model.pth --test --speed
```

### 8. Генерация submission

```bash
# Одна метка:
python generate_submission.py --checkpoint checkpoints/best_model.pth --test_folder data/test --format label

# Вероятности:
python generate_submission.py --checkpoint checkpoints/best_model.pth --test_folder data/test --format probabilities

# YOLO detection:
python generate_submission.py --checkpoint yolov8n.pt --test_folder data/test --format yolo
```

### 9. Веб-интерфейс

```bash
python app.py
# http://localhost:7860
```

### 10. Полный тест пайплайна

```bash
bash test_pipeline.sh
```

## 📚 Документация

- **MANUAL.txt** — полное руководство по всем файлам и функциям
- **CHEATSHEET.txt** — быстрые команды
- **HACKATHON_GUIDE.txt** — гайд по хакатону
- **APP_GUIDE.txt** — веб-интерфейс
- **PRESENTATION_TEMPLATE.txt** — презентация
- **embedding_search_README.txt** — поиск объектов
- **hf_models_README.txt** — HF модели

## 🎯 Поддерживаемые задачи

### 1. Классификация изображений
- **timm**: resnet18, resnet34, resnet50, efficientnet_b0, efficientnet_b3, mobilenetv3_large_100, vit_base_patch16_224
- **Hugging Face**: ViT, Swin, ConvNeXT, ResNet

### 2. Детекция объектов (YOLO)
- yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
- Задачи: detect, segment, classify, pose

### 3. Сегментация
- YOLO-seg: инстанс-сегментация
- SAM: Segment Anything Model
- DETR: детекция + сегментация

### 4. Распознавание и поиск
- Поиск похожих объектов
- Распознавание лиц
- Re-identification

### 5. Zero-shot классификация (CLIP)
- Без обучения
- Новые классы

### 6. Генерация изображений
- Stable Diffusion
- Синтетические данные

## 📊 Аугментации (конфигурируемые, все off по умолчанию)

```yaml
augmentation:
  horizontal_flip: false
  vertical_flip: false
  random_rotate: false
  random_crop: false
  brightness_contrast: false
  hue_saturation: false
  coarse_dropout: false
```

## 🔍 Ключевые улучшения

- ✅ Reproducibility (seed для Python, NumPy, PyTorch, CUDA)
- ✅ Стратифицированный split с защитой от пустых классов
- ✅ Гарантия сохранения чекпоинта (даже при val_acc=0)
- ✅ Единый препроцессинг (mean/std/image_size из чекпоинта)
- ✅ Поиск дубликатов (MD5 + pHash + leakage)
- ✅ Smoke test (forward, optimizer, save/load, DataLoader)
- ✅ Журнал экспериментов (JSON)
- ✅ CSV-разметка
- ✅ Submission generator (label / probabilities)
- ✅ Offline-режим (download_models.py)
- ✅ HF кеширование в памяти
- ✅ ONNX эквивалентность
- ✅ Полный тест пайплайна (test_pipeline.sh, 13 шагов)
- ✅ Lock-файл зависимостей
- ✅ Group-aware split
- ✅ YOLO submission
- ✅ Smoke test с exit code

## 🔧 Основные команды

```bash
# Проверка окружения
python check_env.py

# Offline модели
python download_models.py --all

# Данные
python utils/split_data.py --source data/all_data --seed 42
python utils/check_data.py --path data/train
python utils/find_duplicates.py --data data/train --exact

# Обучение
python train.py --model resnet18 --epochs 50 --seed 42 --exp_name baseline

# Инференс
python predict.py --checkpoint checkpoints/best_model.pth --image test.jpg

# TTA
python tta_predict.py --checkpoint checkpoints/best_model.pth --image test.jpg --tta all

# Submission
python generate_submission.py --checkpoint checkpoints/best_model.pth --test_folder data/test

# Полный тест
bash test_pipeline.sh
```

## 💡 Советы для хакатона

1. **Начните с простого**: ResNet18 + базовые аугментации
2. **Итеративно улучшайте**: добавляйте сложность постепенно
3. **Следите за метриками**: графики и confusion matrix
4. **Автоматизируйте**: app.py для экспериментов
5. **Документируйте**: журналы экспериментов
6. **Работайте в команде**: разделите задачи
7. **Готовьте демо**: веб-интерфейс
8. **Тестируйте заранее**: test_pipeline.sh
9. **Скачайте модели заранее**: download_models.py --all

## 🔍 Решение проблем

### CUDA out of memory

```bash
python train.py --batch_size 16
python train.py --model resnet18
```

### Медленное обучение

- Уменьшите `image_size`
- Увеличьте `num_workers`
- Используйте `mixed_precision: true`

### Overfitting

- Включите аугментации в config
- Увеличьте `weight_decay`
- Используйте early stopping

### Порт занят

```bash
lsof -ti:7860 | xargs kill -9
```

## 🎓 Подготовка к хакатону

### Перед хакатоном

1. Установите зависимости
2. Проверьте окружение (check_env.py)
3. Скачайте модели (download_models.py)
4. Запустите полный тест (test_pipeline.sh)
5. Изучите документацию

### Во время хакатона

1. Изучите данные
2. Проверьте дубликаты
3. Запустите baseline
4. Улучшайте итеративно
5. Готовьте submission