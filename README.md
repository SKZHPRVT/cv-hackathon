# CV Hackathon Toolkit Pro 🚀

Готовый инструментарий для быстрого решения задач компьютерного зрения на хакатонах.

## 📁 Структура проекта

```
cv_hackathon/
├── configs/                    # Конфигурационные файлы
│   ├── config.yaml             # Основной конфиг обучения
│   └── yolo_data.yaml          # Конфиг для YOLO
│
├── utils/                      # Вспомогательные модули
│   ├── __init__.py             # Инициализация пакета
│   ├── dataset.py              # Загрузка данных и аугментации
│   ├── metrics.py              # Метрики и визуализации
│   ├── split_data.py           # Разделение данных
│   └── check_data.py           # Анализ датасета
│
├── checkpoints/                # Сохраненные модели
│   ├── best_model.pth          # Лучшая модель
│   └── last_model.pth          # Последняя модель
│
├── data/                       # Данные
│   ├── train/                  # Тренировочные данные
│   ├── val/                    # Валидационные данные
│   └── test/                   # Тестовые данные
│
├── models/                     # Дополнительные модели
│
├── train.py                    # Скрипт обучения (классификация)
├── yolo_train.py               # Скрипт обучения YOLO (детекция)
├── predict.py                  # Скрипт инференса
├── tta_predict.py              # Инференс с TTA
├── export_onnx.py              # Конвертация в ONNX
├── app.py                      # Веб-интерфейс (Gradio)
├── check_env.py                # Проверка окружения
│
├── MANUAL.txt                  # Полное руководство
├── CHEATSHEET.txt              # Шпаргалка с командами
├── PRESENTATION_TEMPLATE.txt   # Шаблон презентации
├── HACKATHON_GUIDE.txt         # Руководство по хакатону
│
├── requirements.txt            # Зависимости
├── .gitignore                  # Игнорирование файлов
└── README.md                   # Этот файл
```

## 🚀 Быстрый старт

### 1. Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Проверка окружения

```bash
python check_env.py
```

### 3. Подготовка данных

```bash
# Если данные в одной папке:
python utils/split_data.py --source data/all_data

# Проверить данные:
python utils/check_data.py --path data/train
```

### 4. Обучение классификации

```bash
# Быстрый тест (1 эпоха):
python train.py --fast

# Полное обучение:
python train.py

# С конкретной моделью:
python train.py --model efficientnet_b0
```

### 5. Обучение детекции (YOLO)

```bash
# Обучение YOLO:
python yolo_train.py --mode train --data configs/yolo_data.yaml --model n --epochs 50

# Инференс YOLO:
python yolo_train.py --mode predict --model runs/detect/train/weights/best.pt --source test.jpg

# Экспорт YOLO в ONNX:
python yolo_train.py --mode export --model runs/detect/train/weights/best.pt
```

### 6. Инференс

```bash
# Обычное предсказание:
python predict.py --checkpoint checkpoints/best_model.pth --image test.jpg

# Предсказание с TTA (повышенная точность):
python tta_predict.py --checkpoint checkpoints/best_model.pth --image test.jpg

# Конвертация в ONNX:
python export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx
```

### 7. Веб-интерфейс

```bash
python app.py
# Открыть http://localhost:7860
```

## 📚 Документация

- **MANUAL.txt** — полное руководство по всем файлам и функциям
- **CHEATSHEET.txt** — быстрые команды для работы
- **PRESENTATION_TEMPLATE.txt** — шаблон для презентации результатов
- **HACKATHON_GUIDE.txt** — руководство по хакатону с подводными камнями

## 🎯 Поддерживаемые задачи

### Классификация (timm)
- `resnet18`, `resnet34`, `resnet50` — быстрые и надежные
- `efficientnet_b0`, `efficientnet_b3` — точные и эффективные
- `mobilenetv3_large_100` — для мобильных устройств
- `vit_base_patch16_224` — Vision Transformer

### Детекция (YOLO)
- `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- Задачи: detect, segment, classify, pose

## 📊 Аугментации (Albumentations)

- HorizontalFlip
- RandomRotate90
- ShiftScaleRotate
- RandomBrightnessContrast
- HueSaturationValue
- CoarseDropout (CutOut)

## 🔧 Основные команды

```bash
# Проверка окружения
python check_env.py

# Анализ данных
python utils/check_data.py --path data/train

# Разделение данных
python utils/split_data.py --source data/all_data

# Обучение классификации
python train.py --model resnet18 --epochs 50

# Обучение детекции
python yolo_train.py --mode train --data configs/yolo_data.yaml

# Инференс
python predict.py --checkpoint checkpoints/best_model.pth --image test.jpg

# TTA предсказание
python tta_predict.py --checkpoint checkpoints/best_model.pth --image test.jpg

# ONNX конвертация
python export_onnx.py --checkpoint checkpoints/best_model.pth

# Веб-интерфейс
python app.py
```

## 💡 Советы для хакатона

1. **Начните с простого**: ResNet18 + базовые аугментации
2. **Итеративно улучшайте**: добавляйте сложность постепенно
3. **Следите за метриками**: используйте графики и confusion matrix
4. **Автоматизируйте**: используйте app.py для быстрых экспериментов
5. **Документируйте**: сохраняйте все эксперименты и результаты
6. **Работайте в команде**: разделите задачи между участниками
7. **Готовьте демо**: сделайте красивый интерфейс для презентации
8. **Тестируйте заранее**: проверьте весь пайплайн до хакатона

## 🔍 Решение проблем

### CUDA out of memory

```bash
python train.py --batch_size 16
python train.py --model resnet18
```

### Медленное обучение

- Уменьшите `image_size` в конфиге
- Увеличьте `num_workers`
- Используйте `mixed_precision: true`

### Overfitting

- Добавьте аугментации
- Увеличьте `weight_decay`
- Используйте early stopping

## 📈 Метрики

Модель автоматически считает:

- Accuracy
- F1-score (macro и weighted)
- Precision
- Recall
- Confusion Matrix

Все графики сохраняются в `checkpoints/`:

- `training_history_*.png` — графики обучения
- `confusion_matrix.png` — матрица ошибок

## 🎓 Подготовка к хакатону

### Перед хакатоном

1. Установите все зависимости
2. Проверьте работу скриптов
3. Скачайте предобученные модели
4. Изучите документацию
5. Потренируйтесь на тестовых данных

### Во время хакатона

1. Изучите данные
2. Запустите baseline
3. Улучшайте итеративно
4. Сохраняйте результаты
5. Готовьте презентацию