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
├── train.py                    # Обучение классификации
├── yolo_train.py               # Обучение YOLO (детекция/сегментация)
├── predict.py                  # Инференс классификации
├── tta_predict.py              # Инференс с TTA
├── export_onnx.py              # Конвертация в ONNX
├── embedding_search.py         # Поиск похожих объектов
├── hf_models.py                # Hugging Face модели
├── app.py                      # Веб-интерфейс (Gradio)
├── check_env.py                # Проверка окружения
│
├── MANUAL.txt                  # Полное руководство
├── CHEATSHEET.txt              # Шпаргалка с командами
├── PRESENTATION_TEMPLATE.txt   # Шаблон презентации
├── HACKATHON_GUIDE.txt         # Руководство по хакатону
├── embedding_search_README.txt # Руководство по поиску объектов
├── hf_models_README.txt        # Руководство по HF моделям
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
# Обучение детекции:
python yolo_train.py --mode train --data configs/yolo_data.yaml --model n --epochs 50

# Обучение сегментации:
python yolo_train.py --mode train --data configs/yolo_data.yaml --task segment

# Инференс YOLO:
python yolo_train.py --mode predict --model runs/detect/train/weights/best.pt --source test.jpg
```

### 6. Hugging Face модели

```bash
# Список доступных моделей:
python hf_models.py --mode list

# Классификация:
python hf_models.py --mode classify --image test.jpg --model google/vit-base-patch16-224

# Zero-shot классификация:
python hf_models.py --mode zero_shot --image test.jpg --labels cat dog car person

# Сегментация (SAM):
python hf_models.py --mode segment --image test.jpg

# Генерация:
python hf_models.py --mode generate --prompt "a cat sitting on a table"
```

### 7. Поиск похожих объектов

```bash
# Создать базу эмбеддингов:
python embedding_search.py --mode build --input data/database_images/ --database embeddings.pkl

# Найти похожие:
python embedding_search.py --mode search --query test.jpg --database embeddings.pkl --top_k 5

# Сравнить два изображения:
python embedding_search.py --mode compare --input image1.jpg --image2 image2.jpg
```

### 8. Инференс

```bash
# Обычное предсказание:
python predict.py --checkpoint checkpoints/best_model.pth --image test.jpg

# Предсказание с TTA (повышенная точность):
python tta_predict.py --checkpoint checkpoints/best_model.pth --image test.jpg

# Конвертация в ONNX:
python export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx
```

### 9. Веб-интерфейс

```bash
python app.py
# Открыть http://localhost:7860
```

Веб-интерфейс включает 5 вкладок:
- 🔮 Инференс — классификация, батч-обработка
- 📊 Анализ данных — статистика, аугментации
- 🔀 Разделение данных — train/val/test
- 🎓 Обучение — запуск обучения
- 🤗 Hugging Face — ViT, CLIP, SAM, Stable Diffusion

## 📚 Документация

- **MANUAL.txt** — полное руководство по всем файлам и функциям
- **CHEATSHEET.txt** — быстрые команды для работы
- **PRESENTATION_TEMPLATE.txt** — шаблон для презентации результатов
- **HACKATHON_GUIDE.txt** — руководство по хакатону с подводными камнями
- **embedding_search_README.txt** — руководство по поиску похожих объектов
- **hf_models_README.txt** — руководство по Hugging Face моделям

## 🎯 Поддерживаемые задачи

### 1. Классификация изображений
- **timm**: resnet18, resnet34, resnet50, efficientnet_b0, efficientnet_b3, mobilenetv3_large_100, vit_base_patch16_224
- **Hugging Face**: ViT, Swin, ConvNeXT, ResNet

### 2. Детекция объектов (YOLO)
- yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
- Задачи: detect, segment, classify, pose

### 3. Сегментация
- **YOLO-seg**: инстанс-сегментация
- **SAM**: Segment Anything Model
- **DETR**: детекция + сегментация

### 4. Распознавание и поиск
- Поиск похожих объектов (embedding search)
- Распознавание лиц
- Re-identification
- Поиск дубликатов

### 5. Zero-shot классификация (CLIP)
- Классификация без обучения
- Распознавание новых классов
- Текстовая классификация

### 6. Генерация изображений
- Stable Diffusion
- Создание синтетических данных
- Аугментация датасета

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

# Hugging Face
python hf_models.py --mode list

# Поиск похожих
python embedding_search.py --mode search --query test.jpg --database embeddings.pkl

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
9. **Скачайте модели заранее**: HF модели требуют интернет при первом запуске

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

### HF модель не загружается

- Проверьте интернет-соединение
- Скачайте модель заранее
- Используйте локальный путь

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
6. Скачайте HF модели заранее

### Во время хакатона

1. Изучите данные
2. Запустите baseline
3. Улучшайте итеративно
4. Сохраняйте результаты
5. Готовьте презентацию