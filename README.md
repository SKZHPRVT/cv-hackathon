# CV HACKATHON TOOLKIT PRO

Готовый инструментарий для быстрого решения задач компьютерного зрения на хакатонах.

## СТРУКТУРА ПРОЕКТА

cv_hackathon/
├── configs/                    # Конфигурационные файлы
│   └── config.yaml             # Основной конфиг обучения
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
├── train.py                    # Скрипт обучения
├── predict.py                  # Скрипт инференса
├── app.py                      # Веб-интерфейс (Gradio)
├── check_env.py                # Проверка окружения
│
├── MANUAL.txt                  # Полное руководство
├── CHEATSHEET.txt              # Шпаргалка с командами
├── PRESENTATION_TEMPLATE.txt   # Шаблон презентации
│
├── requirements.txt            # Зависимости
├── .gitignore                  # Игнорирование файлов
└── README.md                   # Этот файл

## БЫСТРЫЙ СТАРТ

### 1. Установка

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 2. Проверка окружения

python check_env.py

### 3. Подготовка данных

# Если данные в одной папке:
python utils/split_data.py --source data/all_data

# Проверить данные:
python utils/check_data.py --path data/train

### 4. Обучение

# Быстрый тест (1 эпоха):
python train.py --fast

# Полное обучение:
python train.py

# С конкретной моделью:
python train.py --model efficientnet_b0

### 5. Инференс

# Одно изображение:
python predict.py --checkpoint checkpoints/best_model.pth --image test.jpg

# Папка с изображениями:
python predict.py --checkpoint checkpoints/best_model.pth --folder test_images/

### 6. Веб-интерфейс

python app.py
# Открыть http://localhost:7860

## ДОКУМЕНТАЦИЯ

- MANUAL.txt - полное руководство по всем файлам и функциям
- CHEATSHEET.txt - быстрые команды для работы
- PRESENTATION_TEMPLATE.txt - шаблон для презентации результатов

## ПОДДЕРЖИВАЕМЫЕ МОДЕЛИ (timm)

- resnet18, resnet34, resnet50 - быстрые и надежные
- efficientnet_b0, efficientnet_b3 - точные и эффективные
- mobilenetv3_large_100 - для мобильных устройств
- vit_base_patch16_224 - Vision Transformer

## АУГМЕНТАЦИИ (Albumentations)

- HorizontalFlip
- RandomRotate90
- ShiftScaleRotate
- RandomBrightnessContrast
- HueSaturationValue
- CoarseDropout (CutOut)

## ОСНОВНЫЕ КОМАНДЫ

# Проверка окружения
python check_env.py

# Анализ данных
python utils/check_data.py --path data/train

# Разделение данных
python utils/split_data.py --source data/all_data

# Обучение
python train.py --model resnet18 --epochs 50

# Инференс
python predict.py --checkpoint checkpoints/best_model.pth --image test.jpg

# Веб-интерфейс
python app.py

## СОВЕТЫ ДЛЯ ХАКАТОНА

1. Начните с простого: ResNet18 + базовые аугментации
2. Итеративно улучшайте: добавляйте сложность постепенно
3. Следите за метриками: используйте графики и confusion matrix
4. Автоматизируйте: используйте app.py для быстрых экспериментов
5. Документируйте: сохраняйте все эксперименты и результаты
6. Работайте в команде: разделите задачи между участниками
7. Готовьте демо: сделайте красивый интерфейс для презентации
8. Тестируйте заранее: проверьте весь пайплайн до хакатона

## РЕШЕНИЕ ПРОБЛЕМ

### CUDA out of memory

python train.py --batch_size 16
python train.py --model resnet18

### Медленное обучение

- Уменьшите image_size в конфиге
- Увеличьте num_workers
- Используйте mixed_precision: true

### Overfitting

- Добавьте аугментации
- Увеличьте weight_decay
- Используйте early stopping

## МЕТРИКИ

Модель автоматически считает:
- Accuracy
- F1-score (macro и weighted)
- Precision
- Recall
- Confusion Matrix

Все графики сохраняются в checkpoints/:
- training_history_*.png - графики обучения
- confusion_matrix.png - матрица ошибок

## ПОДГОТОВКА К ХАКАТОНУ

Перед хакатоном:
1. Установите все зависимости
2. Проверьте работу скриптов
3. Скачайте предобученные модели
4. Изучите документацию
5. Потренируйтесь на тестовых данных

Во время хакатона:
1. Изучите данные
2. Запустите baseline
3. Улучшайте итеративно
4. Сохраняйте результаты
5. Готовьте презентацию
