import gradio as gr
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import os
import sys
import yaml
from pathlib import Path
import shutil
import tempfile
import io
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from contextlib import redirect_stdout
from datetime import datetime
import threading
import queue
import time

# Добавляем utils в path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import create_dataloaders, get_transforms
from utils.check_data import analyze_dataset
from utils.split_data import split_dataset

# Глобальные переменные
MODEL = None
CLASS_NAMES = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAINING_QUEUE = queue.Queue()
TRAINING_LOG = []

def load_model_ui(checkpoint_path):
    """Загрузка модели через UI"""
    global MODEL, CLASS_NAMES
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        config = checkpoint['config']
        model_name = config['model']['name']
        CLASS_NAMES = checkpoint['class_names']
        
        MODEL = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES))
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()
        
        return f"✅ Модель загружена: {model_name}\nКлассы: {CLASS_NAMES}"
    except Exception as e:
        return f"❌ Ошибка загрузки модели: {str(e)}"

def predict_ui(image, checkpoint_path, top_k=3):
    """Предсказание через UI"""
    if image is None:
        return "Пожалуйста, загрузите изображение", None
    
    # Загружаем модель если не загружена
    if MODEL is None:
        result = load_model_ui(checkpoint_path)
        if "❌" in result:
            return result, None
    
    try:
        # Предобработка
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        augmented = transform(image=image_rgb)
        image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        
        # Предсказание
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Топ-K предсказания
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(CLASS_NAMES)))
        
        # Текстовый результат
        result = "🎯 Предсказания:\n"
        for prob, idx in zip(top_probs[0], top_indices[0]):
            result += f"{CLASS_NAMES[idx]}: {prob.item()*100:.2f}%\n"
        
        # График
        fig = go.Figure(data=[
            go.Bar(
                x=[CLASS_NAMES[idx] for idx in top_indices[0]],
                y=[prob.item()*100 for prob in top_probs[0]],
                marker_color='lightblue'
            )
        ])
        fig.update_layout(
            title="Вероятности классов",
            yaxis_title="Вероятность (%)",
            xaxis_title="Класс",
            height=400
        )
        
        return result, fig
    except Exception as e:
        return f"❌ Ошибка: {str(e)}", None

def predict_batch_ui(folder_path, checkpoint_path):
    """Батч-предсказание для папки"""
    if not folder_path:
        return "Выберите папку"
    
    try:
        if MODEL is None:
            result = load_model_ui(checkpoint_path)
            if "❌" in result:
                return result
        
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        folder = Path(folder_path)
        
        for img_path in folder.iterdir():
            if img_path.suffix.lower() in image_extensions:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transform = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
                augmented = transform(image=image_rgb)
                image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = MODEL(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                
                results.append({
                    'image': img_path.name,
                    'predicted_class': CLASS_NAMES[predicted.item()],
                    'confidence': probabilities[0][predicted].item() * 100
                })
        
        if results:
            df = pd.DataFrame(results)
            return df
        else:
            return "Нет изображений в папке"
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

def analyze_data_ui(data_path):
    """Анализ данных через UI с визуализацией"""
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            analyze_dataset(data_path)
        text_output = f.getvalue()
        
        # Создаем визуализацию распределения классов
        data_path = Path(data_path)
        classes = [d.name for d in data_path.iterdir() if d.is_dir()]
        class_counts = []
        
        for class_name in classes:
            class_path = data_path / class_name
            count = len([f for f in class_path.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            class_counts.append({'class': class_name, 'count': count})
        
        df = pd.DataFrame(class_counts)
        fig = px.bar(df, x='class', y='count', title='Распределение классов')
        fig.update_layout(height=400)
        
        return text_output, fig
    except Exception as e:
        return f"❌ Ошибка: {str(e)}", None

def split_data_ui(source_path, train_ratio, val_ratio, test_ratio):
    """Разделение данных через UI"""
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            split_dataset(source_path, train_ratio, val_ratio, test_ratio)
        return f.getvalue()
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

def train_ui_async(config_path, epochs, model_name, fast_mode, progress=gr.Progress()):
    """Асинхронное обучение с прогрессом"""
    try:
        import subprocess
        import sys
        
        cmd = [sys.executable, "train.py", "--config", config_path]
        
        if epochs:
            cmd.extend(["--epochs", str(int(epochs))])
        if model_name:
            cmd.extend(["--model", model_name])
        if fast_mode:
            cmd.append("--fast")
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1
        )
        
        output_lines = []
        
        # Читаем вывод в реальном времени
        import select
        import fcntl
        
        # Делаем stdout неблокирующим
        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        
        start_time = time.time()
        
        while True:
            # Проверяем процесс
            if process.poll() is not None:
                break
            
            # Читаем вывод
            try:
                line = process.stdout.readline()
                if line:
                    output_lines.append(line.strip())
                    # Пытаемся распарсить прогресс
                    if 'Epoch' in line and '/' in line:
                        try:
                            current_epoch = int(line.split('Epoch')[1].split('/')[0])
                            total_epochs = int(line.split('/')[1].split(':')[0])
                            progress(current_epoch/total_epochs, desc=f"Обучение: эпоха {current_epoch}/{total_epochs}")
                        except:
                            pass
            except:
                pass
            
            time.sleep(0.1)
            
            # Таймаут
            if time.time() - start_time > 3600:  # 1 час максимум
                process.kill()
                return "⏱️ Превышен таймаут обучения (1 час)"
        
        # Читаем оставшийся вывод
        remaining_output = process.stdout.read()
        if remaining_output:
            output_lines.extend(remaining_output.strip().split('\n'))
        
        return "\n".join(output_lines[-100:])  # Последние 100 строк
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

def visualize_augmentations_ui(image, num_examples=5):
    """Визуализация аугментаций"""
    if image is None:
        return None
    
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = get_transforms(224, is_train=True)
        
        augmented_images = []
        for i in range(num_examples):
            augmented = transform(image=image_rgb)
            aug_img = augmented['image'].permute(1, 2, 0).numpy()
            # Денормализация
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            aug_img = (aug_img * std + mean)
            aug_img = np.clip(aug_img, 0, 1)
            augmented_images.append(aug_img)
        
        # Создаем сетку изображений
        fig = go.Figure()
        for i, img in enumerate(augmented_images):
            fig.add_trace(
                go.Image(z=img, name=f'Augmentation {i+1}')
            )
        fig.update_layout(
            title="Примеры аугментаций",
            grid={'rows': 1, 'columns': num_examples},
            height=300
        )
        
        return fig
    except Exception as e:
        print(f"Ошибка визуализации: {e}")
        return None

# Создаем интерфейс
with gr.Blocks(title="CV Hackathon Toolkit Pro", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🚀 CV Hackathon Toolkit Pro
    Максимально быстрый интерфейс для победы на хакатоне
    """)
    
    with gr.Tab("🔮 Инференс"):
        gr.Markdown("### Предсказание на изображении")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Загрузите изображение")
                checkpoint_input = gr.Textbox(
                    label="Путь к чекпоинту",
                    value="checkpoints/best_model.pth",
                    placeholder="checkpoints/best_model.pth"
                )
                top_k = gr.Slider(1, 5, value=3, step=1, label="Топ-K предсказаний")
                predict_btn = gr.Button("🎯 Предсказать", variant="primary", size="lg")
                
                gr.Markdown("### Батч-предсказание")
                folder_input = gr.Textbox(label="Путь к папке с изображениями")
                batch_btn = gr.Button("📁 Обработать папку", variant="secondary")
            
            with gr.Column(scale=1):
                prediction_output = gr.Textbox(label="Результат", lines=8)
                prediction_plot = gr.Plot(label="Вероятности")
                batch_output = gr.Dataframe(label="Результаты батч-обработки")
        
        predict_btn.click(
            fn=predict_ui,
            inputs=[image_input, checkpoint_input, top_k],
            outputs=[prediction_output, prediction_plot]
        )
        
        batch_btn.click(
            fn=predict_batch_ui,
            inputs=[folder_input, checkpoint_input],
            outputs=batch_output
        )
    
    with gr.Tab("📊 Анализ данных"):
        gr.Markdown("### Анализ датасета")
        with gr.Row():
            with gr.Column():
                data_path_input = gr.Textbox(
                    label="Путь к данным",
                    value="data/train",
                    placeholder="data/train"
                )
                analyze_btn = gr.Button("📈 Анализировать", variant="primary", size="lg")
                
                gr.Markdown("### Визуализация аугментаций")
                aug_image_input = gr.Image(label="Исходное изображение")
                aug_btn = gr.Button("🎨 Показать аугментации", variant="secondary")
            
            with gr.Column():
                analysis_output = gr.Textbox(label="Результаты анализа", lines=15)
                class_distribution = gr.Plot(label="Распределение классов")
                aug_output = gr.Plot(label="Аугментации")
        
        analyze_btn.click(
            fn=analyze_data_ui,
            inputs=[data_path_input],
            outputs=[analysis_output, class_distribution]
        )
        
        aug_btn.click(
            fn=visualize_augmentations_ui,
            inputs=[aug_image_input],
            outputs=aug_output
        )
    
    with gr.Tab("🔀 Разделение данных"):
        gr.Markdown("### Разделение на train/val/test")
        with gr.Row():
            source_path_input = gr.Textbox(
                label="Путь к исходным данным",
                placeholder="data/all_data"
            )
        with gr.Row():
            train_ratio = gr.Slider(0.5, 0.95, value=0.8, label="Train Ratio")
            val_ratio = gr.Slider(0.05, 0.3, value=0.1, label="Val Ratio")
            test_ratio = gr.Slider(0.05, 0.3, value=0.1, label="Test Ratio")
        split_btn = gr.Button("🔀 Разделить", variant="primary", size="lg")
        split_output = gr.Textbox(label="Результат", lines=10)
        
        split_btn.click(
            fn=split_data_ui,
            inputs=[source_path_input, train_ratio, val_ratio, test_ratio],
            outputs=split_output
        )
    
    with gr.Tab("🎓 Обучение"):
        gr.Markdown("### Запуск обучения")
        with gr.Row():
            with gr.Column():
                config_path_input = gr.Textbox(
                    label="Путь к конфигу",
                    value="configs/config.yaml"
                )
                epochs_input = gr.Number(label="Эпохи (пусто = из конфига)", value=None, precision=0)
                model_input = gr.Dropdown(
                    label="Модель (пусто = из конфига)",
                    choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b3", "mobilenetv3_large_100"],
                    value=None
                )
                fast_mode = gr.Checkbox(label="Fast mode (1 эпоха)", value=False)
                train_btn = gr.Button("🚀 Обучить", variant="primary", size="lg")
            
            with gr.Column():
                train_output = gr.Textbox(label="Лог обучения", lines=25)
        
        train_btn.click(
            fn=train_ui_async,
            inputs=[config_path_input, epochs_input, model_input, fast_mode],
            outputs=train_output
        )
    
    gr.Markdown("""
    ---
    ### 💡 Горячие клавиши и советы:
    - **Ctrl+Enter** - быстрое предсказание
    - **Drag & Drop** - перетащите изображение для анализа
    - **Fast Mode** - быстрая проверка пайплайна (1 эпоха)
    - **Батч-обработка** - обработайте сотни изображений за раз
    """)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
