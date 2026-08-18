"""
Веб-интерфейс CV Hackathon Toolkit.
Исправления:
- RGB/BGR正确处理
- Кеширование HF моделей в памяти
"""
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
import io
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from contextlib import redirect_stdout
from datetime import datetime
import time
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import create_dataloaders, get_transforms, MEAN, STD, DEFAULT_IMAGE_SIZE
from utils.check_data import analyze_dataset
from utils.split_data import split_dataset

# Глобальные переменные
MODEL = None
CLASS_NAMES = None
IMAGE_SIZE = 224
LOADED_CHECKPOINT = None  # Путь к загруженной модели
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Кеш HF моделей
HF_CACHE = {}

def load_model_ui(checkpoint_path):
    """Загрузка модели с правильным препроцессингом из чекпоинта."""
    global MODEL, CLASS_NAMES, IMAGE_SIZE, LOADED_CHECKPOINT
    
    # Перезагружаем только если путь изменился
    if LOADED_CHECKPOINT == checkpoint_path and MODEL is not None:
        return f"✅ Модель уже загружена: {checkpoint_path}"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        config = checkpoint['config']
        model_name = config['model']['name']
        CLASS_NAMES = checkpoint['class_names']
        
        # Берем image_size из чекпоинта
        IMAGE_SIZE = checkpoint.get('image_size', config['training']['image_size'])
        
        MODEL = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES))
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()
        LOADED_CHECKPOINT = checkpoint_path
        
        return f"✅ Модель: {model_name}\nКлассы: {CLASS_NAMES}\nРазмер: {IMAGE_SIZE}px"
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

def preprocess_image(image_rgb, image_size=224):
    """Предобработка: Gradio отдает RGB, нормализуем напрямую."""
    # Gradio отдает RGB (не BGR!)
    
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    
    augmented = transform(image=image_rgb)
    return augmented['image'].unsqueeze(0)

def predict_ui(image, checkpoint_path, top_k=3):
    """Предсказание через UI."""
    if image is None:
        return "Загрузите изображение", None
    
    if MODEL is None or LOADED_CHECKPOINT != checkpoint_path:
        result = load_model_ui(checkpoint_path)
        if "❌" in result:
            return result, None
    
    try:
        image_tensor = preprocess_image(image, IMAGE_SIZE).to(DEVICE)
        
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(CLASS_NAMES)))
        
        result = "🎯 Предсказания:\n"
        labels = []
        probs_list = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            result += f"{CLASS_NAMES[idx]}: {prob.item()*100:.2f}%\n"
            labels.append(CLASS_NAMES[idx])
            probs_list.append(prob.item()*100)
        
        fig = go.Figure(data=[go.Bar(x=labels, y=probs_list, marker_color='lightblue')])
        fig.update_layout(title="Вероятности", yaxis_title="%", height=400)
        
        return result, fig
    except Exception as e:
        return f"❌ Ошибка: {str(e)}", None

def predict_batch_ui(folder_path, checkpoint_path):
    """Батч-предсказание."""
    if not folder_path:
        return "Выберите папку"
    
    try:
        if MODEL is None or LOADED_CHECKPOINT != checkpoint_path:
            result = load_model_ui(checkpoint_path)
            if "❌" in result:
                return result
        
        results = []
        folder = Path(folder_path)
        
        for img_path in folder.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # cv2.imread возвращает BGR, preprocess_image ожидает RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = preprocess_image(image_rgb, IMAGE_SIZE).to(DEVICE)
                
                with torch.no_grad():
                    outputs = MODEL(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                
                results.append({
                    'image': img_path.name,
                    'predicted': CLASS_NAMES[predicted.item()],
                    'confidence': probs[0][predicted].item() * 100
                })
        
        return pd.DataFrame(results) if results else "Нет изображений"
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

def analyze_data_ui(data_path):
    """Анализ данных."""
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            analyze_dataset(data_path)
        text_output = f.getvalue()
        
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
    """Разделение данных."""
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            split_dataset(source_path, train_ratio, val_ratio, test_ratio)
        return f.getvalue()
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

def train_ui_async(config_path, epochs, model_name, fast_mode, seed=42):
    """Асинхронное обучение."""
    try:
        import subprocess
        
        cmd = [sys.executable, "train.py", "--config", config_path, "--seed", str(seed)]
        if epochs:
            cmd.extend(["--epochs", str(int(epochs))])
        if model_name:
            cmd.extend(["--model", model_name])
        if fast_mode:
            cmd.append("--fast")
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output_lines = []
        
        while True:
            if process.poll() is not None:
                break
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())
            time.sleep(0.1)
        
        remaining = process.stdout.read()
        if remaining:
            output_lines.extend(remaining.strip().split('\n'))
        
        return "\n".join(output_lines[-100:])
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

def visualize_augmentations_ui(image, num_examples=5):
    """Визуализация аугментаций."""
    if image is None:
        return None
    
    try:
        image_rgb = image  # Gradio уже отдает RGB
        transform = get_transforms(224, is_train=True, augmentation_config={
            'horizontal_flip': True,
            'brightness_contrast': True,
        })
        
        augmented_images = []
        for _ in range(num_examples):
            augmented = transform(image=image_rgb)
            aug_img = augmented['image'].permute(1, 2, 0).numpy()
            aug_img = (aug_img * np.array(STD) + np.array(MEAN))
            aug_img = np.clip(aug_img, 0, 1)
            augmented_images.append(aug_img)
        
        fig = go.Figure()
        for i, img in enumerate(augmented_images):
            fig.add_trace(go.Image(z=img, name=f'Aug {i+1}'))
        fig.update_layout(height=300)
        
        return fig
    except Exception as e:
        return None

# ============ HUGGING FACE (с кешированием) ============

def get_hf_model(model_name, model_type='classification'):
    """Получение HF модели из кеша или загрузка."""
    cache_key = f"{model_type}_{model_name}"
    
    if cache_key in HF_CACHE:
        print(f"📦 Модель {model_name} из кеша")
        return HF_CACHE[cache_key]
    
    print(f"📥 Загрузка модели {model_name}...")
    
    if model_type == 'classification':
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE)
        HF_CACHE[cache_key] = (model, processor)
    elif model_type == 'clip':
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(model_name)
        HF_CACHE[cache_key] = (model, processor)
    elif model_type == 'sam':
        from transformers import SamModel, SamProcessor
        model = SamModel.from_pretrained(model_name).to(DEVICE)
        processor = SamProcessor.from_pretrained(model_name)
        HF_CACHE[cache_key] = (model, processor)
    elif model_type == 'diffusion':
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        HF_CACHE[cache_key] = pipe
        return pipe
    
    return HF_CACHE[cache_key]

def hf_classify(image, model_name):
    """Классификация через HF (с кешированием)."""
    if image is None:
        return "Загрузите изображение", None
    
    try:
        model, processor = get_hf_model(model_name, 'classification')
        image_pil = Image.fromarray(image)  # Gradio уже отдает RGB
        inputs = processor(images=image_pil, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        top_probs, top_indices = torch.topk(probs, 5)
        
        result = "🎯 Топ-5:\n"
        labels = []
        probs_list = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = model.config.id2label[idx.item()]
            result += f"{label}: {prob.item()*100:.2f}%\n"
            labels.append(label)
            probs_list.append(prob.item()*100)
        
        fig = go.Figure(data=[go.Bar(x=labels, y=probs_list, marker_color='lightgreen')])
        fig.update_layout(title="HF Classification", height=400)
        
        return result, fig
    except Exception as e:
        return f"❌ Ошибка: {str(e)}", None

def hf_zero_shot(image, labels_text):
    """Zero-shot через CLIP (с кешированием)."""
    if image is None:
        return "Загрузите изображение", None
    if not labels_text:
        return "Введите метки", None
    
    try:
        model, processor = get_hf_model("openai/clip-vit-base-patch32", 'clip')
        labels = [l.strip() for l in labels_text.split(',')]
        image_pil = Image.fromarray(image)  # Gradio уже отдает RGB
        
        inputs = processor(text=labels, images=image_pil, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        result = "🏷️ Zero-shot:\n"
        labels_list = []
        probs_list = []
        for label, prob in zip(labels, probs[0]):
            result += f"{label}: {prob.item()*100:.2f}%\n"
            labels_list.append(label)
            probs_list.append(prob.item()*100)
        
        fig = go.Figure(data=[go.Bar(x=labels_list, y=probs_list, marker_color='orange')])
        fig.update_layout(title="CLIP Zero-shot", height=400)
        
        return result, fig
    except Exception as e:
        return f"❌ Ошибка: {str(e)}", None

def hf_segment(image):
    """Сегментация через SAM (с кешированием)."""
    if image is None:
        return "Загрузите изображение", None
    
    try:
        model, processor = get_hf_model("facebook/sam-vit-base", 'sam')
        image_pil = Image.fromarray(image)  # Gradio уже отдает RGB
        inputs = processor(image_pil, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        mask = outputs.pred_masks[0, 0].cpu().numpy()
        mask = (mask > 0).astype(np.uint8) * 255
        
        fig = go.Figure(data=go.Heatmap(z=mask, colorscale='gray'))
        fig.update_layout(title="SAM Mask", height=400)
        
        return "✅ Сегментация выполнена", fig
    except Exception as e:
        return f"❌ Ошибка: {str(e)}", None

def hf_generate(prompt):
    """Генерация через Stable Diffusion (с кешированием)."""
    if not prompt:
        return "Введите промпт", None
    
    try:
        pipe = get_hf_model("runwayml/stable-diffusion-v1-5", 'diffusion')
        image = pipe(prompt).images[0]
        
        return f"✅ Сгенерировано: {prompt}", image
    except Exception as e:
        return f"❌ Ошибка: {str(e)}", None

# ============ ИНТЕРФЕЙС ============

with gr.Blocks(title="CV Hackathon Toolkit Pro") as app:
    gr.Markdown("""
    # 🚀 CV Hackathon Toolkit Pro
    Максимально быстрый интерфейс для победы на хакатоне
    """)
    
    with gr.Tab("🔮 Инференс"):
        gr.Markdown("### Предсказание на изображении")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Загрузите изображение")
                checkpoint_input = gr.Textbox(label="Путь к чекпоинту", value="checkpoints/best_model.pth")
                top_k = gr.Slider(1, 5, value=3, step=1, label="Топ-K")
                predict_btn = gr.Button("🎯 Предсказать", variant="primary", size="lg")
                
                gr.Markdown("### Батч-предсказание")
                folder_input = gr.Textbox(label="Путь к папке")
                batch_btn = gr.Button("📁 Обработать", variant="secondary")
            
            with gr.Column(scale=1):
                prediction_output = gr.Textbox(label="Результат", lines=8)
                prediction_plot = gr.Plot(label="Вероятности")
                batch_output = gr.Dataframe(label="Батч-результаты")
        
        predict_btn.click(fn=predict_ui, inputs=[image_input, checkpoint_input, top_k], outputs=[prediction_output, prediction_plot])
        batch_btn.click(fn=predict_batch_ui, inputs=[folder_input, checkpoint_input], outputs=batch_output)
    
    with gr.Tab("📊 Анализ данных"):
        gr.Markdown("### Анализ датасета")
        with gr.Row():
            with gr.Column():
                data_path_input = gr.Textbox(label="Путь к данным", value="data/train")
                analyze_btn = gr.Button("📈 Анализировать", variant="primary", size="lg")
                
                gr.Markdown("### Аугментации")
                aug_image_input = gr.Image(label="Исходное изображение")
                aug_btn = gr.Button("🎨 Показать", variant="secondary")
            
            with gr.Column():
                analysis_output = gr.Textbox(label="Результаты", lines=15)
                class_distribution = gr.Plot(label="Распределение")
                aug_output = gr.Plot(label="Аугментации")
        
        analyze_btn.click(fn=analyze_data_ui, inputs=[data_path_input], outputs=[analysis_output, class_distribution])
        aug_btn.click(fn=visualize_augmentations_ui, inputs=[aug_image_input], outputs=aug_output)
    
    with gr.Tab("🔀 Разделение данных"):
        gr.Markdown("### Разделение на train/val/test")
        source_path_input = gr.Textbox(label="Путь к данным", placeholder="data/all_data")
        with gr.Row():
            train_ratio = gr.Slider(0.5, 0.95, value=0.8, label="Train")
            val_ratio = gr.Slider(0.05, 0.3, value=0.1, label="Val")
            test_ratio = gr.Slider(0.05, 0.3, value=0.1, label="Test")
        split_btn = gr.Button("🔀 Разделить", variant="primary", size="lg")
        split_output = gr.Textbox(label="Результат", lines=10)
        
        split_btn.click(fn=split_data_ui, inputs=[source_path_input, train_ratio, val_ratio, test_ratio], outputs=split_output)
    
    with gr.Tab("🎓 Обучение"):
        gr.Markdown("### Запуск обучения")
        with gr.Row():
            with gr.Column():
                config_path_input = gr.Textbox(label="Конфиг", value="configs/config.yaml")
                epochs_input = gr.Number(label="Эпохи", value=None, precision=0)
                model_input = gr.Dropdown(
                    label="Модель",
                    choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b3", "mobilenetv3_large_100"],
                    value=None
                )
                fast_mode = gr.Checkbox(label="Fast mode", value=False)
                seed_input = gr.Number(label="Seed", value=42, precision=0)
                train_btn = gr.Button("🚀 Обучить", variant="primary", size="lg")
            
            with gr.Column():
                train_output = gr.Textbox(label="Лог", lines=25)
        
        train_btn.click(fn=train_ui_async, inputs=[config_path_input, epochs_input, model_input, fast_mode, seed_input], outputs=train_output)
    
    with gr.Tab("🤗 Hugging Face"):
        gr.Markdown("### HF модели (кешируются в памяти)")
        
        with gr.Tab("Классификация"):
            with gr.Row():
                with gr.Column():
                    hf_classify_image = gr.Image(label="Изображение")
                    hf_model_select = gr.Dropdown(
                        label="Модель",
                        choices=["google/vit-base-patch16-224", "microsoft/swin-tiny-patch4-window7-224", "facebook/convnext-tiny-224"],
                        value="google/vit-base-patch16-224"
                    )
                    hf_classify_btn = gr.Button("🏷️ Классифицировать", variant="primary")
                with gr.Column():
                    hf_classify_output = gr.Textbox(label="Результат", lines=5)
                    hf_classify_plot = gr.Plot(label="Вероятности")
        
        with gr.Tab("Zero-shot (CLIP)"):
            with gr.Row():
                with gr.Column():
                    hf_zs_image = gr.Image(label="Изображение")
                    hf_zs_labels = gr.Textbox(label="Метки", value="cat, dog, car, person")
                    hf_zs_btn = gr.Button("🏷️ Zero-shot", variant="primary")
                with gr.Column():
                    hf_zs_output = gr.Textbox(label="Результат", lines=5)
                    hf_zs_plot = gr.Plot(label="Вероятности")
        
        with gr.Tab("Сегментация (SAM)"):
            with gr.Row():
                with gr.Column():
                    hf_seg_image = gr.Image(label="Изображение")
                    hf_seg_btn = gr.Button("🧩 Сегментировать", variant="primary")
                with gr.Column():
                    hf_seg_output = gr.Textbox(label="Результат", lines=3)
                    hf_seg_plot = gr.Plot(label="Маска")
        
        with gr.Tab("Генерация (SD)"):
            with gr.Row():
                with gr.Column():
                    hf_gen_prompt = gr.Textbox(label="Промпт", placeholder="a cat sitting on a table")
                    hf_gen_btn = gr.Button("🎨 Генерировать", variant="primary")
                with gr.Column():
                    hf_gen_output = gr.Textbox(label="Статус", lines=3)
                    hf_gen_image = gr.Image(label="Результат")
        
        hf_classify_btn.click(fn=hf_classify, inputs=[hf_classify_image, hf_model_select], outputs=[hf_classify_output, hf_classify_plot])
        hf_zs_btn.click(fn=hf_zero_shot, inputs=[hf_zs_image, hf_zs_labels], outputs=[hf_zs_output, hf_zs_plot])
        hf_seg_btn.click(fn=hf_segment, inputs=[hf_seg_image], outputs=[hf_seg_output, hf_seg_plot])
        hf_gen_btn.click(fn=hf_generate, inputs=[hf_gen_prompt], outputs=[hf_gen_output, hf_gen_image])
    
    gr.Markdown("""
    ---
    ### 💡 Советы:
    - HF модели кешируются после первой загрузки
    - Ctrl+Enter — быстрое предсказание
    """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
