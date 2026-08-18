"""
Поддержка Hugging Face моделей для CV-задач.
Включает:
- Классификация (ViT, Swin, ConvNeXT)
- Zero-shot классификация (CLIP)
- Сегментация (SAM, DETR)
- Генерация (Stable Diffusion)

Режимы:
1. classify - классификация изображений
2. zero_shot - zero-shot классификация (CLIP)
3. segment - сегментация (SAM)
4. generate - генерация изображений
"""
import torch
import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def check_install():
    """Проверка установки библиотек"""
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers не установлен")
        print("Установите: pip install transformers")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
    except ImportError:
        print("⚠️ Datasets не установлен (необязательно)")
    
    return True

def classify_image(image_path, model_name='google/vit-base-patch16-224'):
    """Классификация изображения через Hugging Face"""
    try:
        from transformers import ViTImageProcessor, ViTForImageClassification
    except ImportError:
        print("❌ Установите: pip install transformers")
        return
    
    print(f"📦 Загрузка модели {model_name}...")
    
    # Загружаем модель
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    
    # Загружаем изображение
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    # Инференс
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
    
    # Топ-5 предсказаний
    top_probs, top_indices = torch.topk(probs, 5)
    
    print(f"\n📸 {image_path}")
    print("Топ-5 предсказаний:")
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), 1):
        label = model.config.id2label[idx.item()]
        print(f"  {i}. {label}: {prob.item()*100:.2f}%")

def zero_shot_classify(image_path, labels=['cat', 'dog', 'car', 'person']):
    """Zero-shot классификация через CLIP"""
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        print("❌ Установите: pip install transformers")
        return
    
    print("📦 Загрузка CLIP модели...")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Загружаем изображение
    image = Image.open(image_path)
    
    # Подготавливаем inputs
    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Инференс
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    print(f"\n📸 {image_path}")
    print("Zero-shot классификация:")
    for label, prob in zip(labels, probs[0]):
        print(f"  {label}: {prob.item()*100:.2f}%")

def segment_image(image_path, model_name='facebook/sam-vit-base'):
    """Сегментация через SAM"""
    try:
        from transformers import SamModel, SamProcessor
    except ImportError:
        print("❌ Установите: pip install transformers")
        return
    
    print(f"📦 Загрузка SAM модели {model_name}...")
    
    model = SamModel.from_pretrained(model_name)
    processor = SamProcessor.from_pretrained(model_name)
    
    # Загружаем изображение
    image = Image.open(image_path)
    
    # Подготавливаем inputs
    inputs = processor(image, return_tensors="pt")
    
    # Инференс
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"\n📸 {image_path}")
    print(f"✅ Сегментация выполнена")
    print(f"📊 Размер масок: {outputs.pred_masks.shape}")
    
    # Сохраняем маску
    mask = outputs.pred_masks[0, 0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    
    output_path = 'segmentation_mask.png'
    cv2.imwrite(output_path, mask)
    print(f"💾 Маска сохранена в {output_path}")

def generate_image(prompt='a cat sitting on a table', output='generated.png'):
    """Генерация изображения через Stable Diffusion"""
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        print("❌ Установите: pip install diffusers")
        return
    
    print(f"📦 Загрузка Stable Diffusion...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    print(f"🎨 Генерация изображения: '{prompt}'")
    
    image = pipe(prompt).images[0]
    image.save(output)
    
    print(f"✅ Изображение сохранено в {output}")

def list_available_models():
    """Список популярных HF моделей для CV"""
    models = {
        'Классификация': [
            'google/vit-base-patch16-224',
            'microsoft/swin-tiny-patch4-window7-224',
            'facebook/convnext-tiny-224',
            'microsoft/resnet-50',
        ],
        'Zero-shot (CLIP)': [
            'openai/clip-vit-base-patch32',
            'openai/clip-vit-large-patch14',
        ],
        'Сегментация': [
            'facebook/sam-vit-base',
            'facebook/sam-vit-large',
            'facebook/detr-resnet-50',
        ],
        'Генерация': [
            'runwayml/stable-diffusion-v1-5',
            'stabilityai/stable-diffusion-2-1',
        ],
    }
    
    print("\n📚 Доступные HF модели для CV:")
    print("="*60)
    for category, models_list in models.items():
        print(f"\n{category}:")
        for model in models_list:
            print(f"  - {model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hugging Face модели для CV')
    parser.add_argument('--mode', choices=['classify', 'zero_shot', 'segment', 'generate', 'list'],
                       required=True, help='Режим работы')
    parser.add_argument('--image', help='Путь к изображению')
    parser.add_argument('--model', help='Название модели')
    parser.add_argument('--labels', nargs='+', help='Метки для zero-shot')
    parser.add_argument('--prompt', help='Промпт для генерации')
    parser.add_argument('--output', default='output.png', help='Выходной файл')
    
    args = parser.parse_args()
    
    if args.mode == 'list':
        list_available_models()
    elif args.mode == 'classify':
        classify_image(args.image, args.model or 'google/vit-base-patch16-224')
    elif args.mode == 'zero_shot':
        labels = args.labels or ['cat', 'dog', 'car', 'person']
        zero_shot_classify(args.image, labels)
    elif args.mode == 'segment':
        segment_image(args.image, args.model or 'facebook/sam-vit-base')
    elif args.mode == 'generate':
        generate_image(args.prompt, args.output)
