"""
Поддержка Hugging Face моделей для CV-задач.
Включает: классификацию, zero-shot, сегментацию, генерацию.
Использует datasets и accelerate для оптимизации.
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
        return False
    
    try:
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
    except ImportError:
        print("⚠️ Datasets не установлен")
    
    try:
        import accelerate
        print(f"✅ Accelerate {accelerate.__version__}")
    except ImportError:
        print("⚠️ Accelerate не установлен")
    
    try:
        import safetensors
        print(f"✅ Safetensors {safetensors.__version__}")
    except ImportError:
        print("⚠️ Safetensors не установлен")
    
    return True

def load_dataset_info(dataset_name):
    """Загрузка информации о датасете через datasets"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Установите: pip install datasets")
        return
    
    print(f"📦 Загрузка датасета {dataset_name}...")
    
    dataset = load_dataset(dataset_name, split='train')
    
    print(f"✅ Датасет загружен")
    print(f"📊 Размер: {len(dataset)} записей")
    print(f"📋 Колонки: {dataset.column_names}")
    
    # Показываем пример
    print(f"\n📝 Пример:")
    example = dataset[0]
    for key, value in example.items():
        if isinstance(value, (str, int, float)):
            print(f"  {key}: {value}")
        elif isinstance(value, Image.Image):
            print(f"  {key}: <Image {value.size}>")

def classify_image(image_path, model_name='google/vit-base-patch16-224'):
    """Классификация изображения через Hugging Face"""
    from transformers import ViTImageProcessor, ViTForImageClassification
    
    print(f"📦 Загрузка модели {model_name}...")
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
    
    top_probs, top_indices = torch.topk(probs, 5)
    
    print(f"\n📸 {image_path}")
    print("Топ-5 предсказаний:")
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), 1):
        label = model.config.id2label[idx.item()]
        print(f"  {i}. {label}: {prob.item()*100:.2f}%")

def zero_shot_classify(image_path, labels=['cat', 'dog', 'car', 'person']):
    """Zero-shot классификация через CLIP"""
    from transformers import CLIPProcessor, CLIPModel
    
    print("📦 Загрузка CLIP модели...")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    image = Image.open(image_path)
    
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    
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
    from transformers import SamModel, SamProcessor
    
    print(f"📦 Загрузка SAM модели {model_name}...")
    
    model = SamModel.from_pretrained(model_name)
    processor = SamProcessor.from_pretrained(model_name)
    
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"\n📸 {image_path}")
    print(f"✅ Сегментация выполнена")
    print(f"📊 Размер масок: {outputs.pred_masks.shape}")
    
    mask = outputs.pred_masks[0, 0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    
    output_path = 'segmentation_mask.png'
    cv2.imwrite(output_path, mask)
    print(f"💾 Маска сохранена в {output_path}")

def generate_image(prompt='a cat sitting on a table', output='generated.png'):
    """Генерация изображения через Stable Diffusion"""
    from diffusers import StableDiffusionPipeline
    
    print(f"📦 Загрузка Stable Diffusion...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    print(f"🎨 Генерация изображения: '{prompt}'")
    
    image = pipe(prompt).images[0]
    image.save(output)
    
    print(f"✅ Изображение сохранено в {output}")

def list_available_models():
    """Список популярных HF моделей"""
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
    
    print("\n📚 Доступные HF модели:")
    print("="*60)
    for category, models_list in models.items():
        print(f"\n{category}:")
        for model in models_list:
            print(f"  - {model}")
    
    print("\n📦 Популярные датасеты:")
    print("  - cifar10")
    print("  - mnist")
    print("  - fashion_mnist")
    print("  - imagenet-1k")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hugging Face модели для CV')
    parser.add_argument('--mode', choices=['classify', 'zero_shot', 'segment', 'generate', 'list', 'dataset', 'check'],
                       required=True, help='Режим работы')
    parser.add_argument('--image', help='Путь к изображению')
    parser.add_argument('--model', help='Название модели')
    parser.add_argument('--labels', nargs='+', help='Метки для zero-shot')
    parser.add_argument('--prompt', help='Промпт для генерации')
    parser.add_argument('--output', default='output.png', help='Выходной файл')
    parser.add_argument('--dataset', help='Название датасета')
    
    args = parser.parse_args()
    
    if args.mode == 'check':
        check_install()
    elif args.mode == 'list':
        list_available_models()
    elif args.mode == 'dataset':
        load_dataset_info(args.dataset)
    elif args.mode == 'classify':
        classify_image(args.image, args.model or 'google/vit-base-patch16-224')
    elif args.mode == 'zero_shot':
        labels = args.labels or ['cat', 'dog', 'car', 'person']
        zero_shot_classify(args.image, labels)
    elif args.mode == 'segment':
        segment_image(args.image, args.model or 'facebook/sam-vit-base')
    elif args.mode == 'generate':
        generate_image(args.prompt, args.output)
