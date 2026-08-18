"""
Поиск похожих объектов на основе эмбеддингов.
Использует предобученные модели для извлечения признаков.

Режимы:
1. build - создать базу эмбеддингов из папки с изображениями
2. search - найти похожие изображения по запросу
3. compare - сравнить два изображения
"""
import torch
import timm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import pickle
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import shutil

class EmbeddingExtractor:
    """Извлечение эмбеддингов из изображений"""
    
    def __init__(self, model_name='resnet18', device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Создаем модель без классификационного слоя
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Определяем размер эмбеддинга
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(self.device)
            self.embedding_size = self.model(dummy).shape[1]
        
        print(f"✅ Модель {model_name} загружена")
        print(f"📐 Размер эмбеддинга: {self.embedding_size}")
    
    def extract(self, image_path):
        """Извлечение эмбеддинга из изображения"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        return embedding.cpu().numpy().flatten()
    
    def extract_batch(self, image_paths):
        """Извлечение эмбеддингов из списка изображений"""
        embeddings = []
        valid_paths = []
        
        for img_path in image_paths:
            embedding = self.extract(img_path)
            if embedding is not None:
                embeddings.append(embedding)
                valid_paths.append(str(img_path))
        
        return np.array(embeddings), valid_paths


def build_database(image_folder, output_file='embeddings.pkl', model_name='resnet18'):
    """Создание базы эмбеддингов из папки с изображениями"""
    extractor = EmbeddingExtractor(model_name)
    
    image_folder = Path(image_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Собираем все изображения
    image_paths = [f for f in image_folder.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_paths:
        print(f"❌ В папке {image_folder} нет изображений")
        return
    
    print(f"📁 Найдено {len(image_paths)} изображений")
    
    # Извлекаем эмбеддинги
    embeddings, valid_paths = extractor.extract_batch(image_paths)
    
    # Сохраняем
    database = {
        'embeddings': embeddings,
        'paths': valid_paths,
        'model_name': model_name
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"✅ База эмбеддингов сохранена в {output_file}")
    print(f"📊 Изображений в базе: {len(valid_paths)}")
    print(f"📐 Размер базы: {Path(output_file).stat().st_size / 1e6:.1f} MB")


def search_similar(query_image, database_file='embeddings.pkl', top_k=5, model_name='resnet18'):
    """Поиск похожих изображений"""
    extractor = EmbeddingExtractor(model_name)
    
    # Загружаем базу
    with open(database_file, 'rb') as f:
        database = pickle.load(f)
    
    # Извлекаем эмбеддинг запроса
    query_embedding = extractor.extract(query_image)
    
    if query_embedding is None:
        print(f"❌ Не удалось загрузить {query_image}")
        return
    
    # Считаем схожесть
    similarities = cosine_similarity([query_embedding], database['embeddings'])[0]
    
    # Топ-K похожих
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    print(f"\n🔍 Результаты поиска для: {query_image}")
    print("="*60)
    
    results = []
    for i, idx in enumerate(top_indices, 1):
        similarity = similarities[idx] * 100
        path = database['paths'][idx]
        results.append({
            'rank': i,
            'path': path,
            'similarity': similarity
        })
        print(f"{i}. {path} ({similarity:.2f}%)")
    
    return results


def compare_images(image1, image2, model_name='resnet18'):
    """Сравнение двух изображений"""
    extractor = EmbeddingExtractor(model_name)
    
    emb1 = extractor.extract(image1)
    emb2 = extractor.extract(image2)
    
    if emb1 is None or emb2 is None:
        print("❌ Ошибка загрузки изображений")
        return
    
    similarity = cosine_similarity([emb1], [emb2])[0][0] * 100
    
    print(f"\n🖼️ Сравнение изображений:")
    print(f"  Image 1: {image1}")
    print(f"  Image 2: {image2}")
    print(f"  Схожесть: {similarity:.2f}%")
    
    if similarity > 90:
        print("  ✅ Очень похожи (возможно, одно и то же)")
    elif similarity > 70:
        print("  🔶 Похожи")
    elif similarity > 50:
        print("  🔸 Немного похожи")
    else:
        print("  ❌ Не похожи")


def visualize_results(query_image, results, output_folder='search_results'):
    """Визуализация результатов поиска"""
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # Копируем query
    shutil.copy(query_image, output_folder / 'query.jpg')
    
    # Копируем результаты
    for result in results:
        src = Path(result['path'])
        dst = output_folder / f"rank_{result['rank']}_{src.name}"
        shutil.copy(src, dst)
    
    print(f"\n📁 Результаты сохранены в {output_folder}")
    print("Можно открыть папку и посмотреть найденные изображения")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Поиск похожих объектов по эмбеддингам')
    parser.add_argument('--mode', choices=['build', 'search', 'compare'], required=True,
                       help='Режим: build (создать базу), search (поиск), compare (сравнение)')
    parser.add_argument('--input', help='Папка с изображениями (для build) или изображение (для search/compare)')
    parser.add_argument('--database', default='embeddings.pkl', help='Файл базы эмбеддингов')
    parser.add_argument('--query', help='Изображение для поиска')
    parser.add_argument('--image2', help='Второе изображение для сравнения')
    parser.add_argument('--model', default='resnet18', help='Модель для эмбеддингов')
    parser.add_argument('--top_k', type=int, default=5, help='Количество результатов')
    parser.add_argument('--save_results', action='store_true', help='Сохранить результаты поиска')
    
    args = parser.parse_args()
    
    if args.mode == 'build':
        build_database(args.input, args.database, args.model)
    elif args.mode == 'search':
        results = search_similar(args.query, args.database, args.top_k, args.model)
        if args.save_results and results:
            visualize_results(args.query, results)
    elif args.mode == 'compare':
        compare_images(args.input, args.image2, args.model)
