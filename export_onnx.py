"""
Конвертация PyTorch модели в ONNX для быстрого инференса.
"""
import torch
import timm
import argparse
import os

def export_to_onnx(checkpoint_path, output_path='model.onnx', image_size=224):
    """Конвертация модели в ONNX."""
    # Загружаем чекпоинт
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    model_name = config['model']['name']
    class_names = checkpoint['class_names']
    
    # Создаем модель
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Создаем dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Экспортируем
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )
    
    print(f"✅ Модель экспортирована в {output_path}")
    print(f"📊 Классы: {class_names}")
    print(f"📐 Input size: {image_size}x{image_size}")

def test_onnx(onnx_path, image_size=224):
    """Проверка ONNX модели."""
    import onnxruntime as ort
    import numpy as np
    
    # Загружаем модель
    session = ort.InferenceSession(onnx_path)
    
    # Тестовый input
    dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    
    # Инференс
    outputs = session.run(None, {'input': dummy_input})
    
    print(f"✅ ONNX модель работает")
    print(f"📊 Output shape: {outputs[0].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--output', default='model.onnx', help='Output path')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--test', action='store_true', help='Test ONNX model after export')
    
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint, args.output, args.image_size)
    
    if args.test:
        try:
            import onnxruntime
            test_onnx(args.output, args.image_size)
        except ImportError:
            print("⚠️ Установите onnxruntime: pip install onnxruntime")
