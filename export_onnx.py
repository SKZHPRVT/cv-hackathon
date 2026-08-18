"""
Конвертация PyTorch модели в ONNX для быстрого инференса.
Включает проверку ONNX модели через onnxruntime.
"""
import torch
import timm
import argparse
import os
import numpy as np

def export_to_onnx(checkpoint_path, output_path='model.onnx', image_size=224):
    """Конвертация модели в ONNX."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    model_name = config['model']['name']
    class_names = checkpoint['class_names']
    
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
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
    
    return output_path

def test_onnx(onnx_path, image_size=224):
    """Проверка ONNX модели через onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ onnxruntime не установлен")
        print("Установите: pip install onnxruntime")
        return False
    
    # Загружаем модель
    session = ort.InferenceSession(onnx_path)
    
    # Тестовый input
    dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    
    # Инференс
    outputs = session.run(None, {'input': dummy_input})
    
    print(f"✅ ONNX модель работает")
    print(f"📊 Output shape: {outputs[0].shape}")
    print(f"📊 Output (первые 5 значений): {outputs[0][0][:5]}")
    
    # Сравнение скорости
    import time
    
    # PyTorch
    model = timm.create_model('resnet18', pretrained=False, num_classes=2)
    model.eval()
    dummy_tensor = torch.randn(1, 3, image_size, image_size)
    
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            model(dummy_tensor)
    torch_time = (time.time() - start) / 10
    
    # ONNX
    start = time.time()
    for _ in range(10):
        session.run(None, {'input': dummy_input})
    onnx_time = (time.time() - start) / 10
    
    print(f"\n⚡ Сравнение скорости (10 запусков):")
    print(f"  PyTorch: {torch_time*1000:.2f} ms")
    print(f"  ONNX: {onnx_time*1000:.2f} ms")
    print(f"  Ускорение: {torch_time/onnx_time:.2f}x")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--output', default='model.onnx', help='Output path')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--test', action='store_true', help='Test ONNX model after export')
    
    args = parser.parse_args()
    
    onnx_path = export_to_onnx(args.checkpoint, args.output, args.image_size)
    
    if args.test:
        test_onnx(onnx_path, args.image_size)
