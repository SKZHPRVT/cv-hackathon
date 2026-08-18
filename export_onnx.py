"""
Конвертация в ONNX с проверкой эквивалентности PyTorch vs ONNX.
"""
import torch
import timm
import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils.dataset import MEAN, STD

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
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✅ Экспортировано: {output_path}")
    return model, output_path

def test_equivalence(pytorch_model, onnx_path, image_size=224, tolerance=1e-4):
    """Проверка эквивалентности PyTorch и ONNX моделей."""
    import onnxruntime as ort
    
    session = ort.InferenceSession(onnx_path)
    
    # Тестовые данные (с batch dimension)
    np.random.seed(42)
    test_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    
    # PyTorch вывод
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input)
        torch_output = pytorch_model(torch_input).numpy()
    
    # ONNX вывод
    onnx_output = session.run(None, {'input': test_input})[0]
    
    # Сравнение
    diff = np.abs(torch_output - onnx_output)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"\n🔬 Проверка эквивалентности:")
    print(f"  PyTorch output: {torch_output[0][:5]}")
    print(f"  ONNX output:   {onnx_output[0][:5]}")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < tolerance:
        print(f"✅ Модели эквивалентны (max diff < {tolerance})")
        return True
    else:
        print(f"⚠️ Модели различаются (max diff = {max_diff:.6f} > {tolerance})")
        return False

def compare_speed(onnx_path, pytorch_model, image_size=224, num_runs=20):
    """Сравнение скорости PyTorch vs ONNX."""
    import onnxruntime as ort
    import time
    
    session = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    
    # PyTorch
    torch_input = torch.from_numpy(test_input)
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            pytorch_model(torch_input)
    torch_time = (time.time() - start) / num_runs
    
    # ONNX
    start = time.time()
    for _ in range(num_runs):
        session.run(None, {'input': test_input})
    onnx_time = (time.time() - start) / num_runs
    
    print(f"\n⚡ Скорость ({num_runs} запусков):")
    print(f"  PyTorch: {torch_time*1000:.2f} ms")
    print(f"  ONNX: {onnx_time*1000:.2f} ms")
    print(f"  Ускорение: {torch_time/onnx_time:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export to ONNX')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='model.onnx')
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--test', action='store_true', help='Проверить эквивалентность')
    parser.add_argument('--speed', action='store_true', help='Сравнить скорость')
    
    args = parser.parse_args()
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    image_size = args.image_size or checkpoint.get('image_size', 224)
    
    model, onnx_path = export_to_onnx(args.checkpoint, args.output, image_size)
    
    if args.test:
        test_equivalence(model, onnx_path, image_size)
    
    if args.speed:
        compare_speed(onnx_path, model, image_size)
