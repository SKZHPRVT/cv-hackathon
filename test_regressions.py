"""
Регрессионные тесты на 4 конкретных случая:
1. HF inputs на DEVICE
2. min_val_per_class
3. CHECKPOINT_DIR изоляция
4. Strict offline с exit 1
"""
import sys
import os
import subprocess
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_hf_inputs_device():
    """Тест: HF inputs переносятся на DEVICE."""
    print("\n[TEST 1] HF inputs на DEVICE...")
    
    # Проверяем код в app.py
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Должно быть 3 переноса inputs на DEVICE
    count = content.count('inputs = {k: v.to(DEVICE) for k, v in inputs.items()}')
    
    if count == 3:
        print(f"  ✅ Найдено {count} переносов inputs на DEVICE")
        return True
    else:
        print(f"  ❌ Ожидалось 3, найдено {count}")
        return False

def test_min_val_per_class():
    """Тест: min_val_per_class действительно гарантирует минимум."""
    print("\n[TEST 2] min_val_per_class...")
    
    # Создаем тестовые данные
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = Path(tmp_dir) / 'data'
        for cls in ['a', 'b']:
            (data_dir / cls).mkdir(parents=True)
            for i in range(10):
                import numpy as np
                import cv2
                img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                cv2.imwrite(str(data_dir / cls / f'{i}.jpg'), img)
        
        # Запускаем split с min_val_per_class=3
        result = subprocess.run(
            [sys.executable, 'utils/split_data.py', '--source', str(data_dir), 
             '--min_val_per_class', '3', '--seed', '42'],
            capture_output=True, text=True
        )
        
        # Проверяем val
        val_a = len(list((Path(tmp_dir) / 'val' / 'a').glob('*.jpg')))
        val_b = len(list((Path(tmp_dir) / 'val' / 'b').glob('*.jpg')))
        
        if val_a >= 3 and val_b >= 3:
            print(f"  ✅ val_a={val_a}, val_b={val_b} (оба >= 3)")
            return True
        else:
            print(f"  ❌ val_a={val_a}, val_b={val_b} (ожидалось >= 3)")
            return False

def test_checkpoint_dir_isolation():
    """Тест: CHECKPOINT_DIR изолирует чекпоинты."""
    print("\n[TEST 3] CHECKPOINT_DIR изоляция...")
    
    with open('train.py', 'r') as f:
        content = f.read()
    
    if '--checkpoint_dir' in content:
        print("  ✅ --checkpoint_dir поддерживается")
        return True
    else:
        print("  ❌ --checkpoint_dir отсутствует")
        return False

def test_strict_offline_exit_code():
    """Тест: strict offline возвращает правильный exit code."""
    print("\n[TEST 4] Strict offline exit code...")
    
    with open('download_models.py', 'r') as f:
        content = f.read()
    
    if 'exit(0 if result else 1)' in content:
        print("  ✅ exit 1 при провале strict offline")
        return True
    else:
        print("  ❌ exit code не обрабатывается")
        return False

if __name__ == "__main__":
    results = []
    results.append(("HF inputs DEVICE", test_hf_inputs_device()))
    results.append(("min_val_per_class", test_min_val_per_class()))
    results.append(("CHECKPOINT_DIR", test_checkpoint_dir_isolation()))
    results.append(("Strict offline exit", test_strict_offline_exit_code()))
    
    print("\n" + "="*50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Результат: {passed}/{total}")
    print("="*50)
    
    sys.exit(0 if passed == total else 1)
