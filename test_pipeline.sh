#!/bin/bash
# Расширенный тест пайплайна (временная папка)
set -e

TMP_DIR=$(mktemp -d)
echo "📁 Временная папка: $TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "========================================="
echo "🧪 РАСШИРЕННЫЙ ТЕСТ ПАЙПЛАЙНА"
echo "========================================="

# 1. check_env (без YOLO — авто)
echo -e "\n[1/12] check_env..."
python check_env.py < /dev/null || { echo "❌ check_env провален"; exit 1; }

# 2. Создание данных
echo -e "\n[2/12] Создание данных..."
mkdir -p "$TMP_DIR/data/all_data/cat" "$TMP_DIR/data/all_data/dog"
python -c "
import numpy as np, cv2
for cls in ['cat', 'dog']:
    for i in range(8):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(f'$TMP_DIR/data/all_data/{cls}/{i}.jpg', img)
print('✅ 16 изображений создано')
"

# 3. check_data
echo -e "\n[3/12] check_data..."
python utils/check_data.py --path "$TMP_DIR/data/all_data"

# 4. find_duplicates exact
echo -e "\n[4/12] find_duplicates (exact)..."
python utils/find_duplicates.py --data "$TMP_DIR/data/all_data" --exact

# 5. find_duplicates near
echo -e "\n[5/12] find_duplicates (near)..."
python utils/find_duplicates.py --data "$TMP_DIR/data/all_data" --near

# 6. split
echo -e "\n[6/12] split..."
python utils/split_data.py --source "$TMP_DIR/data/all_data" --seed 42 --min_val_per_class 1

# 7. leakage check
echo -e "\n[7/12] leakage check..."
python utils/find_duplicates.py --data "$TMP_DIR/data/all_data" --leakage \
    --train "$TMP_DIR/data/train" --val "$TMP_DIR/data/val" --test "$TMP_DIR/data/test"

# 8. train fast (с временными путями)
echo -e "\n[8/12] train fast..."
python train.py --fast --seed 42 --exp_name test_pipeline \
    --train_path "$TMP_DIR/data/train" \
    --val_path "$TMP_DIR/data/val"

# 9. predict
echo -e "\n[9/12] predict..."
python predict.py --checkpoint checkpoints/best_model.pth --image "$TMP_DIR/data/val/cat/0.jpg"

# 10. TTA
echo -e "\n[10/12] TTA..."
python tta_predict.py --checkpoint checkpoints/best_model.pth --image "$TMP_DIR/data/val/cat/0.jpg" --tta hflip

# 11. ONNX
echo -e "\n[11/12] ONNX..."
python export_onnx.py --checkpoint checkpoints/best_model.pth --test

# 12. submission
echo -e "\n[12/12] submission..."
python generate_submission.py --checkpoint checkpoints/best_model.pth --test_folder "$TMP_DIR/data/test" --format label

echo -e "\n========================================="
echo "🎉 ВСЕ 12 ТЕСТОВ ПРОЙДЕНЫ!"
echo "========================================="
