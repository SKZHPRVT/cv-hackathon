#!/bin/bash
# Расширенный тест пайплайна (временная папка)
set -e

TMP_DIR=$(mktemp -d)
CHECKPOINT_DIR="$TMP_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
echo "📁 Временная папка: $TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "========================================="
echo "🧪 РАСШИРЕННЫЙ ТЕСТ ПАЙПЛАЙНА (13 шагов)"
echo "========================================="

# 1. check_env
echo -e "\n[1/13] check_env..."
python check_env.py --with-yolo || { echo "❌ check_env провален"; exit 1; }

# 2. Создание данных
echo -e "\n[2/13] Создание данных..."
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
echo -e "\n[3/13] check_data..."
python utils/check_data.py --path "$TMP_DIR/data/all_data"

# 4. find_duplicates exact
echo -e "\n[4/13] find_duplicates (exact)..."
python utils/find_duplicates.py --data "$TMP_DIR/data/all_data" --exact

# 5. find_duplicates near
echo -e "\n[5/13] find_duplicates (near)..."
python utils/find_duplicates.py --data "$TMP_DIR/data/all_data" --near

# 6. split
echo -e "\n[6/13] split..."
python utils/split_data.py --source "$TMP_DIR/data/all_data" --seed 42 --min_val_per_class 1

# 7. leakage check
echo -e "\n[7/13] leakage check..."
python utils/find_duplicates.py --data "$TMP_DIR/data/all_data" --leakage \
    --train "$TMP_DIR/data/train" --val "$TMP_DIR/data/val" --test "$TMP_DIR/data/test"

# 8. train fast
echo -e "\n[8/13] train fast..."
python train.py --fast --seed 42 --exp_name test_pipeline \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --train_path "$TMP_DIR/data/train" \
    --val_path "$TMP_DIR/data/val"

# 9. predict
echo -e "\n[9/13] predict..."
VAL_IMAGE=$(ls "$TMP_DIR/data/val/cat/"*.jpg 2>/dev/null | head -1)
python predict.py --checkpoint "$CHECKPOINT_DIR/best_model.pth" --image "$VAL_IMAGE"

# 10. TTA
echo -e "\n[10/13] TTA..."
python tta_predict.py --checkpoint "$CHECKPOINT_DIR/best_model.pth" --image "$VAL_IMAGE" --tta hflip

# 11. ONNX
echo -e "\n[11/13] ONNX..."
python export_onnx.py --checkpoint "$CHECKPOINT_DIR/best_model.pth" --test

# 12. YOLO test
echo -e "\n[12/13] YOLO test..."
if [ -f "yolov8n.pt" ]; then
    echo "✅ yolov8n.pt найден"
else
    echo "📥 Скачиваю yolov8n.pt..."
    python -c 'from ultralytics import YOLO; YOLO("yolov8n.pt")'
fi
python yolo_train.py --mode predict --model yolov8n.pt --source "$TMP_DIR/data/val/cat/" --conf 0.25
echo "✅ YOLO predict выполнен"

# 13. submission
echo -e "\n[13/13] submission..."
python generate_submission.py --checkpoint "$CHECKPOINT_DIR/best_model.pth" --test_folder "$TMP_DIR/data/test" --format label

echo -e "\n========================================="
echo "🎉 ВСЕ 13 ТЕСТОВ ПРОЙДЕНЫ!"
echo "========================================="
