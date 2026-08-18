"""
Метрики и визуализации.
Число классов берется из class_names, а не из валидационной выборки.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def calculate_metrics(y_true, y_pred, class_names=None):
    """Расчет метрик. class_names обязателен для корректного числа классов."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if class_names is not None:
        num_classes = len(class_names)
    else:
        num_classes = len(np.unique(y_true))
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0, labels=range(num_classes)),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=range(num_classes)),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0, labels=range(num_classes)),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0, labels=range(num_classes))
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Визуализация confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_history(history, save_dir='checkpoints'):
    """Графики обучения."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history_{timestamp}.png')
    plt.close()

def print_metrics(metrics, title="Metrics"):
    """Красивый вывод метрик."""
    print(f"\n{'='*50}")
    print(f"📊 {title}")
    print('='*50)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print('='*50)
