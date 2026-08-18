"""
Обучение классификационной модели с:
- Reproducibility (seed)
- Гарантированным сохранением чекпоинта
- Журналом экспериментов
- Единым препроцессингом
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import timm
import yaml
import os
import sys
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import create_dataloaders, MEAN, STD, DEFAULT_IMAGE_SIZE
from utils.metrics import calculate_metrics, plot_confusion_matrix, plot_training_history

def set_seed(seed=42):
    """Устанавливает seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='CV Hackathon Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--exp_name', type=str, default=None, help='Имя эксперимента')
    parser.add_argument('--train_path', type=str, default=None, help='Путь к train (переопределяет конфиг)')
    parser.add_argument('--val_path', type=str, default=None, help='Путь к val (переопределяет конфиг)')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if config['training']['mixed_precision'] and device.type == 'cuda':
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    metrics = calculate_metrics(all_targets, all_preds, class_names)
    return running_loss/len(val_loader), metrics['accuracy'], all_preds, all_targets

def save_checkpoint(model, optimizer, epoch, val_acc, class_names, config, 
                    image_size, mean, std, save_path, is_best=False):
    """Сохраняет чекпоинт с полной информацией о препроцессинге."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'class_names': class_names,
        'config': config,
        'image_size': image_size,
        'mean': mean,
        'std': std,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, save_path)
    if is_best:
        print(f"💾 Saved best model with val_acc: {val_acc:.2f}%")

def log_experiment(exp_name, config, model_name, seed, metrics_history, best_val_acc, save_dir):
    """Сохраняет журнал эксперимента."""
    log = {
        'experiment_name': exp_name,
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'seed': seed,
        'config': config,
        'metrics_history': metrics_history,
        'best_val_acc': best_val_acc
    }
    
    log_path = f"{save_dir}/experiment_{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"📝 Журнал эксперимента сохранен: {log_path}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    config = load_config(args.config)
    
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.fast:
        config['training']['epochs'] = 1
        config['training']['batch_size'] = 8
        print("🚀 Fast mode: 1 epoch, batch_size=8")
    
    if args.train_path:
        config['data']['train_path'] = args.train_path
    if args.val_path:
        config['data']['val_path'] = args.val_path
    
    exp_name = args.exp_name or f"{config['model']['name']}_seed{args.seed}"
    image_size = config['training']['image_size']
    
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    print(f"🌱 Seed: {args.seed}")
    print(f"📐 Image size: {image_size}")
    print(f"📊 Mean: {MEAN}, Std: {STD}")
    
    print("📦 Loading data...")
    augmentation_config = config.get('augmentation', {})
    train_loader, val_loader, class_names, class_to_idx = create_dataloaders(
        config['data']['train_path'],
        config['data']['val_path'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        image_size=image_size,
        augmentation_config=augmentation_config,
        train_csv=config['data'].get('train_csv'),
        val_csv=config['data'].get('val_csv')
    )
    print(f"✅ Train: {len(train_loader.dataset)} images")
    print(f"✅ Val: {len(val_loader.dataset)} images")
    print(f"📋 Classes ({len(class_names)}): {class_names}")
    
    print(f"🏗️ Model: {config['model']['name']}")
    model = timm.create_model(
        config['model']['name'],
        pretrained=config['model']['pretrained'],
        num_classes=len(class_names)
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    scaler = GradScaler() if config['training']['mixed_precision'] and device.type == 'cuda' else None
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = -1.0  # Начинаем с -1, чтобы сохранить даже при val_acc=0
    best_model_path = f"{config['logging']['save_dir']}/best_model.pth"
    last_model_path = f"{config['logging']['save_dir']}/last_model.pth"
    early_stop_counter = 0
    
    print("\n🚀 Training...")
    for epoch in range(config['training']['epochs']):
        print(f"\n📊 Epoch {epoch+1}/{config['training']['epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, config)
        val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device, class_names)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"📉 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Сохраняем всегда (гарантия)
        save_checkpoint(model, optimizer, epoch, val_acc, class_names, config,
                       image_size, MEAN, STD, last_model_path)
        
        # Сохраняем best (при >= лучшего)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, class_names, config,
                          image_size, MEAN, STD, best_model_path, is_best=True)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= config['training']['early_stopping']:
            print(f"⏹️ Early stopping after {config['training']['early_stopping']} epochs")
            break
    
    print("\n📊 Generating plots...")
    plot_training_history(history, config['logging']['save_dir'])
    
    print("\n🎯 Final evaluation...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device, class_names)
    
    plot_confusion_matrix(val_targets, val_preds, class_names, 
                         f"{config['logging']['save_dir']}/confusion_matrix.png")
    
    final_metrics = calculate_metrics(val_targets, val_preds, class_names)
    print("\n" + "="*50)
    print("🎉 FINAL RESULTS:")
    print("="*50)
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  best_val_acc: {best_val_acc:.2f}%")
    print(f"  model: {best_model_path}")
    print("="*50)
    
    # Журнал эксперимента
    log_experiment(exp_name, config, config['model']['name'], args.seed, 
                  history, best_val_acc, config['logging']['save_dir'])

if __name__ == "__main__":
    main()
