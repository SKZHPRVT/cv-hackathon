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

# Добавляем utils в path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import create_dataloaders
from utils.metrics import calculate_metrics, plot_confusion_matrix, plot_training_history

def parse_args():
    parser = argparse.ArgumentParser(description='CV Hackathon Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default=None, help='Model name (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--fast', action='store_true', help='Quick test run (1 epoch, small batches)')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, config):
    """Одна эпоха обучения"""
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
        
        # Обновляем прогресс бар
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    """Валидация"""
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
    
    metrics = calculate_metrics(all_targets, all_preds, len(set(all_targets)))
    return running_loss/len(val_loader), metrics['accuracy'], all_preds, all_targets

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command line arguments
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
    
    # Создаем директорию для чекпоинтов
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Data
    print("📦 Loading data...")
    train_loader, val_loader, class_names = create_dataloaders(
        config['data']['train_path'],
        config['data']['val_path'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        image_size=config['training']['image_size']
    )
    print(f"✅ Loaded {len(train_loader.dataset)} train images, {len(val_loader.dataset)} val images")
    print(f"📋 Classes: {class_names}")
    
    # Model
    print(f"🏗️ Creating model: {config['model']['name']}")
    model = timm.create_model(
        config['model']['name'],
        pretrained=config['model']['pretrained'],
        num_classes=len(class_names)
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Mixed precision
    scaler = GradScaler() if config['training']['mixed_precision'] and device.type == 'cuda' else None
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    best_model_path = None
    early_stop_counter = 0
    
    print("\n🚀 Starting training...")
    for epoch in range(config['training']['epochs']):
        print(f"\n📊 Epoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, config)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        print(f"📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"📉 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': class_names,
            'config': config
        }
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = f"{config['logging']['save_dir']}/best_model.pth"
            torch.save(checkpoint, best_model_path)
            print(f"💾 Saved best model with val_acc: {val_acc:.2f}%")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Save latest model
        torch.save(checkpoint, f"{config['logging']['save_dir']}/last_model.pth")
        
        # Early stopping
        if early_stop_counter >= config['training']['early_stopping']:
            print(f"⏹️ Early stopping after {config['training']['early_stopping']} epochs without improvement")
            break
    
    # Plot results
    print("\n📊 Generating plots...")
    plot_training_history(history, config['logging']['save_dir'])
    
    # Final evaluation on best model
    print("\n🎯 Loading best model for final evaluation...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device)
    
    # Plot confusion matrix
    plot_confusion_matrix(val_targets, val_preds, class_names, 
                         f"{config['logging']['save_dir']}/confusion_matrix.png")
    
    # Final metrics
    final_metrics = calculate_metrics(val_targets, val_preds, len(class_names))
    print("\n" + "="*50)
    print("🎉 FINAL RESULTS:")
    print("="*50)
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  best_val_acc: {best_val_acc:.2f}%")
    print(f"  model saved: {best_model_path}")
    print("="*50)

if __name__ == "__main__":
    main()
