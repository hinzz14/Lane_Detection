import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from lane_detector import LaneDataset, ENet, compute_loss

def calculate_metrics(binary_logits, binary_labels):

    preds = torch.argmax(binary_logits, dim=1)
    
    correct_pixels = (preds == binary_labels).sum().item()
    total_pixels = binary_labels.numel()
    accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0

    intersection = ((preds == 1) & (binary_labels == 1)).sum().item()
    union = ((preds == 1) | (binary_labels == 1)).sum().item()
    iou = intersection / union if union > 0 else 0.0
    
    tp = intersection
    fp = ((preds == 1) & (binary_labels == 0)).sum().item()
    fn = ((preds == 0) & (binary_labels == 1)).sum().item()
    
    return accuracy, iou, tp, fp, fn

def plot_history(history, save_path=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    num_epochs = len(history['train_loss'])
    x_ticks = np.arange(0, num_epochs, 2)

    # Plot Loss
    ax1.plot(history['train_loss'], label='Train loss')
    ax1.plot(history['val_loss'], label='Valid loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_xticks(x_ticks)
    ax1.legend()

    # Plot Accuracy
    ax2.plot(history['train_acc'], label='Train accuracy')
    ax2.plot(history['val_acc'], label='Valid accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks(x_ticks)
    ax2.legend()

    # Plot IoU
    ax3.plot(history['train_iou'], label='Train IoU')
    ax3.plot(history['val_iou'], label='Valid IoU')
    ax3.set_title('Training and Validation IoU')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('IoU')
    ax3.set_xticks(x_ticks)
    ax3.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Biểu đồ đã được lưu tại: {save_path}")

    plt.show()

def main():
    dataset_path = '/home/loipham/INTERN/Lane_detection/train_set' 
    save_dir = './models'
    num_epochs = 20 
    batch_size = 4
    learning_rate = 5e-4
    
    binary_classes = 2
    embedding_dim = 4

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    full_dataset = LaneDataset(dataset_path=dataset_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Tổng số mẫu: {len(full_dataset)}")
    print(f"Kích thước tập Train: {len(train_dataset)}")
    print(f"Kích thước tập Validation: {len(val_dataset)}")

    model = ENet(binary_seg=binary_classes, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_iou': [], 'val_iou': []
    }

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        model.train()
        total_train_loss, total_train_acc, total_train_iou = 0.0, 0.0, 0.0
        
        for images, binary_labels, instance_labels in tqdm(train_loader, desc="Training"):
            images, binary_labels = images.to(device), binary_labels.to(device)
            instance_labels = instance_labels.to(device)

            optimizer.zero_grad()
            binary_logits, instance_logits = model(images)

            binary_loss, instance_loss = compute_loss(binary_logits, instance_logits, binary_labels, instance_labels)
            total_loss = binary_loss + instance_loss
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            
            acc, iou, _, _, _ = calculate_metrics(binary_logits.detach(), binary_labels)
            total_train_acc += acc
            total_train_iou += iou
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)
        avg_train_iou = total_train_iou / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['train_iou'].append(avg_train_iou)
        print(f"Train | Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, IoU: {avg_train_iou:.4f}")

        model.eval()
        total_val_loss, total_val_acc, total_val_iou = 0.0, 0.0, 0.0
        total_tp, total_fp, total_fn = 0, 0, 0
        
        with torch.no_grad():
            for images, binary_labels, instance_labels in tqdm(val_loader, desc="Validation"):
                images, binary_labels = images.to(device), binary_labels.to(device)
                instance_labels = instance_labels.to(device)

                binary_logits, instance_logits = model(images)
                
                binary_loss, instance_loss = compute_loss(binary_logits, instance_logits, binary_labels, instance_labels)
                total_loss = binary_loss + instance_loss
                total_val_loss += total_loss.item()
                
                acc, iou, tp, fp, fn = calculate_metrics(binary_logits, binary_labels)
                total_val_acc += acc
                total_val_iou += iou
                total_tp += tp
                total_fp += fp
                total_fn += fn

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)
        avg_val_iou = total_val_iou / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        history['val_iou'].append(avg_val_iou)
        
        scheduler.step(avg_val_loss)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"Valid | Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}, IoU: {avg_val_iou:.4f}")
        print(f"Valid | Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

        model_path = os.path.join(save_dir, f'ENET_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)

    chart_path = 'training_history.png'
    plot_history(history, save_path=chart_path)


if __name__ == '__main__':
    main()