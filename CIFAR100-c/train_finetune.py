"""
Training Script for CIFAR-100 Classification using ResNet26.

This script demonstrates:
1. Data augmentation for CIFAR-100.
2. Differential learning rate application for transfer learning (Finetuning).
3. Implementation of Cosine Annealing learning rate scheduling.
4. Model checkpointing based on top-1 validation accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

# Import the model builder from external module
try:
    from build_resnet26 import get_pretrained_resnet26
except ImportError:
    raise ImportError("Required dependency 'build_resnet26.py' not found in current directory.")
# ===========================
# 1. Hyperparameters & Configuration
# ===========================
BATCH_SIZE = 128
EPOCHS = 100          
NUM_CLASSES = 100     # CIFAR-100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Standard CIFAR-100 Normalization Statistics
CIFAR100_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR100_STD  = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

def main():
    print(f"Executing on device: {DEVICE}")

    # ===========================
    # 2. Data Pipeline
    # ===========================
    print("Loading CIFAR-100 dataset...")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

    # Dataset initialization
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ===========================
    # 3. Model Initialization
    # ===========================
    print("正在构建并加载移植后的 ResNet26...")
    model = get_pretrained_resnet26(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # ===========================
    # 4. Optimization Strategy
    # ===========================
    # Group parameters for Discriminative Learning Rates:
    # Use higher LR for randomly initialized layers (head), lower LR for pretrained layers (backbone).
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "conv1" in name or "bn1" in name or "fc" in name: # 注意这里的 fc 对应 build_resnet26 中的命名
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    
    optimizer = optim.SGD([
        {'params': head_params, 'lr': 0.1},
        {'params': backbone_params, 'lr': 0.01}
    ], momentum=0.9, weight_decay=5e-4)
    # Cosine Annealing Decay for smooth convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # ===========================
    # 5. Training and Validation Loop
    # ===========================
    print("Starting training process...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        
        # 验证
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        acc = 100. * test_correct / test_total
        scheduler.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(trainloader):.3f} | Train Acc: {train_acc:.2f}% | Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            print(f"   --> New optimal weights detected. Saving checkpoint (Acc: {best_acc:.2f}%).")
            torch.save(model.state_dict(), 'resnet26_cifar100_finetuned.pth')

if __name__ == "__main__":
    main()