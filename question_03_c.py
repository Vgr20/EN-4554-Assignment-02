# #import necessary modules
# import numpy as np
# import torch
# import torchvision
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from torchvision import datasets,transforms 
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from torch.utils.data import Subset
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
# from tqdm import tqdm

# # Define directory
# data_dir = './caltech101'

# # Define the transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Download the dataset
# dataset = datasets.Caltech101(data_dir, download=True, transform=transform)

# # Get the labels for stratification
# labels = dataset.y

# # Stratified train-test split
# train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.3333, stratify=labels, random_state=42)

# # Create subsets of the original dataset for train and test sets
# train_dataset = Subset(dataset, train_idx)
# test_dataset = Subset(dataset, test_idx)

# # Define transformations for training (with augmentation) and testing (without augmentation)
# transform_train = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# transform_test = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Assign transformations to the train and test datasets
# train_dataset.dataset.transform = transform_train
# test_dataset.dataset.transform = transform_test

# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# print("Train and test split completed successfully!")

# print(len(train_loader))
# print(len(test_loader))

# # Load Pretrained Model (ResNet-18 in this case)
# model = resnet18(pretrained=True)

# # Modify the last layer for 102 classes (Caltech-101 + 1 background class)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 102)

# # Move model to device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Define Loss Function and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # Learning rate scheduler to decay the learning rate after every few epochs
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# # Training the model
# def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch+1}/{num_epochs}')
        
#         # Training phase
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for inputs, labels in tqdm(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device)

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()

#             # Calculate accuracy
#             _, preds = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (preds == labels).sum().item()

#             running_loss += loss.item() * inputs.size(0)
        
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = correct / total

#         print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')

#         # Step the scheduler
#         scheduler.step()

#         # Validation phase
#         val_loss, val_acc = evaluate_model(model, test_loader)
#         print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# # Evaluation function for validation
# def evaluate_model(model, loader):
#     model.eval()
#     correct = 0
#     total = 0
#     val_loss = 0.0

#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             # Forward pass
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
#             val_loss += loss.item() * inputs.size(0)
            
#             # Calculate accuracy
#             _, preds = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (preds == labels).sum().item()

#     val_loss /= len(loader.dataset)
#     val_acc = correct / total
#     return val_loss, val_acc

# # Fine-tune the model
# train_model(model, criterion, optimizer, scheduler, num_epochs=10)

# # Test the model on the test set
# def test_model(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             # Forward pass
#             outputs = model(inputs)
            
#             # Calculate accuracy
#             _, preds = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (preds == labels).sum().item()

#     test_acc = correct / total
#     print(f'Test Accuracy: {test_acc * 100:.2f}%')

# # Test on validation dataset
# test_model(model, test_loader)


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18,resnet50
from tqdm import tqdm

# Define the directory to store the dataset
data_dir = './caltech101'

# Data Augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Download and load dataset
dataset = torchvision.datasets.Caltech101(root=data_dir, download=True, transform=transform_train)
test_dataset = torchvision.datasets.Caltech101(root=data_dir, download=True, transform=transform_test)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load Pretrained Model (ResNet-18 in this case)
model = resnet50(pretrained=True)

# Modify the last layer for 102 classes (Caltech-101 + 1 background class)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 102)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler to decay the learning rate after every few epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')

        # Step the scheduler
        scheduler.step()

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# Evaluation function for validation
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_loss /= len(loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc

# Fine-tune the model
train_model(model, criterion, optimizer, scheduler, num_epochs=10)

# Test the model on the test set
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    test_acc = correct / total
    print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Test on validation dataset
test_model(model, val_loader)

