import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Define the tile classes and their respective sample sizes
class_counts = {}
for trainings_dir in os.listdir("dataset/train"):
    class_counts[trainings_dir] = len(os.listdir(f"dataset/train/{trainings_dir}"))

# Calculate class weights (inverse of the class frequency)
total_samples = sum(class_counts.values())
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

# Convert weights to a tensor, ensuring order matches dataset class order
# Assuming the dataset uses the same alphabetical order as the class names
weights = torch.tensor([class_weights[cls] for cls in sorted(class_counts.keys())], dtype=torch.float)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load your dataset (use appropriate directory)
data_dir = 'dataset/'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model (ResNet18)
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
num_classes = len(class_counts)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

if torch.cuda.is_available():
    print('Using GPU for training.')
else:
    print('Using CPU for training.')

# Move class weights to the same device as the model
weights = weights.to(device)

# Use class weights in CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=weights)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 250
best_accuracy = 0.0
patience = 50  # Number of epochs to wait for improvement
trigger_times = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Training loop
    for inputs, labels in train_loader:
        # Move inputs and labels to the same device as the model (cuda or cpu)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / total:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    # Test/Validation loop
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the same device as the model (cuda or cpu)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Early Stopping
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        trigger_times = 0
        # Optionally save the best model
        torch.save(model.state_dict(), 'models/best_tile_classifier.pth')
    else:
        trigger_times += 1

    if trigger_times >= patience:
        print(f'Early stopping at epoch {epoch + 1}. Best accuracy: {best_accuracy:.2f}%')
        break

# Test loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        # Move inputs and labels to the same device as the model (cuda or cpu)
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'models/tile_classifier_with_weights.pth')
