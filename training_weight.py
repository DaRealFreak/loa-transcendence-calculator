import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# Function to inject random Gaussian noise into the image
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class WeightModelTraining:
    def __init__(self, model_name='resnet18',
                 num_epochs=250, patience=75, all_correct_required=5,
                 data_dir='dataset/', data_type='tiles',
                 add_noise=False, add_brightness=False):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.patience = patience
        self.all_correct_required = all_correct_required
        self.data_dir = data_dir
        self.data_type = data_type

        # Load and initialize the model
        self.model = self._initialize_model()

        # Setup device (GPU or CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        self.model = self.model.to(self.device)

        # Define class weights based on the dataset
        self.class_counts = self._calculate_class_counts()
        self.class_weights = self._calculate_class_weights()

        # Move class weights to the same device as the model
        self.weights = torch.tensor([self.class_weights[cls] for cls in sorted(self.class_counts.keys())],
                                    dtype=torch.float).to(self.device)

        # Define criterion and optimizer
        self.criterion = nn.CrossEntropyLoss(weight=self.weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Set up data loaders
        self.train_loader, self.test_loader = self._load_data(add_noise=add_noise, add_brightness=add_brightness)

    def _initialize_model(self):
        """Initialize the model based on the model_name passed."""
        num_classes = len(os.listdir(os.path.join(self.data_dir, self.data_type, 'train')))

        if self.model_name == 'resnet18':
            model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        elif self.model_name == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # Add more models as needed
        else:
            raise ValueError(f"Model {self.model_name} not supported")

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def _calculate_class_counts(self):
        """Calculate the number of samples per class in the dataset."""
        class_counts = {}
        for trainings_dir in os.listdir(os.path.join(self.data_dir, self.data_type, 'train')):
            class_counts[trainings_dir] = len(
                os.listdir(os.path.join(self.data_dir, self.data_type, 'train', trainings_dir)))
        return class_counts

    def _calculate_class_weights(self):
        """Calculate class weights based on class frequency."""
        total_samples = sum(self.class_counts.values())
        class_weights = {cls: total_samples / count for cls, count in self.class_counts.items()}
        return class_weights

    def _load_data(self, add_noise=False, add_brightness=False):
        """Load training and testing datasets."""
        transformations = [transforms.Resize((224, 224))]
        if add_brightness:
            # Randomly adjust brightness (50% to 150% of the original)
            transformations.append(transforms.ColorJitter(brightness=(0.5, 1.5)))
        transformations.append(transforms.ToTensor())
        if add_noise:
            # Add Gaussian noise to the image
            transformations.append(AddGaussianNoise(mean=0.0, std=0.1))
        # Normalize the image based on ImageNet statistics
        transformations.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        transform = transforms.Compose(transformations)

        train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, self.data_type, 'train'), transform=transform)
        test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, self.data_type, 'test'), transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader

    def train(self):
        """Train the model with early stopping."""
        best_accuracy = 0.0
        trigger_times = 0
        all_correct_times = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Training loop
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_accuracy = 100 * correct / total
            print(
                f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / total:.4f}, Accuracy: {epoch_accuracy:.2f}%'
            )

            # Validation loop
            test_accuracy = self.evaluate()
            print(f'Test Accuracy: {test_accuracy:.2f}%')

            if test_accuracy == 100 and self.all_correct_required > 0:
                all_correct_times += 1
                if all_correct_times >= self.all_correct_required:
                    print(f'All correct {self.all_correct_required} times. Stopping training.')
                    torch.save(self.model.state_dict(), f'models/{self.data_type}/best_tile_classifier.pth')
                    break

            # Early Stopping
            if test_accuracy >= best_accuracy:
                best_accuracy = test_accuracy
                trigger_times = 0
                torch.save(self.model.state_dict(), f'models/{self.data_type}/best_tile_classifier.pth')
            else:
                trigger_times += 1

            if 0 < self.patience <= trigger_times:
                print(f'Early stopping at epoch {epoch + 1}. Best accuracy: {best_accuracy:.2f}%')
                break

        print(f'Final Test Accuracy: {best_accuracy:.2f}%')

    def evaluate(self):
        """Evaluate the model on the test dataset."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total


# Example usage:
# To train a ResNet18 model
classifier = WeightModelTraining(
    model_name='resnet18',
    data_dir='dataset/', data_type='grace',
    num_epochs=250,
    patience=0,
    all_correct_required=0,
    add_noise=True, add_brightness=True
)
classifier.train()
