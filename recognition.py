import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms


class ImageRecognizer:
    def __init__(self, data_type='tiles'):
        """
        Initializes the ImageRecognizer class.

        :param data_type: Type of data being classified (default is 'tiles').
        """
        self.data_type = data_type

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Initialize model and load weights
        self.class_names = os.listdir(f"dataset/{data_type}/train")
        self.num_classes = len(self.class_names)
        self.model = self.load_model()

        # Define the transformations to apply to the input images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
        ])

    def load_model(self):
        """
        Loads the model architecture and weights based on the trained model.
        """
        # Using ResNet18 by default, can extend for other types
        model = models.resnet18()

        # Modify the final layer to match the number of classes
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        # Load the trained weights
        model.load_state_dict(
            torch.load(
                os.path.join(f'models/{self.data_type}/best_tile_classifier.pth'),
                map_location=self.device,
                weights_only=True
            )
        )

        # Move model to the device
        model = model.to(self.device)

        # Set model to evaluation mode
        model.eval()

        return model

    def classify_tile(self, image_path):
        """
        Classifies a single tile image and returns the predicted class and confidence.

        :param image_path: Path to the image file.
        :return: (predicted_class, confidence_value)
        """
        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply the transformations to the image
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)

        # Move image tensor to the same device as the model
        image_tensor = image_tensor.to(self.device)

        # Perform the classification
        with torch.no_grad():
            output = self.model(image_tensor)

        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)

        # Get the predicted class and confidence
        confidence, predicted_class_idx = torch.max(probabilities, 1)
        predicted_class = self.class_names[predicted_class_idx.item()]
        confidence_value = confidence.item()

        return predicted_class, confidence_value

    def evaluate_screenshots(self, screenshot_dir='screenshots'):
        """
        Evaluates the model on the screenshots and prints out incorrect predictions.

        :param screenshot_dir: Directory of the screenshots.
        """
        # list all directories in the screenshot directory and iterate over them
        target_dir = os.path.join(screenshot_dir, self.data_type)
        for directory in os.listdir(target_dir):
            if not os.path.isdir(os.path.join(target_dir, directory)):
                continue

            for file in os.listdir(os.path.join(target_dir, directory)):
                image_path = os.path.join(target_dir, directory, file)
                predicted_class, confidence = self.classify_tile(image_path)

                # Print only when the prediction is incorrect
                if predicted_class != directory:
                    print(
                        f'Predicted tile for {file} is: {predicted_class}, correct is {directory}, confidence: {confidence:.2f}'
                    )

    def evaluate_dataset(self, dataset_dir='test'):
        """
        Evaluates the model on the test dataset and prints out incorrect predictions.

        :param dataset_dir: Directory of the test dataset.
        """
        for root, dirs, files in os.walk(os.path.join(f'dataset/{self.data_type}/{dataset_dir}')):
            for single_dir in dirs:
                for file in os.listdir(os.path.join(root, single_dir)):
                    image_path = os.path.join(root, single_dir, file)
                    predicted_class, confidence = self.classify_tile(image_path)

                    # Print only when the prediction is incorrect
                    if predicted_class != single_dir:
                        print(
                            f'Predicted tile for {file} is: {predicted_class}, correct is {single_dir}, confidence: {confidence:.2f}'
                        )


# Example usage:
# Initialize the recognizer for tile classification
recognizer = ImageRecognizer(data_type='level')

# Evaluate the test dataset and print incorrect classifications
# recognizer.evaluate_dataset(dataset_dir="test")
# recognizer.evaluate_dataset(dataset_dir="train")
recognizer.evaluate_screenshots(screenshot_dir='screenshots')
