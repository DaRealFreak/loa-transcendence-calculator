import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms


def load_model():
    model = models.resnet18()  # No pretraining this time, as we're loading a trained model
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust the final layer to match the number of classes
    model.load_state_dict(torch.load('models/best_tile_classifier.pth', map_location=torch.device('cpu'),
                                     weights_only=True))  # Load the model weights
    model.eval()  # Set the model to evaluation mode
    return model


# Image transformations (should match the ones used in training)
input_size = 224  # Input size for the model (for ResNet, 224x224 is standard)
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels, must match the order used during training (folders in ImageFolder)
class_names = os.listdir("dataset/train")
# Define the same model structure as in the training script
num_classes = len(class_names)  # Number of tile types


# Function to classify an image with confidence score
def classify_tile(model, image_path):
    # Load the image using Pillow
    image = Image.open(image_path).convert('RGB')  # Ensure the image is RGB

    # Apply the transformations
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension (1, C, H, W)

    # Perform the prediction
    with torch.no_grad():  # Disable gradient computation (we're just doing inference)
        output = model(image_tensor)

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)

    # Get the predicted class index and its probability
    confidence, predicted_class_idx = torch.max(probabilities, 1)
    predicted_class = class_names[predicted_class_idx.item()]

    # Convert confidence from tensor to a Python float
    confidence_value = confidence.item()

    return predicted_class, confidence_value


# Load the model
model = load_model()

# go through all folders in dataset/test and classify all files in each folder and classify them
for root, dirs, files in os.walk("dataset/test"):
    for dir in dirs:
        for file in os.listdir(f"dataset/test/{dir}"):
            image_path = os.path.join(root, dir, file)
            predicted_class, confidence = classify_tile(model, image_path)
            print(f'predicted tile for {file} is: {predicted_class}, correct is {dir}, confidence: {confidence:.2f}')

# Example usage
# image_path = 'dataset/test/Addition/row_4_tile_6_1727083909347.png'  # Path to the tile image
# predicted_class = classify_tile(image_path)
# print(f'The predicted tile class is: {predicted_class}')
