import os
import time
from os.path import realpath, dirname
from pprint import pprint

import pyautogui
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import win32gui
import win32ui
from PIL import Image
from torchvision import transforms
from win32api import GetSystemMetrics


class Row:
    def __init__(self, top_left: tuple, bot_right: tuple, tile_count: int = 6, row: int = None):
        """
        Initialize a row with the given parameters

        :param top_left: top left xy coordinates of the row
        :param bot_right: bottom right xy coordinates of the row
        :param tile_count: number of tiles in the row
        :param row: index of the row, used primarily for debugging purposes
        """
        self.row = row
        self.top_left = top_left
        self.bot_right = bot_right
        self.tile_count = tile_count
        self.width = (bot_right[0] - top_left[0]) / tile_count

    def get_field(self, index: int) -> tuple[int, int, int, int]:
        """
        Get the coordinates of the field at the given index

        :param index:
        :return:
        """
        return (
            int(0 + index * self.width),
            0,
            int((index + 1) * self.width) - 5,
            self.bot_right[1] - self.top_left[1]
        )


class Transcendence:
    def __init__(self):
        self.script_dir = realpath(dirname(__file__))
        self.model = None

        # Class labels, must match the order used during training (folders in ImageFolder)
        self.class_names = os.listdir("dataset/train")
        # Define the same model structure as in the training script
        self.num_classes = len(self.class_names)  # Number of tile types

        # Image transformations (should match the ones used in training)
        # Input size for the model (for ResNet, 224x224 is standard)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.load_model()

        # select tile  (tr -> row, td -> column, current one is 7th row, 8th column)
        # document.querySelector('.border-separate > tbody:nth-child(1) > tr:nth-child(7) > td:nth-child(8) > div:nth-child(1) > div:nth-child(3)').click()
        # select special field (1 -> normal, 2 -> destroyed, 3 -> distorted, 4 -> addition, 5 -> blessing, 6 -> mystery, 7 -> enhancement, 8 -> clone, 9 -> relocation)
        # document.querySelector('.z-20 > ul:nth-child(1) > li:nth-child(2)').click()

    def load_model(self):
        if self.model is None:
            # No pretraining this time, as we're loading a trained model
            model = models.resnet18()
            # Adjust the final layer to match the number of classes
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            # Load the model weights
            model.load_state_dict(torch.load('models/best_tile_classifier.pth', map_location=device,
                                             weights_only=True))
            model.eval()  # Set the model to evaluation mode
            self.model = model
        return self.model

    @staticmethod
    def highlight_row(row: Row):
        """
        Highlight the row on the screen for validation reasons

        :param row:
        :return:
        """
        dc = win32gui.GetDC(0)
        dcObj = win32ui.CreateDCFromHandle(dc)
        hwnd = win32gui.WindowFromPoint((0, 0))
        monitor = (0, 0, GetSystemMetrics(0), GetSystemMetrics(1))

        while True:
            # draw rectangle around the tiles on the first row
            for i in range(row.tile_count):
                dcObj.Rectangle((int(row.top_left[0] + i * row.width), int(row.top_left[1]),
                                 int(row.top_left[0] + (i + 1) * row.width), int(row.bot_right[1])))
            win32gui.InvalidateRect(hwnd, monitor, True)  # Refresh the entire monitor

    def check_row(self, screenshot: Image, row: Row, save_screenshot: bool = False):
        """
        Check the row for the different tile types

        :param save_screenshot:
        :param screenshot:
        :param row:
        :return:
        """
        # Ensure the input and model are on the same device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tmp_screenshot = screenshot.crop((row.top_left[0], row.top_left[1], row.bot_right[0], row.bot_right[1]))

        tile_types = []
        for i in range(row.tile_count):
            single_field = tmp_screenshot.crop(row.get_field(i))
            if save_screenshot:
                single_field.save(
                    f"{self.script_dir}/assets/transcendence/dumps/row_{row.row}_tile_{i + 1}_{int(time.time() * 1000)}.png"
                )

            # Ensure the image is RGB
            image = single_field.convert('RGB')

            # Apply the transformations
            # Add a batch dimension (1, C, H, W)
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(device)

            # Perform the prediction
            # Disable gradient computation (we're just doing inference)
            with torch.no_grad():
                output = self.model(image_tensor)

            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)

            # Get the predicted class index and its probability
            confidence, predicted_class_idx = torch.max(probabilities, 1)
            predicted_class = self.class_names[predicted_class_idx.item()]

            # Convert confidence from tensor to a Python float
            confidence_value = confidence.item()

            print(f"Row {row.row} Tile {i + 1} is {predicted_class} with confidence {confidence_value * 100:.2f}%")
            tile_types.append((confidence_value, predicted_class))

        return tile_types

    def run(self):
        print("Transcendence script started")

        complexities = {
            # level 1,2,3 of transcendence have 6 tiles in each row
            6: [
                Row((786, 357), (1138, 391), 6, 1),
                Row((782, 392), (1143, 426), 6, 2),
                Row((778, 428), (1147, 465), 6, 3),
                Row((773, 467), (1152, 505), 6, 4),
                Row((768, 507), (1157, 547), 6, 5),
                Row((763, 549), (1162, 592), 6, 6),
            ],
            # level 4,5,6,7 of transcendence have 7 tiles in each row
            7: [
                Row((760, 341), (1165, 373), 7, 1),
                Row((755, 375), (1170, 408), 7, 2),
                Row((750, 410), (1175, 445), 7, 3),
                Row((745, 447), (1181, 485), 7, 4),
                Row((740, 486), (1186, 526), 7, 5),
                Row((735, 528), (1191, 569), 7, 6),
                Row((730, 571), (1196, 615), 7, 7),

            ],
        }

        rows = complexities[7]

        start_time = time.time()
        iterations = 1
        current_board = {}
        #self.highlight_row(complexities[7][6])
        for i in range(iterations):
            # take a screenshot of the current board to analyze (faster than taking a screenshot for each tile)
            current_screenshot = pyautogui.screenshot()
            for row in rows:
                tmp_row = self.check_row(current_screenshot, row, True)
                if row.row not in current_board:
                    current_board[row.row] = tmp_row
                else:
                    # iterate over all rows and update the row info with the highest confidence
                    for j in range(row.tile_count):
                        if tmp_row[j][0] > current_board[row.row][j][0]:
                            current_board[row.row][j] = tmp_row[j]

        pprint(current_board)

        print("Transcendence script finished in", time.time() - start_time, "seconds")


if __name__ == '__main__':
    Transcendence().run()
