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


class ScreenshotArea:
    def __init__(self, top_left: tuple, bot_right: tuple, description: str = None):
        """
        Initialize a screenshot area with the given parameters
        :param top_left:
        :param bot_right:
        """
        self.top_left = top_left
        self.bot_right = bot_right

    def get_field(self) -> tuple[int, int, int, int]:
        """
        Get the coordinates of the field

        :return:
        """
        return (
            self.top_left[0],
            self.top_left[1],
            self.bot_right[0],
            self.bot_right[1]
        )


class Card(ScreenshotArea):
    def __init__(self, top_left: tuple, bot_right: tuple, position: int):
        """
        Initialize a card with the given parameters

        :param top_left: top left xy coordinates of the card
        :param bot_right: bottom right xy coordinates of the card
        """
        super().__init__(top_left, bot_right)
        self.position = position


class Prediction:
    def __init__(self, confidence: float, prediction: str):
        self.confidence = confidence
        self.prediction = prediction

    def __repr__(self) -> str:
        return f"({self.prediction}, {('{:.6f}'.format(self.confidence * 100)).rstrip('0').rstrip('.')}%)"


class TranscendenceInfo:
    def __init__(self, gear_part: str, level: Prediction, grace: Prediction,
                 retries: Prediction, changes: Prediction,
                 cards: dict[int, Prediction], board: dict[int | None, list[Prediction]],
                 duration: float = 0):
        self.gear_part = gear_part
        self.level = level
        self.grace = grace
        self.retries = retries
        self.changes = changes
        self.cards = cards
        self.board = board
        self.duration = duration

    def __repr__(self) -> str:
        # Formatting the Cards dictionary with each key-value pair on a new line
        formatted_cards = "\n".join([f"  {k}: {v}" for k, v in self.cards.items()])

        # Formatting the Board dictionary with each key-value pair on a new line
        formatted_board = "\n".join([f"  {k}: {v}" for k, v in self.board.items()])

        return (f"Gear part: {self.gear_part}\n"
                f"Level: {self.level}\n"
                f"Grace: {self.grace}\n"
                f"Retries: {self.retries}\n"
                f"Changes: {self.changes}\n"
                f"Cards:\n{formatted_cards}\n"
                f"Board:\n{formatted_board}\n"
                f"Duration: {self.duration:.2f} seconds")


class Transcendence:
    def __init__(self):
        self.script_dir = realpath(dirname(__file__))
        self.tile_model = None
        self.card_model = None
        self.tries_model = None
        self.changes_model = None
        self.level_model = None
        self.grace_model = None

        # Class labels, must match the order used during training (folders in ImageFolder)
        self.tile_class_names = os.listdir("dataset/tiles/train")
        self.tile_num_classes = len(self.tile_class_names)

        self.card_class_names = os.listdir("dataset/cards/train")
        self.card_num_classes = len(self.card_class_names)

        self.tries_class_names = os.listdir("dataset/tries/train")
        self.tries_num_classes = len(self.tries_class_names)

        self.changes_class_names = os.listdir("dataset/changes/train")
        self.changes_num_classes = len(self.changes_class_names)

        self.level_class_names = os.listdir("dataset/level/train")
        self.level_num_classes = len(self.level_class_names)

        self.grace_class_names = os.listdir("dataset/grace/train")
        self.grace_num_classes = len(self.grace_class_names)

        # Image transformations (should match the ones used in training)
        # Input size for the model (for ResNet, 224x224 is standard)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._load_models()

        # select tile  (tr -> row, td -> column, current one is 7th row, 8th column)
        # document.querySelector('.border-separate > tbody:nth-child(1) > tr:nth-child(7) > td:nth-child(8) > div:nth-child(1) > div:nth-child(3)').click()
        # select special field (1 -> normal, 2 -> destroyed, 3 -> distorted, 4 -> addition, 5 -> blessing, 6 -> mystery, 7 -> enhancement, 8 -> clone, 9 -> relocation)
        # document.querySelector('.z-20 > ul:nth-child(1) > li:nth-child(2)').click()

    def _load_models(self):
        if self.tile_model is None:
            # No pretraining this time, as we're loading a trained model
            model = models.resnet18()
            # Adjust the final layer to match the number of classes
            model.fc = nn.Linear(model.fc.in_features, self.tile_num_classes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            # Load the model weights
            model.load_state_dict(torch.load('models/tiles/best_tile_classifier.pth', map_location=device,
                                             weights_only=True))
            model.eval()
            self.tile_model = model

        if self.card_model is None:
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, self.card_num_classes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.load_state_dict(torch.load('models/cards/best_tile_classifier.pth', map_location=device,
                                             weights_only=True))
            model.eval()
            self.card_model = model

        if self.tries_model is None:
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, self.tries_num_classes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.load_state_dict(torch.load('models/tries/best_tile_classifier.pth', map_location=device,
                                             weights_only=True))
            model.eval()
            self.tries_model = model

        if self.changes_model is None:
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, self.changes_num_classes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.load_state_dict(torch.load('models/changes/best_tile_classifier.pth', map_location=device,
                                             weights_only=True))
            model.eval()
            self.changes_model = model

        if self.level_model is None:
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, self.level_num_classes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.load_state_dict(torch.load('models/level/best_tile_classifier.pth', map_location=device,
                                             weights_only=True))
            model.eval()
            self.level_model = model

        if self.grace_model is None:
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, self.grace_num_classes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.load_state_dict(torch.load('models/grace/best_tile_classifier.pth', map_location=device,
                                             weights_only=True))
            model.eval()
            self.grace_model = model

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

    def _check_card(self, screenshot: Image, card: Card, save_screenshot: bool = False) -> Prediction:
        """
        Check the card for the different card types

        :param screenshot:
        :param card:
        :param save_screenshot:
        :return:
        """
        # Ensure the input and model are on the same device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tmp_screenshot = screenshot.crop(card.get_field())

        if save_screenshot:
            tmp_screenshot.save(
                f"{self.script_dir}/assets/transcendence/dumps/card_{card.position}_{int(time.time() * 1000)}.png"
            )

        # Ensure the image is RGB
        image = tmp_screenshot.convert('RGB')

        # Apply the transformations
        # Add a batch dimension (1, C, H, W)
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Perform the prediction
        # Disable gradient computation (we're just doing inference)
        with torch.no_grad():
            output = self.card_model(image_tensor)

        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)

        # Get the predicted class index and its probability
        confidence, predicted_class_idx = torch.max(probabilities, 1)
        predicted_class = self.card_class_names[predicted_class_idx.item()]

        # Convert confidence from tensor to a Python float
        confidence_value = confidence.item()

        # print(f"Card {card.position} is {predicted_class} with confidence {confidence_value * 100:.2f}%")
        return Prediction(confidence_value, predicted_class)

    def _check_screenshot_area(self,
                               screenshot: Image, area: ScreenshotArea,
                               model: nn.Module, class_names: list[str],
                               screenshot_type='untyped',
                               save_screenshot: bool = False
                               ) -> Prediction:
        """
        Check the screenshot area for the different tile types

        :param screenshot:
        :param area:
        :param model:
        :param class_names:
        :param save_screenshot:
        :return:
        """

        # Ensure the input and model are on the same device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tmp_screenshot = screenshot.crop(area.get_field())

        if save_screenshot:
            tmp_screenshot.save(
                f"{self.script_dir}/assets/transcendence/dumps/{screenshot_type}_{int(time.time() * 1000)}.png"
            )

        # Ensure the image is RGB
        image = tmp_screenshot.convert('RGB')

        # Apply the transformations
        # Add a batch dimension (1, C, H, W)
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Perform the prediction
        # Disable gradient computation (we're just doing inference)
        with torch.no_grad():
            output = model(image_tensor)

        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)

        # Get the predicted class index and its probability
        confidence, predicted_class_idx = torch.max(probabilities, 1)
        predicted_item = class_names[predicted_class_idx.item()]

        # Convert confidence from tensor to a Python float
        confidence_value = confidence.item()

        return Prediction(confidence_value, predicted_item)

    def _check_row(self, screenshot: Image, row: Row, save_screenshot: bool = False) -> list[Prediction]:
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
                output = self.tile_model(image_tensor)

            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)

            # Get the predicted class index and its probability
            confidence, predicted_class_idx = torch.max(probabilities, 1)
            predicted_class = self.tile_class_names[predicted_class_idx.item()]

            # Convert confidence from tensor to a Python float
            confidence_value = confidence.item()

            # print(f"Row {row.row} Tile {i + 1} is {predicted_class} with confidence {confidence_value * 100:.2f}%")
            tile_types.append(Prediction(confidence_value, predicted_class))

        return tile_types

    def _get_current_equipment_parts(self) -> str:
        """
        Get the current equipment parts

        :return:
        """
        try:
            selection = pyautogui.locateOnScreen(
                f"{self.script_dir}/assets/transcendence/current_equipment_selection.png",
                confidence=0.9,
                region=(60, 80, 150 - 60, 700 - 80))

            # switch case for the different equipment parts
            if selection is not None:
                y = selection.top + selection.height // 2
                if 110 < y < 200:
                    return 'helmet'
                elif 210 < y < 300:
                    return 'shoulders'
                elif 310 < y < 400:
                    return 'chestpiece'
                elif 410 < y < 500:
                    return 'pants'
                elif 510 < y < 600:
                    return 'gloves'
                elif 610 < y < 700:
                    return 'weapon'
                else:
                    return ''
        except pyautogui.ImageNotFoundException:
            return ''

    def get_current_information(self) -> TranscendenceInfo:
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
            # level 4,5 of transcendence have 7 tiles in each row
            7: [
                Row((760, 341), (1165, 373), 7, 1),
                Row((755, 375), (1170, 408), 7, 2),
                Row((750, 410), (1175, 445), 7, 3),
                Row((745, 447), (1181, 485), 7, 4),
                Row((740, 486), (1186, 526), 7, 5),
                Row((735, 528), (1191, 569), 7, 6),
                Row((730, 571), (1196, 615), 7, 7),

            ],
            # level 6,7 of transcendence have 8 tiles in each row
            8: [
                Row((734, 324), (1191, 356), 8, 1),
                Row((729, 357), (1197, 390), 8, 2),
                Row((723, 392), (1203, 426), 8, 3),
                Row((717, 428), (1209, 465), 8, 4),
                Row((711, 467), (1215, 505), 8, 5),
                Row((704, 507), (1222, 547), 8, 6),
                Row((697, 549), (1230, 592), 8, 7),
                Row((690, 594), (1237, 638), 8, 8),
            ]
        }

        cards = [
            Card((434, 951), (483, 1014), 5),
            Card((514, 951), (563, 1014), 4),
            Card((594, 918), (670, 1014), 3),
            Card((735, 787), (888, 1018), 2),
            Card((1029, 787), (1181, 1018), 1),
        ]

        retry_area = ScreenshotArea((1066, 726), (1173, 750))
        change_area = ScreenshotArea((902, 1046), (1013, 1076))
        level_area = ScreenshotArea((772, 20), (1136, 45))
        grace_area = ScreenshotArea((414, 44), (476, 67))

        start_time = time.time()
        iterations = 1
        current_board = {}
        current_cards = {}
        current_retries = (0, 0)
        current_changes = (0, 0)
        current_grace = (0, 0)
        current_level = (0, 0)
        current_equipment_part = self._get_current_equipment_parts()

        # self.highlight_row(complexities[7][6])
        for i in range(iterations):
            # take a screenshot of the current board to analyze (faster than taking a screenshot for each tile)
            current_screenshot = pyautogui.screenshot()
            for card in cards[::-1]:
                current_cards[card.position] = self._check_card(current_screenshot, card)

            current_retries = self._check_screenshot_area(
                current_screenshot, retry_area, self.tries_model, self.tries_class_names, 'retry'
            )
            current_changes = self._check_screenshot_area(
                current_screenshot, change_area, self.changes_model, self.changes_class_names, 'change'
            )
            current_grace = self._check_screenshot_area(
                current_screenshot, grace_area, self.grace_model, self.grace_class_names, 'grace'
            )

            current_level = self._check_screenshot_area(
                current_screenshot, level_area, self.level_model, self.level_class_names, 'level'
            )

            if int(current_level.prediction) in (1, 2, 3):
                rows = complexities[6]
            elif int(current_level.prediction) in (4, 5):
                rows = complexities[7]
            else:
                rows = complexities[8]

            for row in rows:
                tmp_row = self._check_row(current_screenshot, row)
                if row.row not in current_board:
                    current_board[row.row] = tmp_row
                else:
                    # iterate over all rows and update the row info with the highest confidence
                    for j in range(row.tile_count):
                        if tmp_row[j].confidence > current_board[row.row][j].confidence:
                            current_board[row.row][j] = tmp_row[j]

        return TranscendenceInfo(
            gear_part=current_equipment_part,
            level=current_level,
            grace=current_grace,
            retries=current_retries,
            changes=current_changes,
            cards=current_cards,
            board=current_board,
            duration=time.time() - start_time
        )


if __name__ == '__main__':
    info = Transcendence().get_current_information()
    pprint(info)
