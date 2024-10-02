import os
import time
from os.path import realpath, dirname
from pprint import pprint

import pyautogui
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch.nn import Module
from torchvision import transforms


class Row:
    def __init__(self, top_left: tuple, bot_right: tuple, tile_count: int = 6, row: int = None):
        """
        Initialize the Row object.

        :param top_left: top left xy coordinates of the row.
        :param bot_right: bottom right xy coordinates of the row.
        :param tile_count: the number of tiles in the row.
        :param row: index of the row, used primarily for debugging purposes.
        """
        self.row = row
        self.top_left = top_left
        self.bot_right = bot_right
        self.tile_count = tile_count
        self.width = (bot_right[0] - top_left[0]) / tile_count

    def get_field(self, index: int) -> tuple[int, int, int, int]:
        """
        Get the coordinates of the field at the given index.

        :param index: The index of the field.
        :return: The coordinates of the field.
        """
        return int(index * self.width), 0, int((index + 1) * self.width) - 5, self.bot_right[1] - self.top_left[1]


class ScreenshotArea:
    def __init__(self, top_left: tuple, bot_right: tuple):
        """
        Initialize the ScreenshotArea object.

        :param top_left: top left xy coordinates of the area.
        :param bot_right: bottom right xy coordinates of the area.
        """
        self.top_left = top_left
        self.bot_right = bot_right

    def get_field(self) -> tuple[int, int, int, int]:
        """
        Get the field based on the top left and bottom right coordinates.

        :return: The field coordinates.
        """
        return self.top_left[0], self.top_left[1], self.bot_right[0], self.bot_right[1]


class Card(ScreenshotArea):
    def __init__(self, top_left: tuple, bot_right: tuple, position: int):
        """
        Initialize the Card object, which is a ScreenshotArea with an added position parameter.

        :param top_left:
        :param bot_right:
        :param position:
        """
        super().__init__(top_left, bot_right)
        self.position = position


class Prediction:
    def __init__(self, confidence: float, prediction: str):
        """
        Initialize the Prediction object.

        :param confidence: the confidence of the prediction.
        :param prediction: the predicted label.
        """
        self.confidence = confidence
        self.prediction = prediction

    def __repr__(self) -> str:
        """
        Format the Prediction object for printing.
        Precision is set to 6 decimal places and the percentage is formatted to remove trailing zeros.

        :return:
        """
        return f"({self.prediction}, {('{:.6f}'.format(self.confidence * 100)).rstrip('0').rstrip('.')}%)"


class TranscendenceInfo:
    def __init__(self, gear_part: str, level: Prediction, grace: Prediction,
                 retries: Prediction, changes: Prediction, cards: dict[int, Prediction],
                 board: dict[int | None, list[Prediction]], duration: float = 0):
        """
        Initialize the TranscendenceInfo object.

        :param gear_part: The current gear part being transcended.
        :param level: The predicted level of the transcendence.
        :param grace: The predicted grace of the transcendence.
        :param retries: The predicted retries of the transcendence.
        :param changes: The predicted changes of the transcendence.
        :param cards: The predicted cards of the transcendence.
        :param board: The predicted board of the transcendence.
        :param duration: The duration of the transcending process.
        """
        self.gear_part = gear_part
        self.level = level
        self.grace = grace
        self.retries = retries
        self.changes = changes
        self.cards = cards
        self.board = board
        self.duration = duration

    def __repr__(self) -> str:
        """
        Format the TranscendenceInfo object for printing.

        :return:
        """
        formatted_cards = "\n".join([f"  {k}: {v}" for k, v in self.cards.items()])
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
    def __init__(self, save_screenshots: bool = False, screenshot_dir: str = "screenshots"):
        """
        Initialize the Transcendence class and load all the models.

        :param save_screenshots:
        """
        self.script_dir = realpath(dirname(__file__))
        self.save_screenshots = save_screenshots
        self.screenshot_dir = screenshot_dir
        self.models = {}
        self.class_names = {}
        self.transform = self._init_transform()

        self._load_all_models()

    @staticmethod
    def _init_transform():
        """
        Initialize the transformations for the model.

        :return:
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def _load_model(path: str, num_classes: int) -> Module:
        """
        Load the model based on the given path and number of classes.

        :param path: The path to the model weights.
        :param num_classes: The number of classes the model should predict.
        :return:
        """
        # No pretraining this time, as we're loading a trained model
        model = models.resnet18()
        # Adjust the final layer to match the number of classes
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        # Load the model weights
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        # Set model to evaluation mode
        model.eval()

        return model

    def _load_all_models(self) -> None:
        """
        Load all the models for the different parts of the transcendence minigame.

        :return:
        """
        model_dirs = [
            'tiles',
            'cards',
            'tries',
            'changes',
            'level',
            'grace',
        ]

        for model_dir in model_dirs:
            self.class_names[model_dir] = os.listdir(f"dataset/{model_dir}/train")
            num_classes = len(self.class_names[model_dir])
            self.models[model_dir] = self._load_model(
                f'models/{model_dir}/best_tile_classifier.pth',
                num_classes
            )

    @staticmethod
    def _save_screenshot(image: Image, filename: str, directory: str = "screenshots") -> None:
        """
        Save the screenshot to a file for debugging or training purposes.

        :param image: The PIL image to save.
        :param filename: The name of the file to save the image as.
        :param directory: The directory to save the screenshot in. Defaults to 'screenshots'.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.save(os.path.join(directory, filename))

    def _process_screenshot(self, screenshot: Image, area: ScreenshotArea, model: torch.nn.Module,
                            class_names: list[str],
                            screenshot_type: str = "general", screenshot_dir: str = "screenshots",
                            save: bool = True) -> Prediction:
        """
        Process the screenshot based on the given area, model, and class names.

        :param screenshot: The screenshot to process.
        :param area: The area to crop from the screenshot.
        :param model: The model to use for prediction.
        :param class_names: The class names for the model.
        :return:
        """
        # move calculations to the GPU if available, otherwise use the CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # crop the image based on the area and convert it to RGB
        cropped_image = screenshot.crop(area.get_field()).convert('RGB')

        # Optionally save the cropped image for testing/training
        if save:
            self._save_screenshot(
                cropped_image,
                f"{screenshot_type}_screenshot_{int(time.time() * 1000)}.png",
                screenshot_dir
            )

        # transform the image and add a batch dimension before moving it to the device
        image_tensor = self.transform(cropped_image).unsqueeze(0).to(device)

        with torch.no_grad():
            # get the output from the model and calculate the probabilities
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            # get the predicted class and confidence
            confidence, predicted_class_idx = torch.max(probabilities, 1)
            return Prediction(confidence.item(), class_names[predicted_class_idx.item()])

    def _check_card(self, screenshot: Image, card: Card) -> Prediction:
        """
        Check the card based on the given screenshot and card area.

        :param screenshot: The screenshot to process.
        :param card: The card area to crop from the screenshot.
        :return:
        """
        return self._process_screenshot(
            screenshot,
            card,
            self.models['cards'],
            self.class_names['cards'],
            screenshot_type=f"card_{card.position}",
            screenshot_dir=os.path.join(self.screenshot_dir, 'cards'),
            save=self.save_screenshots
        )

    def _check_row(self, screenshot: Image, row: Row) -> list[Prediction]:
        """
        Check the row based on the given screenshot and row area.

        :param screenshot: The screenshot to process.
        :param row: The row area to crop from the screenshot.
        :return: A list of predictions for each tile in the row.
        """
        predictions = []
        for i in range(row.tile_count):
            # Calculate the specific area for each tile in the row
            field_left = row.top_left[0] + i * row.width
            field_right = row.top_left[0] + (i + 1) * row.width
            # Create the tile area based on the calculated field
            # The top left is the same y coordinate as the row and the bottom right is the same y coordinate as the row
            tile_area = ScreenshotArea((int(field_left), row.top_left[1]), (int(field_right), row.bot_right[1]))

            # Append the prediction for the tile area
            predictions.append(
                self._process_screenshot(screenshot, tile_area, self.models['tiles'], self.class_names['tiles'],
                                         screenshot_type=f"row_{row.row}_tile_{i}",
                                         screenshot_dir=os.path.join(self.screenshot_dir, 'tiles'),
                                         save=self.save_screenshots)
            )

        return predictions

    def _get_current_equipment_part(self) -> str:
        """
        Get the current equipment part based on where on the screen we find the image.

        :return: The current equipment part.
        """
        try:
            selection = pyautogui.locateOnScreen(
                f"{self.script_dir}/assets/transcendence/current_equipment_selection.png",
                confidence=0.9, region=(60, 80, 150 - 60, 700 - 80))
            if selection:
                # the selection is based on the y-axis of the image
                # the top of the image is at 100 (helmet), the bottom at 700 (weapon)
                return [
                    'helmet',
                    'shoulders',
                    'chestpiece',
                    'pants',
                    'gloves',
                    'weapon'
                ][((selection.top + selection.height // 2) // 100) - 1]
        except pyautogui.ImageNotFoundException:
            return ''

    @staticmethod
    def _get_rows_based_on_level(level: int) -> list[Row]:
        """
        Get the rows based on the current level.

        :param level: The current transcendence level.
        :return: A list of rows based on the current level.
        """
        complexities = {
            6: [Row((786, 357), (1138, 391), 6, 1),
                Row((782, 392), (1143, 426), 6, 2),
                Row((778, 428), (1147, 465), 6, 3),
                Row((773, 467), (1152, 505), 6, 4),
                Row((768, 507), (1157, 547), 6, 5),
                Row((763, 549), (1162, 592), 6, 6)],

            7: [Row((759, 349), (1172, 381), 7, 1),
                Row((754, 385), (1177, 418), 7, 2),
                Row((750, 421), (1181, 457), 7, 3),
                Row((745, 460), (1186, 498), 7, 4),
                Row((741, 500), (1191, 541), 7, 5),
                Row((736, 543), (1196, 587), 7, 6)],

            8: [Row((727, 337), (1207, 364), 8, 1),
                Row((723, 367), (1212, 395), 8, 2),
                Row((718, 398), (1217, 428), 8, 3),
                Row((713, 431), (1222, 464), 8, 4),
                Row((708, 465), (1227, 502), 8, 5),
                Row((703, 502), (1232, 542), 8, 6)]
        }

        # transcendence levels 1,2 and 3 have a complexity of 6 tiles per row
        if level <= 3:
            return complexities.get(6, [])
        # transcendence levels 4 and 5 have a complexity of 7 tiles per row
        elif level <= 5:
            return complexities.get(7, [])
        # transcendence levels 6 and 7 have a complexity of 8 tiles per row
        else:
            return complexities.get(8, [])

    def get_current_information(self) -> TranscendenceInfo:
        """
        Get the current transcendence information from the current screen.

        :return: The current transcendence information.
        """
        cards = [
            Card((434, 951), (483, 1014), 5),
            Card((514, 951), (563, 1014), 4),
            Card((594, 918), (670, 1014), 3),
            Card((735, 787), (888, 1018), 2),
            Card((1029, 787), (1181, 1018), 1)
        ]

        areas = {
            'tries': ScreenshotArea((1066, 726), (1173, 750)),
            'changes': ScreenshotArea((902, 1046), (1013, 1076)),
            'level': ScreenshotArea((772, 20), (1136, 45)),
            'grace': ScreenshotArea((414, 44), (476, 67))
        }

        start_time = time.time()

        # take a screenshot of the current board to analyze (faster than taking a screenshot for each tile)
        screenshot = pyautogui.screenshot()
        current_info = {
            key: self._process_screenshot(
                screenshot,
                areas[key],
                self.models[key],
                self.class_names[key],
                screenshot_type=key,
                screenshot_dir=os.path.join(self.screenshot_dir, key),
                save=self.save_screenshots
            ) for key in areas
        }
        current_cards = {card.position: self._check_card(screenshot, card) for card in cards[::-1]}
        current_equipment_part = self._get_current_equipment_part()
        rows = self._get_rows_based_on_level(int(current_info['level'].prediction))
        board = {row.row: self._check_row(screenshot, row) for row in rows}

        return TranscendenceInfo(
            gear_part=current_equipment_part,
            level=current_info['level'],
            grace=current_info['grace'],
            retries=current_info['tries'],
            changes=current_info['changes'],
            cards=current_cards,
            board=board,
            duration=time.time() - start_time
        )


if __name__ == '__main__':
    info = Transcendence(True).get_current_information()
    pprint(info)
