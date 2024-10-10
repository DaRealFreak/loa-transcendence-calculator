import logging
import os
import shutil
import time
from os.path import realpath, dirname

from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.firefox import GeckoDriverManager

from calculator import TranscendenceInfo, Prediction, Transcendence


class Interaction:
    warning: str = ''
    card: int = None
    reset_recommended: bool = False


class Change(Interaction):
    def __init__(self, warning: str, card: int, probability: str, reset_recommended: bool):
        self.warning = warning
        self.card = card
        self.probability = probability
        self.reset_recommended = reset_recommended

    def __str__(self):
        return f"Change(card={self.card}, probability={self.probability}, warning={self.warning}, " \
               f"reset_recommended={self.reset_recommended})"

    def __repr__(self):
        return str(self)


class Use(Interaction):
    def __init__(self, warning: str, row: int, column: int, card: int, card_name:str,
                 probability: str, reset_recommended: bool):
        self.warning = warning
        self.row = row
        self.column = column
        self.card = card
        self.card_name = card_name
        self.probability = probability
        self.reset_recommended = reset_recommended

    def __str__(self):
        return f"Use(card={self.card}, card_name={self.card_name}, row={self.row}, column={self.column}, " \
               f"probability={self.probability}, warning={self.warning}, reset_recommended={self.reset_recommended})"

    def __repr__(self):
        return str(self)


class Elphago:
    # Define the local bin directory path
    BIN_DIR = os.path.join(realpath(dirname(__file__)), 'bin')
    GECKODRIVER_PATH = os.path.join(BIN_DIR, 'geckodriver')
    ELPHAGO_URL = "https://cho.elphago.work/en"

    def __init__(self, headless: bool = True):
        """
        Initialize the Elphago object and open the Elphago website.

        :param headless: Whether to run the WebDriver in headless mode (default is True).
        """
        # Set up a logger
        self.logger = logging.getLogger(__name__)

        self.headless = headless

        # Set up the Firefox WebDriver
        self._setup_firefox()
        self.driver.get(self.ELPHAGO_URL)

        # Toggle the checkbox and click the confirm button on the page
        self._toggle_checkbox_and_confirm()

        self._last_info = None
        self.current_gear_part = ''
        self.current_level = 0
        self.current_grace = 0

    def __del__(self):
        # Quit the WebDriver when the object is deleted
        self.driver.quit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Quit the WebDriver when exiting the context manager
        self.driver.quit()

    def _download_geckodriver_to_bin(self) -> None:
        """
        Download geckodriver using webdriver_manager and move it to the bin directory.
        """
        self.logger.info("Geckodriver not found locally. Downloading...")
        # Use webdriver_manager to download geckodriver
        gecko_path = GeckoDriverManager().install()

        # Ensure the bin directory exists
        os.makedirs(self.BIN_DIR, exist_ok=True)

        # Move the downloaded geckodriver to the bin folder
        shutil.copy(gecko_path, self.GECKODRIVER_PATH)
        self.logger.info(f"Geckodriver downloaded and saved to {self.GECKODRIVER_PATH}")

    def _setup_firefox(self) -> None:
        """
        Setup Firefox WebDriver using geckodriver, either from the local bin directory or downloading it on first run.
        """
        # Check if geckodriver is already in the bin folder
        if not os.path.exists(self.GECKODRIVER_PATH):
            # If geckodriver is not found, download it and place it in the bin folder
            self._download_geckodriver_to_bin()

        # Setup Firefox options
        firefox_options = Options()
        if self.headless:
            # Run in headless mode
            firefox_options.add_argument("--headless")

        # Start in 1920x1080 resolution
        firefox_options.add_argument("--width=1920")
        firefox_options.add_argument("--height=1080")

        # Initialize the WebDriver with the local geckodriver binary
        service = Service(self.GECKODRIVER_PATH)

        # Start Firefox with the specified driver and options
        self.driver = webdriver.Firefox(service=service, options=firefox_options)

        # Move the window to the left side of the screen and maximize it
        if not self.headless:
            self.driver.set_window_position(-1000, 0)
            self.driver.maximize_window()

    def _toggle_checkbox_and_confirm(self) -> bool:
        """
        Initial call of the page requires you to toggle a checkbox, that you understand that it's only a trained model.

        :return:
        """
        try:
            # Wait for the parent <div> to be present
            parent_div = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.bg-white.dark\\:bg-zinc-700"))
            )

            # Find the checkbox within the parent <div> and toggle it
            checkbox = parent_div.find_element(By.CSS_SELECTOR, "input[type='checkbox'][role='switch']")
            self.driver.execute_script("arguments[0].click();", checkbox)

            # Wait for the Confirm button inside the parent <div> and click it
            confirm_button = parent_div.find_element(
                By.XPATH,
                ".//button[contains(text(), '확인') or contains(text(), 'Confirm')]"
            )
            self.driver.execute_script("arguments[0].click();", confirm_button)
            self.logger.debug("Confirm button clicked.")
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            return False

        return True

    def _select_card(self, card_name: str, position: int) -> None:
        """
        Select the card on the Elphago website.

        :param position: The position of the card to select (1-5).
        :param card_name: The name of the card to select.
        """
        card_mapping = {
            'Thunder': 1,
            'Hellfire': 2,
            'Shockwave': 3,
            'TidalWave': 4,
            'Explosion': 5,
            'Tempest': 6,
            'Lightning': 7,
            'Earthquake': 8,
            'Purify': 9,
            'Tornado': 10,
            # level 2
            'Thunder2': 11,
            'Hellfire2': 12,
            'Shockwave2': 13,
            'TidalWave2': 14,
            'Explosion2': 15,
            'Tempest2': 16,
            'Lightning2': 17,
            'Earthquake2': 18,
            'Purify2': 19,
            'Tornado2': 20,
            # level 3
            'Thunder3': 21,
            'Hellfire3': 22,
            'Shockwave3': 23,
            'TidalWave3': 24,
            'Explosion3': 25,
            'Tempest3': 26,
            'Lightning3': 27,
            'Earthquake3': 28,
            'Purify3': 29,
            'Tornado3': 30,
            # special
            'Outburst': 31,
            'WorldTree': 32,
        }

        card_position_selector = {
            5: "div.border-1:nth-child(1)",
            4: "div.border-1:nth-child(2)",
            3: "div.border-1:nth-child(3)",
            2: "div.group:nth-child(4)",
            1: "div.group:nth-child(6)",
        }

        if card_name not in card_mapping:
            raise ValueError(f"Invalid card name: {card_name}")

        if position not in card_position_selector:
            raise ValueError(f"Invalid card position: {position}")

        if position < 3:
            # first 2 cards can be 2nd and 3rd stage, so the HTML structure is different
            element_selection = f'ul div:nth-child({card_mapping[card_name]}) li.p-1'
        else:
            element_selection = f'ul > li.p-1:nth-child({card_mapping[card_name]})'

        try:
            complete_selector = f"{card_position_selector[position]} {element_selection}"
            input_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, complete_selector))
            )
            self.driver.execute_script("arguments[0].click();", input_element)
            self.logger.debug(f"Card selected: {card_name} at position {position}")
        except Exception as e:
            self.logger.error(f"Error occurred while entering card name: {e}")

    def _select_board(self, gear_part: str, level: int, grace: int) -> None:
        """
        Select the gear part, level, and grace on the Elphago website.

        :param gear_part: The gear part to select (e.g., 'Weapon', 'Armor', 'Accessory').
        :param level: The level of the gear part (e.g., 1, 2, 3, 4, 5).
        :param grace: The grace
        """
        reset_required = False
        if self.current_gear_part != gear_part or self.current_level != level or self.current_grace != grace:
            reset_required = True

        if gear_part != self.current_gear_part:
            # select.my-2:nth-child(1)
            values = {
                'helmet': 0,
                'shoulders': 1,
                'chestpiece': 2,
                'pants': 3,
                'gloves': 4,
                'weapon': 5,
            }
            if gear_part.lower() not in values:
                raise ValueError(f"Invalid gear part: {gear_part}")

            try:
                select_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "select.my-2:nth-child(1)"))
                )

                # Use Select to interact with the dropdown and select the option with value "1"
                select_dropdown = Select(select_element)
                select_dropdown.select_by_value(str(values[gear_part.lower()]))
                self.logger.info(f"Gear part selected: {gear_part}")
                self.current_gear_part = gear_part
            except Exception as e:
                self.logger.error(f"Error occurred while changing select value: {e}")

        if level != self.current_level:
            try:
                # Wait for the <select> element with class 'my-2' and locate the second child
                select_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "select.my-2:nth-child(2)"))
                )

                # Use Select to interact with the dropdown and select the option with value "1"
                select_dropdown = Select(select_element)
                select_dropdown.select_by_value(str(level - 1))
                self.logger.info(f"Level selected: {level}")
                self.current_level = level
            except Exception as e:
                self.logger.error(f"Error occurred while changing select value: {e}")

        if grace != self.current_grace:
            try:
                select_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "select.my-2:nth-child(3)"))
                )

                # Use Select to interact with the dropdown and select the option with value "1"
                select_dropdown = Select(select_element)
                select_dropdown.select_by_value(str(grace))
                self.logger.info(f"Grace selected: {grace}")
                self.current_grace = grace
            except Exception as e:
                self.logger.error(f"Error occurred while changing select value: {e}")

        if reset_required:
            # Click the reset button to clear the current selection
            reset_button = self.driver.find_element(
                By.XPATH,
                ".//button[contains(text(), '초기화') or contains(text(), 'Reset')]"
            )
            self.driver.execute_script("arguments[0].click();", reset_button)
            self.logger.debug("Reset button clicked.")

    def _set_tries(self, tries: int) -> None:
        """
        Set the number of tries to complete the board on the Elphago website.

        :param tries: The number of tries to complete the board.
        """
        try:
            input_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    f".m-4 > div:nth-child(4) > div:nth-child(1) > div:nth-child(3) > div:nth-child({tries})"
                ))
            )

            self.driver.execute_script("arguments[0].click();", input_element)
            self.logger.info(f"Tries set to: {tries}")
        except Exception as e:
            self.logger.error(f"Error occurred while changing tries: {e}")

    def _set_changes(self, changes: int) -> None:
        """
        Set the number of changes of card changes on the Elphago website.

        :param changes: The number of changes of cards left.
        """
        try:
            input_element = WebDriverWait(self.driver, 1).until(
                EC.presence_of_element_located((
                    By.CSS_SELECTOR,
                    f"div.items-end:nth-child(5) > div:nth-child(1) > div:nth-child(3) > div:nth-child({changes + 1})"
                ))
            )

            self.driver.execute_script("arguments[0].click();", input_element)
            self.logger.info(f"Changes set to: {changes}")
        except Exception as parent_error:
            try:
                input_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((
                        By.CSS_SELECTOR,
                        f"div.items-end:nth-child(5) > div:nth-child(1) > div:nth-child(3) > div:last-child"
                    ))
                )

                self.driver.execute_script("arguments[0].click();", input_element)

                # Get the text content of the last div
                last_div_text = input_element.text
                self.logger.info(f"Changes set to the last available option: {last_div_text}")
            except Exception as e:
                self.logger.error(f"Error occurred while changing changes: {parent_error} and {e}")

    def _synchronize_board(self, board: dict[int | None, list[Prediction]]) -> None:
        """
        Synchronize the board with the given board state.

        :param board: The board state to sync with.
        """
        self.logger.debug("Synchronizing board...")

        field_mapping = {
            'Normal': 1,
            'None': 2,
            'Distorted': 3,
            'Addition': 4,
            'Blessing': 5,
            'Mystery': 6,
            'Enhancement': 7,
            'Clone': 8,
            'Relocation': 9,
        }

        for row, predictions in enumerate(board.values(), start=1):
            for tile, prediction in enumerate(predictions, start=1):
                tile_selector = f".border-separate > tbody > tr:nth-child({row}) > td:nth-child({tile}) > div > div"
                field_selector = f".z-20 > ul > li:nth-child({field_mapping[prediction.prediction]})"

                # Ensure the prediction is valid
                if prediction.prediction not in field_mapping:
                    raise ValueError(f"Invalid field prediction: {prediction.prediction}")

                try:
                    # Wait for the tile element
                    tile_element = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, tile_selector))
                    )

                    # Check if the tile has the 'hidden' class, if yes, skip this tile
                    parent_td = self.driver.find_element(
                        By.CSS_SELECTOR,
                        f".border-separate > tbody > tr:nth-child({row}) > td:nth-child({tile}) > div"
                    )
                    if 'hidden' in parent_td.get_attribute("class"):
                        self.logger.debug(f"Tile at row {row}, column {tile} is hidden. Skipping.")
                        continue

                    # Wait for the field element
                    field_element = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, field_selector))
                    )

                    # Click the visible tile and then the corresponding field
                    self.driver.execute_script("arguments[0].click();", tile_element)
                    self.driver.execute_script("arguments[0].click();", field_element)
                except Exception as e:
                    self.logger.error(f"Error occurred while changing tile at row {row}, column {tile}: {e}")

        self.logger.debug("Board synchronized.")

    def _find_red_border_cells(self) -> tuple:
        """
        Find the red border cells on the Elphago website.
        :return: The row and column index of the red border cell.
        """
        try:
            # Find all the td elements with class 'bg-red-500'
            red_border_cells = self.driver.find_elements(
                By.CSS_SELECTOR,
                'table.border-separate > tbody td.bg-red-500'
            )
            for cell in red_border_cells:
                # Find the closest parent <tr> (row)
                row_element = cell.find_element(By.XPATH, './ancestor::tr')

                # Get the row index by checking the position of the <tr> within the <tbody>
                row_index = len(row_element.find_elements(By.XPATH, 'preceding-sibling::tr')) + 1

                # Get the column index by checking the position of the <td> within the row
                column_index = len(cell.find_elements(By.XPATH, 'preceding-sibling::td')) + 1

                self.logger.debug(f"Element found at row {row_index}, column {column_index}")
                return row_index, column_index
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")

        return None, None

    def sync_transcendence_info(self, info: TranscendenceInfo, synchronize_board: bool = True) -> None:
        """
        Synchronize the board with the TranscendenceInfo object.

        :param synchronize_board: Whether to synchronize the board state.
        :param info: The TranscendenceInfo object to sync with.
        """
        self._last_info = info
        self._select_board(info.gear_part, int(info.level.prediction), int(info.grace.prediction))
        self._set_changes(int(info.changes.prediction))
        self._set_tries(int(info.tries.prediction))
        if synchronize_board:
            self._synchronize_board(info.board)
        for pos, card in info.cards.items():
            self._select_card(card.prediction, pos)
        # primarily for visibility purposes since our headless initial selection click opens the selection menu
        # so even though the card got properly selected all menu items are still open
        if not self.headless:
            for pos, card in info.cards.items():
                self._select_card(card.prediction, pos)

    def calculate(self) -> Interaction | None:
        """
        Calculate the best move.

        :return:
        """
        # Click the reset button to clear the current selection
        reset_button = self.driver.find_element(
            By.XPATH,
            ".//button[contains(text(), '계산하기') or contains(text(), 'Calculate')]"
        )
        self.driver.execute_script("arguments[0].click();", reset_button)
        self.logger.debug("Calculate button clicked.")

        # Wait for calculation:
        try:
            probability_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.ml-2:nth-child(2) > div:not(.hidden)'))
            )

            # Get the text content of the last div
            probability_text = probability_element.text.replace('\n', ' ')
            if probability_text.startswith('1st'):
                probability_text = probability_text[4:].strip()
            self.logger.info(f"Probability: {probability_text}")
        except Exception as e:
            self.logger.error(f"Error occurred while getting probability: {e}")
            return None

        try:
            WebDriverWait(self.driver, 1).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.m-4 > div.bg-red-700:not(.hidden)'))
            )
            is_reset_recommended = True
        except TimeoutException:
            is_reset_recommended = False

        warning = ''
        try:
            warning_element = WebDriverWait(self.driver, 1).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.mt-4.ml-2 > div.text-red-600'))
            )

            warning = warning_element.text
        except Exception as e:
            self.logger.error(f"Error occurred while getting warning: {e}")
            return None

        # Check both cards for probabilities and if change or use
        card_positions = {
            1: "div.group:nth-child(6)",
            2: "div.group:nth-child(4)",
        }

        row, column = self._find_red_border_cells()
        if row is not None and column is not None:
            self.logger.debug(f"Red border cell found at row {row}, column {column}")

        for pos, selector in card_positions.items():
            try:
                card_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )

                if 'border-red-600' in card_element.get_attribute("class"):
                    is_recommended_card = True
                else:
                    is_recommended_card = False

                is_change = 'Change' == card_element.find_element(By.CSS_SELECTOR, "span").text
                if is_recommended_card:
                    if row is not None and column is not None:
                        card_name = ''
                        if pos in self._last_info.cards:
                            card_name = self._last_info.cards[pos].prediction
                        return Use(warning, row, column, pos, card_name, probability_text, is_reset_recommended)
                    else:
                        if not is_change:
                            raise ValueError(f"Card {pos} is recommended but neither change nor use.")
                        return Change(warning, pos, probability_text, is_reset_recommended)
            except Exception as e:
                self.logger.error(f"Error occurred while getting card {pos}: {e}")

    def save_screenshot(self, filename: str = 'screenshot.png') -> None:
        """
        Save a screenshot of the current page.

        :param filename: The filename to save the screenshot as. Default is 'screenshot.png'.
        :return:
        """
        # Take a screenshot after interaction
        self.driver.save_screenshot(filename)
        self.logger.info(f"Screenshot saved as '{filename}'")


if __name__ == "__main__":
    start = time.time()
    transcendence_info = Transcendence().get_current_information()
    print(f"Time taken for transcendance info: {time.time() - start:.2f} seconds")

    # Initialize the Elphago object
    elp = Elphago(False)
    start = time.time()
    elp.sync_transcendence_info(transcendence_info)
    print(f"Time taken for sync: {time.time() - start:.2f} seconds")
    print(elp.calculate())

    elp.save_screenshot()
    elp.__del__()
