import logging
import multiprocessing
import re
import time
from os.path import realpath, dirname, join

import keyboard
import psutil
import pyautogui
from pywinauto import Application

from calculator import Transcendence, Row, TranscendenceInfo
from elphago import Elphago, Interaction, Use, Change


class Reset(Interaction):
    def __repr__(self):
        return 'Reset()'


class SelectNextLevel(Interaction):
    def __repr__(self):
        return 'SelectNextLevel()'


class Game:
    def __init__(self, auto_unlock_next_level: bool = True, patience: int = 1, reset_threshold: float = 0,
                 save_screenshots: bool = False, headless: bool = True,
                 sleep_time_after_window_focus: int = 0, ignore_warnings: bool = False):
        """
        Initializes the game with the given settings.

        :param auto_unlock_next_level: if the next level should be automatically unlocked.
        :param patience: the number of recommended resets to tolerate before actually resetting the level.
        :param reset_threshold: the threshold probability to continue transcending if we're above the percentage even
                                if a reset is recommended.
        :param save_screenshots: if the screenshots should be saved for debugging purposes.
        :param headless: if the browser for elphago should be run in headless mode.
        :param sleep_time_after_window_focus: the time to sleep after bringing the window to the foreground.
        :param ignore_warnings: if warnings should be ignored and the game should continue
        """
        # Set up a logger
        self.script_dir = realpath(dirname(__file__))
        self.logger = logging.getLogger(__name__)

        # some main settings
        self.auto_unlock_next_level = auto_unlock_next_level
        self.sleep_time_after_window_focus = sleep_time_after_window_focus

        self.elphago = Elphago(headless)
        self.calculator = Transcendence(
            save_screenshots=save_screenshots,
            screenshot_dir=join(self.script_dir, 'screenshots')
        )
        self.last_information: TranscendenceInfo | None = None

        self.resets_recommended = 0
        self.resetting_patience = patience
        self.resetting_threshold = reset_threshold
        self.ignore_warnings = ignore_warnings

        # Possible gear parts to transcend in order of the game (left to right and top to bottom)
        self.last_seen_gear_part = ''
        self.last_seen_transcendence_level = 0
        self.possible_gear_parts = [
            'helmet',
            'shoulders',
            'chestpiece',
            'pants',
            'gloves',
            'weapon'
        ]

    def _bring_window_to_foreground(self, pid: int) -> None:
        """
        Brings the window of the given process ID to the foreground.

        :param pid: the process ID to bring to the foreground.
        :return:
        """
        try:
            # Use the 'win32' backend instead of 'uia' for better window control
            app = Application(backend="win32").connect(process=pid)
            # Get the main window
            window = app.top_window()

            if window.is_active():
                self.logger.debug(f"Process {pid} is already in the foreground.")
            else:
                # If not already focused, bring it to the foreground
                window.set_focus()
                # # Minimize and restore to ensure it's brought forward
                # window.minimize()
                # time.sleep(0.5)
                # window.restore()
                self.logger.info(f"Process {pid} brought to foreground.")
                if self.sleep_time_after_window_focus > 0:
                    time.sleep(self.sleep_time_after_window_focus)
        except Exception as e:
            self.logger.error(f"Could not bring process {pid} to foreground: {str(e)}")

    def _focus_lostark_window(self, search_term: str = "LOSTARK.exe") -> None:
        """
        Focuses the window of the Lost Ark game.

        :param search_term: the search term to look for in the process name.
        :return:
        """
        found_process = False
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if search_term in proc.info['name']:
                    found_process = True
                    self._bring_window_to_foreground(proc.info['pid'])
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        if not found_process:
            self.logger.error(f"Could not find process with search term '{search_term}'.")
            raise RuntimeError(f"Could not find process with search term '{search_term}'.")

    @staticmethod
    def _extract_probability(text: str) -> float:
        """
        Extracts the probability from the given text. Returns 0.0 if no percentage is found.

        :param text: the text to extract the probability from.
        :return:
        """
        match = re.search(r'(\d+(\.\d+)?)%', text)
        if match:
            return float(match.group(1))
        else:
            return 0.0

    def determine_move(self, refresh_board: bool = True) -> Interaction:
        """
        Determines the next move to make in the game based on elphago.

        :param refresh_board:
        :return:
        """
        self._focus_lostark_window()
        self.last_information = self.calculator.get_current_information()
        if self.last_information.gear_part:
            self.last_seen_transcendence_level = int(self.last_information.level.prediction)
            if self.last_information.gear_part != self.last_seen_gear_part:
                self.last_seen_gear_part = self.last_information.gear_part

        if self.last_information.flowers == 0:
            print('No flowers found on the screen, assume level got completed.')
            return SelectNextLevel()

        if not self.has_flowers_to_continue():
            return Reset()

        self.elphago.sync_transcendence_info(self.last_information, synchronize_board=refresh_board)
        return self.elphago.calculate()

    def has_flowers_to_continue(self) -> bool:
        """
        Checks if there are enough flowers to continue transcending.

        :return: if the current transcending has enough flowers to continue.
        """
        if self.last_information.flowers < 3:
            # go through the board and check if there is a blessing tile, only avoid resetting if there is one
            for row, predictions in enumerate(self.last_information.board.values(), start=1):
                for tile, prediction in enumerate(predictions, start=1):
                    # ignore blessing tile if we have less than 2 flowers
                    if prediction.prediction == 'Blessing' and self.last_information.flowers == 2:
                        return True
            # less than 3 flowers and no blessing tile found
            return False
        # more than or equal to 3 flowers
        return True

    def handle_interaction(self, interaction: Interaction) -> bool:
        """
        Handles the given interaction by performing the necessary actions in the game.

        :param interaction: The interaction to handle.
        :return: if the current transcending is finished.
        """
        # Handle reset threshold recommendation override
        if (isinstance(interaction, Use) or isinstance(interaction, Change)) and interaction.reset_recommended:
            if 0 < self.resetting_threshold < self._extract_probability(interaction.probability):
                print(f'Probability of {interaction.probability} is higher than threshold, '
                      f'overriding reset recommendation.')
                interaction.reset_recommended = False

        # Reset the resets recommended counter
        if not interaction.reset_recommended:
            self.resets_recommended = 0

        # Something most likely got wrongly detected, warn the user and exit
        if interaction.warning:
            print('Warning: ' + interaction.warning)
            self.elphago.save_screenshot(
                filename=f'current_board_{int(time.time() * 1000)}.png',
                folder=self.script_dir
            )

            if not self.ignore_warnings:
                print('Exiting, please check the game state.')
                return True
            else:
                print('Ignoring warning and continuing...')

        # Check if there are enough flowers to continue and we aren't in a SelectNextLevel interaction
        if not self.has_flowers_to_continue() and not isinstance(interaction, SelectNextLevel):
            print('Not enough flowers, resetting the level.')
            self.reset_level()
            return False

        self._focus_lostark_window()
        if isinstance(interaction, Use):
            if interaction.reset_recommended:
                self.resets_recommended += 1

                if self.resets_recommended > self.resetting_patience:
                    print(f'Resetting current level for gear part: {self.last_information.gear_part}')
                    self.reset_level()
                    return False
                else:
                    print('Reset is recommended, but patience is not reached yet. Continuing...')

            rows = self.calculator.get_rows_based_on_level(int(self.last_information.level.prediction))
            # Retrieve the relevant row
            relevant_row: Row = rows[interaction.row - 1]
            # Calculate the center coordinates of the card
            top_left = relevant_row.top_left[0] + relevant_row.width * (interaction.column - 1)
            bot_right = relevant_row.top_left[0] + relevant_row.width * interaction.column
            center = (
                int((top_left + bot_right) // 2),
                int((relevant_row.top_left[1] + relevant_row.bot_right[1]) // 2)
            )
            self.logger.debug(f'Clicking at coordinates {center}')

            if interaction.card == 1:
                self._click(x=1100, y=900)
                time.sleep(0.25)
                self._click(x=int(center[0]), y=int(center[1]))
            elif interaction.card == 2:
                self._click(x=815, y=900)
                time.sleep(0.25)
                self._click(x=int(center[0]), y=int(center[1]))
            else:
                raise ValueError(f'Invalid card number {interaction.card}')

            print(
                f'Using card {interaction.card_name} (row: {interaction.row}, column: {interaction.column}, pos: {interaction.card})'
            )
        elif isinstance(interaction, Change):
            if interaction.reset_recommended:
                print('Reset is recommended, but since changes are free, we will change the card')

            print(f'Changing card {interaction.card}')
            if interaction.card == 1:
                self._click(x=1115, y=1050)
            elif interaction.card == 2:
                self._click(x=815, y=1050)
            else:
                raise ValueError(f'Invalid card number {interaction.card}')
        elif isinstance(interaction, Reset):
            print(f'Resetting current level for gear part: {self.last_information.gear_part}')
            self.reset_level()
            return False
        elif isinstance(interaction, SelectNextLevel):
            if self.auto_unlock_next_level:
                if self.last_seen_transcendence_level == 7:
                    print('Level 7 reached, quitting transcendence.')
                    return True
                else:
                    print('Selecting next level if possible for gear part: ' + self.last_seen_gear_part)
                    self.select_next_level()
                    return False
            else:
                print('Auto unlock is disabled, quitting transcendence.')
                return True

        # sleep 2 second to allow the game to process the change
        time.sleep(2)
        return False

    @staticmethod
    def _click(x: int, y: int, button: str = 'left') -> None:
        """
        Helper function to clicks at the given coordinates.

        :param x: the x-coordinate to click at.
        :param y: the y-coordinate to click at.
        :param button: the button to click with. Default is 'left'.
        :return:
        """
        pyautogui.moveTo(x=x, y=y, duration=0.1)
        pyautogui.click(x=x, y=y, button=button)

    def reset_level(self) -> None:
        """
        Resets the current level for the gear part.

        :return:
        """
        self._click(x=1818, y=960)
        time.sleep(0.5)
        # check the checkbox for understanding the reset rules (was different positions depending on gold or tickets)
        possible_checkbox_positions = [
            (885, 622),
            (887, 631)
        ]
        for position in possible_checkbox_positions:
            self._click(x=position[0], y=position[1])
            # try to press okay button before checking the next checkbox
            self._click(x=901, y=669)

        # wait for menu animation
        time.sleep(1)

        # press okay button for Soundstone fragments
        self._click(x=958, y=674)
        time.sleep(0.25)

        x_coordinate = 246 + self.possible_gear_parts.index(self.last_information.gear_part) * 283
        self._click(x=x_coordinate, y=644)
        time.sleep(0.5)

        self._click(x=898, y=699)
        # wait for menu animation
        time.sleep(3)

        # Reset the resets recommended counter
        self.resets_recommended = 0

    def select_next_level(self) -> None:
        """
        Selects the next level if possible for the gear part.

        :return:
        """
        if self.last_seen_gear_part not in self.possible_gear_parts:
            self.logger.error(f'Invalid gear part {self.last_seen_gear_part}, please start while in the minigame.')
            raise ValueError(f'Invalid gear part {self.last_seen_gear_part}')

        # Sleep to let the finish animation play
        time.sleep(5)

        # Confirm the previous level
        self._click(x=962, y=1026)
        # Wait until we're in the selection menu again
        time.sleep(3)

        # Retrieve the Soundstone fragments from the previous level
        self._click(x=950, y=676)
        time.sleep(0.25)

        # Click on arrow to navigate to the next level
        x_coordinate = 354 + self.possible_gear_parts.index(self.last_seen_gear_part) * 283
        self._click(x=x_coordinate, y=535)
        time.sleep(0.25)

        # Click on the "Liberate" button
        x_coordinate = 246 + self.possible_gear_parts.index(self.last_seen_gear_part) * 283
        self._click(x=x_coordinate, y=644)
        time.sleep(0.5)

        # Confirm Dark Fire usage
        self._click(x=906, y=648)

        # wait for menu animation
        time.sleep(3)

    def transcendence(self) -> None:
        """
        Starts the transcending process.

        :return:
        """
        next_action = self.determine_move()
        while not self.handle_interaction(next_action):
            refresh = not isinstance(next_action, Change)
            next_level = isinstance(next_action, SelectNextLevel)
            next_action = self.determine_move(refresh_board=refresh)
            if next_level and isinstance(next_action, SelectNextLevel):
                print('SelectNextLevel interaction detected twice in a row, probably out of Dark Fires, exiting.')
                break


def start_game_process(**kwargs):
    """
    Starts the game process with the given keyword arguments.

    :param kwargs:
    :return:
    """
    # Set up logging configuration in the child process
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting game process...")

    game = Game(**kwargs)
    game.transcendence()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Available game keyword arguments
    game_kwargs = {
        'auto_unlock_next_level': True,
        'patience': 1,
        'reset_threshold': 10.0,
        'headless': True,
        'save_screenshots': True,
        'sleep_time_after_window_focus': 0,
        'ignore_warnings': True
    }

    # Set up process with target function `start_game_process` and pass the kwargs
    p = multiprocessing.Process(
        target=start_game_process,
        kwargs=game_kwargs,
        name="Transcendence"
    )
    p.start()

    start_time = time.time()
    while True:
        if not p.is_alive():
            print("process has terminated")
            break

        if keyboard.is_pressed('esc'):
            print("esc key pressed, terminating process")
            p.terminate()
            break

        if time.time() - start_time > 60 * 60:
            print("60 minutes have passed, terminating process")
            p.terminate()
            break

        # sleep for 0.1 seconds to avoid 100% CPU usage
        time.sleep(0.1)
