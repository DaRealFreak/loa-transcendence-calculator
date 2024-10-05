import logging
import re
import time

import pyautogui

from calculator import Transcendence, Row, TranscendenceInfo
from elphago import Elphago, Interaction, Use, Change


class Reset(Interaction):
    def __repr__(self):
        return 'Reset()'


class SelectNextLevel(Interaction):
    def __repr__(self):
        return 'SelectNextLevel()'


class Game:
    def __init__(self, auto_unlock_next_level: bool = True, patience: int = 1, reset_threshold: float = 0):
        """
        Initializes the game with the given settings.

        :param auto_unlock_next_level: if the next level should be automatically unlocked.
        :param patience: the number of recommended resets to tolerate before actually resetting the level.
        :param reset_threshold: the threshold probability to continue transcending if we're above the percentage even
                                if a reset is recommended.
        """
        self.auto_unlock_next_level = auto_unlock_next_level
        self.elphago = Elphago(False)
        self.calculator = Transcendence(save_screenshots=True)
        self.last_information: TranscendenceInfo | None = None

        self.resets_recommended = 0
        self.resetting_patience = patience
        self.resetting_threshold = reset_threshold

        # Possible gear parts to transcend in order of the game (left to right and top to bottom)
        self.last_seen_gear_part = ''
        self.last_seen_transcendence_level = 0
        self.possible_gear_parts = [
            'helmet',
            'shoulder',
            'chestpiece',
            'pants',
            'gloves',
            'weapon'
        ]

    @staticmethod
    def _focus_lostark_window() -> None:
        """
        Focuses the 'LostArk.exe' process to ensure it is the active window.
        """
        # try:
        #     # Find the LostArk.exe process using pywinauto
        #     app = pywinauto.Application().connect(title="LOST ARK (64-bit, DX11) v.2.32.4.1")
        #     print("LostArk process found.")
        #     app_dialog = app.top_window()
        #     app_dialog.set_focus()
        #     print("LostArk window focused.")
        # except pywinauto.findwindows.ElementNotFoundError:
        #     print("LostArk process not found.")
        # except Exception as e:
        #     print(f"Error focusing LostArk window: {e}")

    @staticmethod
    def _extract_probability(text: str) -> float:
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
            if self._extract_probability(interaction.probability) > self.resetting_threshold:
                print(f'Probability of {interaction.probability} is higher than threshold, '
                      f'overriding reset recommendation.')
                interaction.reset_recommended = False

        # Reset the resets recommended counter
        if not interaction.reset_recommended:
            self.resets_recommended = 0

        # Something most likely got wrongly detected, warn the user and exit
        if interaction.warning:
            print('Warning: ' + interaction.warning)
            print('Exiting, please check the game state.')
            return True

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
            center = (top_left + bot_right) // 2, (relevant_row.top_left[1] + relevant_row.bot_right[1]) // 2
            print(f'Clicking at coordinates {center}')

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

            print(f'Using card {interaction.card} at position {interaction.row}, {interaction.column}')
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
        self._click(x=885, y=622)
        self._click(x=887, y=631)
        time.sleep(0.25)
        # press okay button
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
            next_action = self.determine_move(refresh_board=refresh)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    game = Game()
    game.transcendence()
    input('Press Enter to exit...')
