import logging
import time

import pyautogui
import pywinauto

from calculator import Transcendence, Row, TranscendenceInfo
from elphago import Elphago, Interaction, Use, Change


class Game:
    def __init__(self, patience: int = 1):
        self.elphago = Elphago(False)
        self.calculator = Transcendence(save_screenshots=True)
        self.last_information: TranscendenceInfo | None = None

        self.resets_recommended = 0
        self.resetting_patience = patience

    @staticmethod
    def _focus_lostark_window() -> None:
        """
        Focuses the 'LostArk.exe' process to ensure it is the active window.
        """
        try:
            # Find the LostArk.exe process using pywinauto
            app = pywinauto.Application().connect(title="LOST ARK (64-bit, DX11) v.2.32.4.1")
            print("LostArk process found.")
            app_dialog = app.top_window()
            app_dialog.set_focus()
            print("LostArk window focused.")
        except pywinauto.findwindows.ElementNotFoundError:
            print("LostArk process not found.")
        except Exception as e:
            print(f"Error focusing LostArk window: {e}")

    def determine_move(self) -> Interaction:
        # self._focus_lostark_window()
        self.last_information = self.calculator.get_current_information()
        self.elphago.sync_transcendence_info(self.last_information)
        return self.elphago.calculate()

    def handle_interaction(self, interaction: Interaction) -> bool:
        """
        Handles the given interaction by performing the necessary actions in the game.

        :param interaction: The interaction to handle.
        :return: if the current transcending is finished.
        """
        if not interaction.reset_recommended:
            # Reset the resets recommended counter
            self.resets_recommended = 0

        if interaction.warning:
            print('Warning: ' + interaction.warning)
            print('Exiting, please check the game state.')
            return True

        # self._focus_lostark_window()
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
        # check the checkbox for understanding the reset rules
        self._click(x=887, y=631)
        time.sleep(0.25)
        # press okay button
        self._click(x=901, y=669)
        # wait for menu animation
        time.sleep(1)

        # press okay button for soundstone fragments
        self._click(x=958, y=674)
        time.sleep(0.25)

        gear_parts = [
            'helmet',
            'shoulder',
            'chestpiece',
            'pants',
            'gloves',
            'weapon'
        ]
        x_coordinate = 246 + gear_parts.index(self.last_information.gear_part) * 283
        self._click(x=x_coordinate, y=644)
        time.sleep(0.5)

        self._click(x=898, y=699)
        # wait for menu animation
        time.sleep(3)

        # Reset the resets recommended counter
        self.resets_recommended = 0

    def transcendence(self):
        next_action = self.determine_move()
        while not self.handle_interaction(next_action):
            next_action = self.determine_move()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    game = Game()
    game.transcendence()
    input('Press Enter to exit...')
