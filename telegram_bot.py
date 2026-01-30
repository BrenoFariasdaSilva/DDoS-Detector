"""
================================================================================
Telegram Bot Notification Module (telegram_bot.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-11-25
Description :
    This module provides a TelegramBot class for sending notifications via a
    Telegram bot. It loads configuration from a .env file, including the bot
    token and chat ID. It supports sending multiple messages and handles long
    messages by splitting them into parts to comply with Telegram's 4096
    character limit.

    Key features include:
        - Loading configuration from .env file
        - Sending messages to a specified Telegram chat
        - Handling long messages by splitting into parts
        - Error handling for message sending failures
        - Integration with sound notification system (optional)

Usage:
    As a module:
        from telegram_bot import TelegramBot
        bot = TelegramBot()
        bot.send_messages(["Hello", "World"])

    Standalone:
        1. Create a .env file in the project root with TELEGRAM_API_KEY and CHAT_ID.
        2. Install dependencies: pip install python-telegram-bot python-dotenv
        3. Run the script: $ python telegram_bot.py
        4. Outputs are sent to the Telegram chat specified in .env.

Outputs:
    - Messages sent to Telegram chat (no local files generated)

TODOs:
    - Add support for sending images or files
    - Implement message queuing for batch processing
    - Add retry mechanism for failed sends
    - Support multiple chat IDs for different notifications

Dependencies:
    - Python >= 3.8
    - python-telegram-bot
    - python-dotenv
    - colorama

Assumptions & Notes:
    - .env file must be present with TELEGRAM_API_KEY and CHAT_ID
    - Bot must be added to the chat and have send message permissions
    - Sound notification is optional and follows project conventions
"""

import atexit  # For playing a sound when the program finishes
import asyncio  # For asynchronous operations
import os  # For environment variables and file operations
import platform  # For getting the operating system name
from colorama import Style  # For coloring the terminal
from dotenv import load_dotenv  # For loading .env file
from telegram import Bot  # For Telegram bot operations
from telegram.error import BadRequest  # For handling Telegram errors


# Macros:
class BackgroundColors:  # Colors for the terminal
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    YELLOW = "\033[93m"  # Yellow
    RED = "\033[91m"  # Red
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline
    CLEAR_TERMINAL = "\033[H\033[J"  # Clear the terminal


# Execution Constants:
VERBOSE = False  # Set to True to output verbose messages

# Sound Constants:
SOUND_COMMANDS = {
    "Darwin": "afplay",
    "Linux": "aplay",
    "Windows": "start",
}  # The commands to play a sound for each operating system
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"  # The path to the sound file

# RUN_FUNCTIONS:
RUN_FUNCTIONS = {
    "Play Sound": True,  # Set to True to play a sound when the program finishes
}


class TelegramBot:
    """
    A class for sending notifications via Telegram bot.
    """

    def __init__(self, env_file=None):
        """
        Initialize the TelegramBot.

        :param env_file: Path to the .env file (optional, defaults to .env in current directory)
        """

        env_path = env_file if env_file else ".env"  # Determine the .env file path

        if not os.path.exists(env_path):  # Verify if the .env file exists
            print(
                f"{BackgroundColors.RED}Error: {BackgroundColors.CYAN}.env{BackgroundColors.RED} file not found at {env_path}.{Style.RESET_ALL}"
            )
            self.bot = None  # Set bot to None if .env file is missing
            return  # Exit the constructor

        load_dotenv(env_path)  # Load environment variables from .env file

        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_API_KEY")  # Get the Telegram bot token
        self.CHAT_ID = os.getenv("CHAT_ID")  # Get the chat ID

        missing_vars = []  # List to track missing variables
        if not self.TELEGRAM_BOT_TOKEN:  # If TELEGRAM_API_KEY is missing
            missing_vars.append("TELEGRAM_API_KEY")  # Add to missing variables list
        if not self.CHAT_ID:  # If CHAT_ID is missing
            missing_vars.append("CHAT_ID")  # Add to missing variables list

        if missing_vars:  # If there are missing variables
            print(
                f"{BackgroundColors.RED}Error: The following required variables were not found in {env_path}: {BackgroundColors.CYAN}{', '.join(missing_vars)}{BackgroundColors.RED}.{Style.RESET_ALL}"
            )
            self.bot = None  # Set bot to None if tokens are missing
        elif self.TELEGRAM_BOT_TOKEN and self.CHAT_ID:  # If both tokens are present
            self.bot = Bot(token=self.TELEGRAM_BOT_TOKEN)  # Initialize the Telegram bot
        else:  # If tokens are missing but no specific missing_vars identified (unlikely)
            print(f"{BackgroundColors.RED}Bot initialization failed due to configuration errors.{Style.RESET_ALL}")
            self.bot = None  # Set bot to None

    def _get_chat_id(self, chat_id):
        """
        Get the chat ID to use, defaulting to self.CHAT_ID if not provided.

        :param chat_id: The chat ID provided (can be None)
        :return: The chat ID to use
        """

        if chat_id is None:  # If no chat_id is provided
            return self.CHAT_ID  # Use the default chat ID
        return chat_id  # Return the provided chat_id

    async def send_message(self, text, chat_id=None):
        """
        Sends a message via Telegram bot.

        :param text: The message text to send
        :param chat_id: The chat ID to send the message to (optional, uses self.CHAT_ID if not provided)
        :return: None
        """

        chat_id = self._get_chat_id(chat_id)  # Get the chat ID to use

        if chat_id is None:  # If chat ID is not set
            print(f"{BackgroundColors.RED}Chat ID not set.{Style.RESET_ALL}")
            return  # Exit the function

        verbose_output(
            f"{BackgroundColors.GREEN}Sending message to chat ID {BackgroundColors.CYAN}{chat_id}{Style.RESET_ALL}"
        )  # Output the verbose message

        if self.bot:  # If the bot is initialized
            async with self.bot:  # Use the bot context
                await self.bot.send_message(text=text, chat_id=chat_id)  # Send the message
        else:  # If the bot is not initialized
            print(f"{BackgroundColors.RED}Bot not initialized.{Style.RESET_ALL}")

    async def send_long_message(self, text, chat_id=None):
        """
        Sends a long message by splitting it into parts if it exceeds 4096 characters.

        :param text: The message text to send
        :param chat_id: The chat ID to send the message to (optional, uses self.CHAT_ID if not provided)
        :return: None
        """

        chat_id = self._get_chat_id(chat_id)  # Get the chat ID to use

        if chat_id is None:  # If chat ID is not set
            print(f"{BackgroundColors.RED}Chat ID not set.{Style.RESET_ALL}")
            return  # Exit the function

        verbose_output(
            f"{BackgroundColors.GREEN}Sending long message to chat ID {BackgroundColors.CYAN}{chat_id}{Style.RESET_ALL}"
        )  # Output the verbose message

        MAX_MESSAGE_LENGTH = 4096  # Maximum message length for Telegram
        parts = [
            text[i : i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)
        ]  # Split the text into parts
        if self.bot:  # If the bot is initialized
            async with self.bot:  # Use the bot context
                for part in parts:  # Send each part
                    try:  # Try to send the message part
                        await self.bot.send_message(chat_id=chat_id, text=part)  # Send the message part
                    except BadRequest as e:  # Handle BadRequest error
                        print(f"{BackgroundColors.RED}Failed to send message part: {str(e)}{Style.RESET_ALL}")
        else:  # If the bot is not initialized
            print(f"{BackgroundColors.RED}Bot not initialized.{Style.RESET_ALL}")

    async def run_bot(self, messages, chat_id=None):
        """
        Runs the bot to send messages.

        :param messages: List of message strings
        :param chat_id: The chat ID to send messages to (optional, uses self.CHAT_ID if not provided)
        :return: None
        """

        chat_id = self._get_chat_id(chat_id)  # Get the chat ID to use

        if chat_id is None:  # If chat ID is not set
            print(f"{BackgroundColors.RED}Chat ID not set.{Style.RESET_ALL}")
            return  # Exit the function

        verbose_output(
            f"{BackgroundColors.GREEN}Running Telegram bot to send messages to chat ID {BackgroundColors.CYAN}{chat_id}{Style.RESET_ALL}"
        )  # Output the verbose message

        text = "\n".join(messages)  # Join messages into a single string
        await self.send_long_message(text, chat_id)  # Send the long message

    async def send_messages(self, messages, chat_id=None):
        """
        Asynchronous wrapper to send messages.

        :param messages: List of message strings or a single string
        :param chat_id: The chat ID to send messages to (optional)
        :return: None
        """

        if isinstance(messages, str):  # If a single string is provided
            messages = [messages]  # Convert it to a list

        if not self.TELEGRAM_BOT_TOKEN or not self.CHAT_ID:  # If the Telegram bot token or chat ID is not set
            print(f"{BackgroundColors.RED}TELEGRAM_API_KEY or CHAT_ID not set.{Style.RESET_ALL}")
            return  # Exit the function

        await self.run_bot(messages, chat_id)  # Run the bot to send messages


def verbose_output(true_string="", false_string=""):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None
    """

    if VERBOSE and true_string != "":  # If VERBOSE is True and a true_string was provided
        print(true_string)
    elif false_string != "":  # If a false_string was provided
        print(false_string)


def send_telegram_message(bot, messages, condition=True):
    """
    Sends a message via Telegram bot if configured and condition is met.

    :param bot: TelegramBot instance
    :param messages: List of messages to send
    :param condition: Additional condition to check before sending
    :return: None
    """

    if condition and bot.TELEGRAM_BOT_TOKEN and bot.CHAT_ID:  # If condition met and Telegram is configured
        try:  # Try to send message
            asyncio.run(bot.send_messages(messages))  # Run the async method synchronously
        except Exception:  # Silently ignore Telegram errors
            pass  # Do nothing


def verify_filepath_exists(filepath):
    """
    Verify if a file or folder exists at the specified path.

    :param filepath: Path to the file or folder
    :return: True if the file or folder exists, False otherwise
    """

    return os.path.exists(filepath)  # Return True if the file or folder exists, False otherwise


def play_sound():
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.

    :param: None
    :return: None
    """

    current_os = platform.system()  # Get the current operating system
    if current_os == "Windows":  # If the current operating system is Windows
        return  # Do nothing

    if verify_filepath_exists(SOUND_FILE):  # If the sound file exists
        if current_os in SOUND_COMMANDS:  # If the platform.system() is in the SOUND_COMMANDS dictionary
            os.system(f"{SOUND_COMMANDS[current_os]} {SOUND_FILE}")  # Play the sound
        else:  # If the platform.system() is not in the SOUND_COMMANDS dictionary
            print(
                f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not in the {BackgroundColors.CYAN}SOUND_COMMANDS dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}"
            )
    else:  # If the sound file does not exist
        print(
            f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{SOUND_FILE}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}"
        )


async def main():
    """
    Main function for standalone execution.

    :param: None
    :return: None
    """

    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Telegram Bot Notification{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
        end="\n\n",
    )  # Output the welcome message

    bot = TelegramBot()  # Initialize the Telegram bot
    if not bot.TELEGRAM_BOT_TOKEN or not bot.CHAT_ID:  # If the Telegram bot token or chat ID is not set
        print(f"{BackgroundColors.RED}TELEGRAM_API_KEY or CHAT_ID not set in .env file.{Style.RESET_ALL}")
        return  # Exit the program

    messages = [  # Test messages
        "Test message",
    ]

    if messages:  # If there are messages to send
        await bot.send_messages(messages)  # Send messages
        print(f"{BackgroundColors.GREEN}Messages sent to Telegram chat.{Style.RESET_ALL}")
    
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Output the end of the program message

    (
        atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None
    )  # Register the play_sound function to be called when the program finishes


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    asyncio.run(main())  # Call the main function
