"""
================================================================================
Telegram Bot Notification Script
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-11-25
Description :
   This script sends notifications via a Telegram bot. It loads configuration
   from a .env file, including the bot token and chat ID. It supports sending
   multiple messages and handles long messages by splitting them into parts
   to comply with Telegram's 4096 character limit.

   Key features include:
      - Loading configuration from .env file
      - Sending messages to a specified Telegram chat
      - Handling long messages by splitting into parts
      - Error handling for message sending failures
      - Integration with sound notification system

Usage:
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

import atexit # For playing a sound when the program finishes
import asyncio # For asynchronous operations
import os # For environment variables and file operations
import platform # For getting the operating system name
from colorama import Style # For coloring the terminal
from dotenv import load_dotenv # For loading .env file
from telegram import Bot # For Telegram bot operations
from telegram.error import BadRequest # For handling Telegram errors

# Macros:
class BackgroundColors: # Colors for the terminal
   CYAN = "\033[96m" # Cyan
   GREEN = "\033[92m" # Green
   YELLOW = "\033[93m" # Yellow
   RED = "\033[91m" # Red
   BOLD = "\033[1m" # Bold
   UNDERLINE = "\033[4m" # Underline
   CLEAR_TERMINAL = "\033[H\033[J" # Clear the terminal

# Execution Constants:
VERBOSE = False # Set to True to output verbose messages

# Sound Constants:
SOUND_COMMANDS = {"Darwin": "afplay", "Linux": "aplay", "Windows": "start"} # The commands to play a sound for each operating system
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav" # The path to the sound file

# RUN_FUNCTIONS:
RUN_FUNCTIONS = {
   "Play Sound": True, # Set to True to play a sound when the program finishes
}

# Load environment variables
load_dotenv() # Load variables from .env file
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_API_KEY") # Get the Telegram bot token from environment variables
CHAT_ID = os.getenv("CHAT_ID") # Get the chat ID from environment variables

# Initialize bot
if TELEGRAM_BOT_TOKEN: # If the Telegram bot token is set
   bot = Bot(token=TELEGRAM_BOT_TOKEN) # Initialize the Telegram bot
else: # If the Telegram bot token is not set
   print(f"{BackgroundColors.RED}TELEGRAM_API_KEY not found in .env file.{Style.RESET_ALL}")
   bot = None # None

# Functions Definitions:

def verbose_output(true_string="", false_string=""):
   """
   Outputs a message if the VERBOSE constant is set to True.

   :param true_string: The string to be outputted if the VERBOSE constant is set to True.
   :param false_string: The string to be outputted if the VERBOSE constant is set to False.
   :return: None
   """

   if VERBOSE and true_string != "": # If the VERBOSE constant is set to True and the true_string is set
      print(true_string) # Output the true statement string
   elif false_string != "": # If the false_string is set
      print(false_string) # Output the false statement string

async def send_long_message(text, chat_id):
   """
   Sends a long message by splitting it into parts if it exceeds 4096 characters.

   :param text: The message text to send
   :param chat_id: The chat ID to send the message to
   :return: None
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Sending long message to chat ID {BackgroundColors.CYAN}{chat_id}{Style.RESET_ALL}") # Output the verbose message
   
   MAX_MESSAGE_LENGTH = 4096 # Maximum message length for Telegram
   parts = [text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)] # Split the text into parts
   if bot: # If the bot is initialized
      async with bot: # Use the bot context
         for part in parts: # Send each part
            try: # Try to send the message part
               await bot.send_message(chat_id=chat_id, text=part) # Send the message part
            except BadRequest as e: # Handle BadRequest error
               print(f"{BackgroundColors.RED}Failed to send message part: {str(e)}{Style.RESET_ALL}")
   else: # If the bot is not initialized
      print(f"{BackgroundColors.RED}Bot not initialized.{Style.RESET_ALL}")

async def run_bot(messages, chat_id):
   """
   Runs the bot to send messages.

   :param messages: List of message strings
   :param chat_id: The chat ID to send messages to
   :return: None
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Running Telegram bot to send messages to chat ID {BackgroundColors.CYAN}{chat_id}{Style.RESET_ALL}") # Output the verbose message

   text = "\n".join(messages) # Join messages into a single string
   await send_long_message(text, chat_id) # Send the long message

def verify_filepath_exists(filepath):
   """
   Verify if a file or folder exists at the specified path.

   :param filepath: Path to the file or folder
   :return: True if the file or folder exists, False otherwise
   """

   verbose_output(f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}") # Output the verbose message

   return os.path.exists(filepath) # Return True if the file or folder exists, False otherwise

def play_sound():
   """
   Plays a sound when the program finishes and skips if the operating system is Windows.

   :param: None
   :return: None
   """

   current_os = platform.system() # Get the current operating system
   if current_os == "Windows": # If the current operating system is Windows
      return # Do nothing

   if verify_filepath_exists(SOUND_FILE): # If the sound file exists
      if current_os in SOUND_COMMANDS: # If the platform.system() is in the SOUND_COMMANDS dictionary
         os.system(f"{SOUND_COMMANDS[current_os]} {SOUND_FILE}") # Play the sound
      else: # If the platform.system() is not in the SOUND_COMMANDS dictionary
         print(f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not in the {BackgroundColors.CYAN}SOUND_COMMANDS dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}")
   else: # If the sound file does not exist
      print(f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{SOUND_FILE}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}")

def main():
   """
   Main function.

   :param: None
   :return: None
   """

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Telegram Bot Notification{BackgroundColors.GREEN} program!{Style.RESET_ALL}", end="\n\n") # Output the welcome message

   if not TELEGRAM_BOT_TOKEN or not CHAT_ID: # If the Telegram bot token or chat ID is not set
      print(f"{BackgroundColors.RED}TELEGRAM_API_KEY or CHAT_ID not set in .env file.{Style.RESET_ALL}")
      return # Exit the program

   messages = [ # Test messages
      "Test message",
   ]

   if messages: # If there are messages to send
      asyncio.run(run_bot(messages, CHAT_ID)) # Run the bot to send messages
      print(f"{BackgroundColors.GREEN}Messages sent to Telegram chat.{Style.RESET_ALL}")

   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called when the program finishes

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """

   main() # Call the main function
