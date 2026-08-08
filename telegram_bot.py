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
if __name__ in {"__main__", "__mp_main__"}:
    try:
        from setproctitle import setproctitle
        setproctitle(f"DDoS-{__file__.rsplit('/', 1)[-1].rsplit('.', 1)[0]}")
    except ImportError:
        pass


import atexit  # For playing a sound when the program finishes
import asyncio  # For asynchronous operations
import os  # For environment variables and file operations
import platform  # For getting the operating system name
import queue  # For thread-safe inbound Telegram message delivery
import re  # For stripping ANSI sequences
import socket  # For getting the local IP address
import sys  # For system-level hooks and excepthook manipulation
import threading  # For non-blocking inbound Telegram polling
import traceback  # For formatting and printing exception tracebacks
from colorama import Style  # For coloring the terminal
from dotenv import load_dotenv  # For loading .env file
from telegram import Bot  # For Telegram bot operations
from telegram.error import BadRequest  # For handling Telegram errors


# Telegram Configuration:
TELEGRAM_DEVICE_INFO = ""  # Device info for Telegram messages, set by calling script
RUNNING_CODE = ""  # Name of the running script, set by calling script
EXECUTION_ID = ""  # Stable top-level execution ID, set by calling script when available
TELEGRAM_BOT = None  # Optional module-level TelegramBot instance usable by the global handler

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
        self.TELEGRAM_BOT_TOKEN = None  # Default token state when configuration is missing
        self.CHAT_ID = None  # Default chat state when configuration is missing
        self.bot = None  # Default bot state before successful initialization
        self.inbound_messages = queue.Queue()  # Store validated inbound messages for application-owned consumers
        self._inbound_listener_thread = None  # Holds the single listener thread owned by this bot instance
        self._inbound_listener_stop_event = threading.Event()  # Signals the listener loop to stop
        self._inbound_listener_lock = threading.Lock()  # Prevents duplicate listener startup in one process
        self._inbound_listener_callback = None  # Optional application callback for validated inbound messages
        self._inbound_listener_owner_pid = None  # Tracks the process that owns the active listener
        self._last_inbound_update_id = None  # Deduplicates repeated Telegram API responses

        if not verify_filepath_exists(env_path):  # Verify if the .env file exists
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

    def get_chat_id(self, chat_id):
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

        chat_id = self.get_chat_id(chat_id)  # Get the chat ID to use

        if chat_id is None:  # If chat ID is not set
            print(f"{BackgroundColors.RED}Chat ID not set.{Style.RESET_ALL}")
            return  # Exit the function

        verbose_output(
            f"{BackgroundColors.GREEN}Sending message to chat ID {BackgroundColors.CYAN}{chat_id}{Style.RESET_ALL}"
        )  # Output the verbose message

        if self.bot:  # If the bot is initialized
            async with self.bot:  # Use the bot context
                await self.bot.send_message(text=text, chat_id=chat_id, parse_mode="MarkdownV2")  # Send the message with MarkdownV2 parse mode
        else:  # If the bot is not initialized
            print(f"{BackgroundColors.RED}Bot not initialized.{Style.RESET_ALL}")

    async def send_long_message(self, text, chat_id=None):
        """
        Sends a long message by splitting it into parts if it exceeds 4096 characters.

        :param text: The message text to send
        :param chat_id: The chat ID to send the message to (optional, uses self.CHAT_ID if not provided)
        :return: None
        """

        chat_id = self.get_chat_id(chat_id)  # Get the chat ID to use

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
                        await self.bot.send_message(chat_id=chat_id, text=part, parse_mode="MarkdownV2")  # Send the message part with MarkdownV2 parse mode
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

        chat_id = self.get_chat_id(chat_id)  # Get the chat ID to use

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

    def start_inbound_listener(self, callback=None, poll_timeout_seconds=10, retry_delay_seconds=5):
        """
        Start one non-blocking Telegram getUpdates listener for configured chat text messages.

        :param callback: Optional callable receiving one validated inbound message dictionary
        :param poll_timeout_seconds: Telegram long-poll timeout in seconds
        :param retry_delay_seconds: Delay after temporary Telegram/network failures
        :return: True if a listener is running, False otherwise
        """

        if not self.TELEGRAM_BOT_TOKEN or not self.CHAT_ID or self.bot is None:  # Require the same configured bot used by outbound messages
            return False  # Cannot listen without configured Telegram credentials
        with self._inbound_listener_lock:  # Serialize startup so one process cannot create duplicate pollers
            if self._inbound_listener_thread is not None and self._inbound_listener_thread.is_alive():  # Listener already active
                return True  # Report the existing listener
            self._inbound_listener_callback = callback  # Store application delivery callback
            self._inbound_listener_stop_event.clear()  # Reset stop signal before starting
            self._inbound_listener_owner_pid = os.getpid()  # Own listener in the creating process only
            self._inbound_listener_thread = threading.Thread(
                target=self._run_inbound_listener,
                args=(poll_timeout_seconds, retry_delay_seconds),
                name="telegram-inbound-listener",
                daemon=True,
            )  # Create one daemon thread so experiment shutdown cannot be blocked forever by network I/O
            self._inbound_listener_thread.start()  # Start polling without blocking the experiment
            return True  # Report that the listener is active

    def stop_inbound_listener(self, timeout_seconds=12):
        """
        Stop the inbound Telegram listener owned by this process.

        :param timeout_seconds: Maximum join time for the listener thread
        :return: None
        """

        with self._inbound_listener_lock:  # Serialize shutdown with startup
            thread = self._inbound_listener_thread  # Read current listener thread
            if thread is None:  # No listener to stop
                return  # Nothing to do
            if self._inbound_listener_owner_pid != os.getpid():  # Do not stop a listener owned by another process
                return  # Preserve process ownership boundary
            self._inbound_listener_stop_event.set()  # Ask polling loop to stop after current request
        if thread is not threading.current_thread():  # Avoid joining the current listener thread
            thread.join(timeout_seconds)  # Wait for bounded long-poll shutdown
        with self._inbound_listener_lock:  # Clear stopped thread state
            if self._inbound_listener_thread is thread and not thread.is_alive():  # Clear only the stopped listener
                self._inbound_listener_thread = None  # Mark listener stopped
                self._inbound_listener_owner_pid = None  # Clear owner identity

    def get_inbound_message(self, timeout=None):
        """
        Read one validated inbound Telegram message from the listener queue.

        :param timeout: Optional queue timeout in seconds, None blocks until one message is available
        :return: Message dictionary, or None when no message is available before timeout
        """

        try:  # Read without exposing queue exceptions to callers
            if timeout is None:  # Blocking read
                return self.inbound_messages.get()  # Return one validated inbound message
            return self.inbound_messages.get(timeout=timeout)  # Return one message within timeout
        except queue.Empty:  # No message available before timeout
            return None  # Preserve narrow, simple consumer API

    def drain_inbound_messages(self):
        """
        Return all currently queued validated inbound Telegram messages.

        :return: List of message dictionaries
        """

        messages = []  # Accumulate currently queued messages
        while True:  # Drain without blocking
            try:  # Read one queued message
                messages.append(self.inbound_messages.get_nowait())  # Add message to result list
            except queue.Empty:  # Queue drained
                return messages  # Return all available messages

    def _run_inbound_listener(self, poll_timeout_seconds, retry_delay_seconds):
        """
        Run the async Telegram polling loop inside the listener thread.

        :param poll_timeout_seconds: Telegram long-poll timeout in seconds
        :param retry_delay_seconds: Delay after temporary Telegram/network failures
        :return: None
        """

        try:  # Keep listener failures isolated from experiments
            asyncio.run(self._poll_inbound_messages(poll_timeout_seconds, retry_delay_seconds))  # Run async getUpdates loop in this thread
        except Exception as e:  # Listener thread must never crash the experiment
            print(f"{BackgroundColors.YELLOW}Telegram inbound listener stopped: {self._safe_inbound_error(e)}{Style.RESET_ALL}")  # Report listener failure without token data

    async def _poll_inbound_messages(self, poll_timeout_seconds, retry_delay_seconds):
        """
        Poll Telegram updates, validate configured chat, and deliver text messages.

        :param poll_timeout_seconds: Telegram long-poll timeout in seconds
        :param retry_delay_seconds: Delay after temporary Telegram/network failures
        :return: None
        """

        polling_bot = Bot(token=self.TELEGRAM_BOT_TOKEN)  # Use a separate bot instance so outbound sends keep their existing context behavior
        safe_poll_timeout = max(1, int(poll_timeout_seconds))  # Avoid tight polling loops
        safe_retry_delay = max(1, int(retry_delay_seconds))  # Avoid tight retry loops on network failures
        async with polling_bot:  # Initialize and close the polling bot in this listener thread
            discard_ok, update_offset = await self._discard_pending_inbound_updates(polling_bot)  # Avoid replaying historical startup backlog
            while not discard_ok and not self._inbound_listener_stop_event.is_set():  # Never process backlog if startup discard failed
                await asyncio.sleep(safe_retry_delay)  # Back off before retrying startup discard
                discard_ok, update_offset = await self._discard_pending_inbound_updates(polling_bot)  # Retry until backlog state is known
            while not self._inbound_listener_stop_event.is_set():  # Continue until owner asks for shutdown
                try:  # Isolate every Telegram API poll
                    updates = await polling_bot.get_updates(offset=update_offset, timeout=safe_poll_timeout, allowed_updates=["message"])  # Long-poll text message updates
                except Exception as e:  # Temporary Telegram/network failure
                    print(f"{BackgroundColors.YELLOW}Telegram inbound polling failed: {self._safe_inbound_error(e)}{Style.RESET_ALL}")  # Report without token data
                    await asyncio.sleep(safe_retry_delay)  # Back off before retrying
                    continue  # Keep experiments unaffected
                for update in updates:  # Process each returned update once
                    update_id = getattr(update, "update_id", None)  # Read Telegram update identity
                    if update_id is not None and self._last_inbound_update_id is not None and int(update_id) <= int(self._last_inbound_update_id):  # Skip duplicate API responses
                        continue  # Do not deliver duplicate update
                    self._last_inbound_update_id = update_id  # Mark update consumed even if chat is unrelated
                    if update_id is not None:  # Advance offset past every seen update, including ignored chats
                        update_offset = int(update_id) + 1  # Prevent repeated processing
                    inbound_message = self._build_inbound_message(update)  # Validate chat and text content
                    if inbound_message is None:  # Ignore unrelated chats or non-text messages
                        continue  # Move to next update
                    self._deliver_inbound_message(inbound_message)  # Expose validated message without interpreting it

    async def _discard_pending_inbound_updates(self, polling_bot):
        """
        Drop historical pending updates on listener startup and return the next offset.

        :param polling_bot: Telegram Bot instance used for polling
        :return: Tuple of discard success flag and update offset for the next polling request
        """

        try:  # Best-effort backlog discard before normal polling
            updates = await polling_bot.get_updates(offset=-1, limit=1, timeout=0, allowed_updates=["message"])  # Ask Telegram for only the newest queued update
        except Exception as e:  # Startup discard failure should not stop listener
            print(f"{BackgroundColors.YELLOW}Telegram inbound backlog discard failed: {self._safe_inbound_error(e)}{Style.RESET_ALL}")  # Report without token data
            return False, None  # Retry discard before normal polling
        if not updates:  # No pending backlog
            return True, None  # Start normal polling without offset
        update_id = getattr(updates[-1], "update_id", None)  # Read newest update id
        if update_id is None:  # Defensive fallback for unexpected Telegram object shape
            return True, None  # Start normal polling without offset
        self._last_inbound_update_id = int(update_id)  # Do not deliver the startup backlog update
        return True, int(update_id) + 1  # Start after newest queued update

    def _build_inbound_message(self, update):
        """
        Convert one Telegram update into a validated inbound message dictionary.

        :param update: Telegram Update object
        :return: Message dictionary, or None when update is not a configured-chat text message
        """

        message = getattr(update, "effective_message", None) or getattr(update, "message", None)  # Resolve message payload across python-telegram-bot versions
        if message is None:  # Ignore non-message updates
            return None  # No application delivery
        chat = getattr(message, "chat", None)  # Resolve chat metadata
        chat_id = getattr(chat, "id", None)  # Resolve chat id
        if str(chat_id) != str(self.CHAT_ID):  # Validate configured chat exactly
            return None  # Ignore unrelated chats
        text = getattr(message, "text", None)  # Receive text messages only
        if not text:  # Ignore non-text or empty messages
            return None  # No application delivery
        from_user = getattr(message, "from_user", None)  # Capture sender metadata without requiring it
        return {
            "update_id": getattr(update, "update_id", None),
            "chat_id": chat_id,
            "message_id": getattr(message, "message_id", None),
            "date": getattr(message, "date", None),
            "text": text,
            "from_user_id": getattr(from_user, "id", None),
            "from_username": getattr(from_user, "username", None),
        }  # Return narrow validated inbound message payload

    def _deliver_inbound_message(self, inbound_message):
        """
        Deliver one validated inbound message through queue and optional callback.

        :param inbound_message: Validated inbound message dictionary
        :return: None
        """

        self.inbound_messages.put(inbound_message)  # Preserve message for application polling
        callback = self._inbound_listener_callback  # Read optional callback
        if callback is None:  # No callback registered
            return  # Queue delivery is enough for this step
        try:  # Keep callback failures isolated from Telegram polling and experiments
            callback(inbound_message)  # Notify application-owned callback
        except Exception as e:  # Callback failed
            print(f"{BackgroundColors.YELLOW}Telegram inbound callback failed: {self._safe_inbound_error(e)}{Style.RESET_ALL}")  # Report without stopping polling

    def _safe_inbound_error(self, error):
        """
        Return an inbound-listener error string with Telegram token redacted.

        :param error: Exception or message object.
        :return: Safe message string.
        """

        message = str(error)  # Preserve existing concise error reporting.
        token = str(self.TELEGRAM_BOT_TOKEN or "")  # Resolve configured token without logging it.
        return message.replace(token, "[redacted]") if token else message  # Remove token if any dependency included it.


def verbose_output(true_string="", false_string=""):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None
    """

    if VERBOSE and true_string != "":  # If VERBOSE is True and a true_string was provided
        print(true_string)  # Output the true statement string
    elif false_string != "":  # If a false_string was provided
        print(false_string)  # Output the false statement string


def get_local_ip():
    """
    Get the local IP address of the device in a cross-platform way.

    :return: Local IP address as string
    """
    
    verbose_output(f"Attempting to get local IP address...")  # Output the verbose message
    
    try:  # Try to get the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Use UDP socket
        s.connect(("8.8.8.8", 80))  # Connect to Google's DNS
        ip = s.getsockname()[0]  # Get the local IP address
        s.close()  # Close the socket
        return ip  # Return the local IP address
    except:  # If any error occurs
        return "127.0.0.1"  # Fallback to localhost


def strip_ansi(text: str) -> str:
    """
    Strips ANSI escape sequences from the given text.
    
    :param text: The text to strip ANSI sequences from
    :return: The text without ANSI sequences
    """
    
    try:  # Try to strip ANSI sequences
        text = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)  # Remove ANSI escape sequences
        text = re.sub(r"\[\d+(?:;\d+)*m", "", text)  # Remove color codes
        return text  # Return the cleaned text
    except Exception:  # If any error occurs
        return text  # Return the original text


def escape_markdown_v2(text: str) -> str:
    """
    Escape Telegram MarkdownV2 special characters in `text` by prefixing
    them with a backslash so Telegram will not interpret them as formatting.

    Characters escaped: _ * [ ] ( ) ~ ` > # + - = | { } . !

    :param text: Input string
    :return: Escaped string safe for MarkdownV2
    """

    try:  # Try to escape special characters
        return re.sub(r"([_\*\[\]\(\)~`>\#\+\-\=\|\{\}\.\!])", r"\\\1", str(text))  # Escape the set of special characters for MarkdownV2
    except Exception:  # If any error occurs
        return text  # Return the original text


def send_telegram_message(bot, messages, condition=True):
    """
    Sends a message via Telegram bot if configured and condition is met.

    :param bot: TelegramBot instance
    :param messages: List of messages to send
    :param condition: Additional condition to verify before sending
    :return: None
    """

    bot_token = getattr(bot, "TELEGRAM_BOT_TOKEN", None) if bot is not None else None  # Get bot token
    chat_id_val = getattr(bot, "CHAT_ID", None) if bot is not None else None  # Get chat ID
    if condition and bot is not None and bot_token and chat_id_val:  # If condition met and Telegram is configured
        try:  # Try to send message
            if isinstance(messages, str):  # If a single string is provided
                messages = [messages]  # Convert it to a list
            
            execution_suffix = f" - Execution ID: {str(EXECUTION_ID)}" if EXECUTION_ID else ""  # Preserve old prefix when no execution ID is set
            prefixed_messages = [  # Prefix each message with device info and running code
                escape_markdown_v2(
                    strip_ansi(
                        f"{str(TELEGRAM_DEVICE_INFO)} - {str(RUNNING_CODE)}{execution_suffix}: {str(msg)}"
                    )
                )
                for msg in messages
            ]
            asyncio.run(bot.send_messages(prefixed_messages))  # Run the async method synchronously
        except Exception:  # Silently ignore Telegram errors
            pass  # Do nothing


def send_exception_via_telegram(exc_type, exc_value, exc_tb):  # Custom exception handler exposed for global use
    """
    Custom exception handler that sends uncaught exceptions to Telegram.
    
    :param exc_type: Exception type
    :param exc_value: Exception value
    :param exc_tb: Exception traceback object
    :return: None
    """
    
    try:  # Attempt to format the traceback into a string for the message
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))  # Traceback string
    except Exception as e:  # Formatting may fail, provide safe fallback and notify minimally
        print(str(e))  # Print formatting error to terminal for visibility
        try:  # Try to inform via Telegram about the formatting failure as a minimal message
            send_telegram_message(TELEGRAM_BOT, f"Failed to format traceback: {e}")  # Minimal notify
        except Exception:  # Sending may fail, but must not raise further
            pass  # Ignore notification failures to avoid recursion
        tb = "Could not format traceback."  # Fallback traceback string when formatting fails

    msg = f"Uncaught exception in {RUNNING_CODE}:\n{exc_value}\n\nTraceback:\n{tb}"  # Construct notification message including running code context

    try:  # Attempt to ensure bot is initialized in this module before sending
        if "TELEGRAM_BOT" in globals() and TELEGRAM_BOT is None:  # If module bot variable exists but not set
            try:  # Try to initialize or at least attempt a setup hint (do not import-circular)
                pass  # No-op placeholder: actual bot initialization controlled by caller modules
            except Exception as e:  # If any problem arises attempting to initialize, print for diagnostics
                print(str(e))  # Print bot setup error to terminal for diagnostics
                try:  # Attempt to notify via any available bot instance about setup failure
                    send_telegram_message(TELEGRAM_BOT, f"Failed to initialize bot: {e}")  # Minimal notify
                except Exception:  # If notification fails, swallow to avoid recursion
                    pass  # Ignore further errors
        send_telegram_message(TELEGRAM_BOT, msg)  # Send the detailed exception message to Telegram
    except Exception as e:  # If sending fails, ensure terminal still receives full diagnostics
        try:  # Try to print an error marker to stderr for immediate visibility
            print(f"{BackgroundColors.RED}Failed to send telegram message: {e}{Style.RESET_ALL}", file=sys.stderr)  # Print to stderr
            traceback.print_exc()  # Print the internal send failure traceback to stderr
        except Exception as e:  # If printing to stderr fails, fall back to stdout printing
            try:  # Attempt to output the error to stdout as a last-resort visible channel
                print(str(e))  # Print to stdout when stderr is unavailable
            except Exception:  # If that also fails, give up silently to avoid crashing handler
                pass  # Swallow to avoid cascading failures

    try:  # Always call the original platform excepthook after notifications
        sys.__excepthook__(exc_type, exc_value, exc_tb)  # Invoke default system excepthook for normal terminal behavior
    except Exception:  # If default excepthook itself fails, swallow to avoid loops
        pass  # Ignore exceptions from the original excepthook to avoid recursion


_GLOBAL_HOOK_INSTALLED = False  # Internal flag tracking whether the global hook was installed


def setup_global_exception_hook():  # Install the global excepthook to route uncaught exceptions here
    """
    Install the module-level global exception hook to forward uncaught exceptions to Telegram.
    
    :return: None
    """
    
    global _GLOBAL_HOOK_INSTALLED  # Refer to the module-level installation flag
    if _GLOBAL_HOOK_INSTALLED:  # If already installed, do nothing to avoid duplicate hooks
        return  # Skip installation when already done
    try:  # Try to set the system excepthook to the module handler
        sys.excepthook = send_exception_via_telegram  # Set the global excepthook to forward to Telegram sender
        _GLOBAL_HOOK_INSTALLED = True  # Mark as installed to prevent re-installation
    except Exception as e:  # If installation fails, log and attempt a safe notification
        print(str(e))  # Print installation error to terminal for visibility
        try:  # Attempt to notify about the hook setup failure via the same handler
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send configuration error details via Telegram
        except Exception:  # If notification fails, do not re-raise to avoid import-time recursion
            pass  # Ignore Telegram send errors during hook configuration


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
