import logging
import os
import uuid
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_FORMATTER = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

agent_logger = logging.getLogger("shopping_agent")
agent_logger.setLevel(logging.DEBUG)

# Console handler — always present, so you still see live output while developing
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_FORMATTER)
agent_logger.addHandler(_console_handler)

# Tracks the file handler for the CURRENT conversation, so we can remove it
# when a new conversation starts (instead of stacking handlers forever)
_current_file_handler = None
_current_session_id = None


def start_new_conversation_log() -> str:
    """
    Call this ONCE when a new chat session begins (app startup) or when the
    user resets the conversation. Creates a brand new log file dedicated to
    that single conversation, and detaches the previous conversation's file
    handler so old turns don't keep getting written into the new file.

    Returns the session_id, useful if you want to display it in the UI
    (e.g. "Session: 7f3a21" in the sidebar for debugging).
    """
    global _current_file_handler, _current_session_id

    # Remove the old conversation's file handler, if one exists
    if _current_file_handler is not None:
        agent_logger.removeHandler(_current_file_handler)
        _current_file_handler.close()

    session_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"conversation_{timestamp}_{session_id}.log"

    file_handler = logging.FileHandler(LOG_DIR / log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)
    agent_logger.addHandler(file_handler)

    _current_file_handler = file_handler
    _current_session_id = session_id

    agent_logger.info(f"=== NEW CONVERSATION STARTED (session_id={session_id}) ===")
    return session_id


