import logging
import os

from config import USER

# ensure a logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# build filename using USER
log_file = os.path.join(LOG_DIR, f"{USER}_activity.log")

logger = logging.getLogger("phasepaint")
logger.setLevel(logging.INFO)
# avoid adding multiple handlers if module is imported repeatedly
if not logger.handlers:
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def log(message: str):
    """Write a timestamped message to the shared activity log."""
    logger.info(message)
