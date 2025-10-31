import sys
from loguru import logger as NistLogger

NistLogger.remove()

# Console: only DEBUG
NistLogger.add(
    sys.stderr,
    level="DEBUG",
    filter=lambda record: record["level"].name == "DEBUG",
    format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | {message}"
)

# File: only DEBUG
NistLogger.add(
    "debug_only.log",
    level="DEBUG",
    filter=lambda record: record["level"].name == "DEBUG",
    rotation="1 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}"
)

