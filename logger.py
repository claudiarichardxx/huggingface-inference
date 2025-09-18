# logger.py
import logging

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO

logging.basicConfig(
    filename="app.log",
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    filemode="a" 
)

logger = logging.getLogger("MBTIApp")