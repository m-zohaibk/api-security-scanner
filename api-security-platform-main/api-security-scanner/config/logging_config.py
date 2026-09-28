import logging
import os
from config.settings import LOG_LEVEL

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | "
               "%(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("scanner.log")
        ]
    )
    return logging.getLogger("api-scanner")

logger = setup_logging()
