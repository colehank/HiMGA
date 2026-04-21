import sys

from loguru import logger


def setup_logger():
    logger.remove()
    custom_format = (
        "<green>{time:MMDD-HH:mm}</green>|"
        "<level>{level: ^4}</level>|"
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, format=custom_format)

    # logger.add("logs/app.log", rotation="10 MB", format=custom_format)


setup_logger()
__all__ = ["logger"]
