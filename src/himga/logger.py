import sys

from loguru import logger

_FMT_DEBUG = (
    "<green>{time:MMDD-HH:mm}</green>|"
    "<level>{level: ^4}</level>|"
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
_FMT_INFO = "<green>{time:MMDD-HH:mm}</green>|<level>{level: ^4}</level>| <level>{message}</level>"


def setup_logger():
    logger.remove()
    # DEBUG 级别：带文件/函数/行号，仅输出 DEBUG 条目
    logger.add(
        sys.stderr, format=_FMT_DEBUG, level="DEBUG", filter=lambda r: r["level"].name == "DEBUG"
    )
    # INFO 及以上：精简格式，不含位置信息
    logger.add(
        sys.stderr, format=_FMT_INFO, level="INFO", filter=lambda r: r["level"].name != "DEBUG"
    )

    # logger.add("logs/app.log", rotation="10 MB", format=_FMT_DEBUG)


setup_logger()
__all__ = ["logger"]
