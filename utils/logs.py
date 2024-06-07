import logging
from dataclasses import dataclass

@dataclass
class CustomFormatter(logging.Formatter):

    grey: str = "\x1b[38;21m"       # DEBUG message
    blue: str = "\x1b[38;5;39m"     # INFO message
    yellow: str = "\x1b[33;20m"     # Warning message
    red: str = "\x1b[31;20m"        # Error message
    bold_red: str = "\x1b[31;1m"    # Critical error message
    reset: str = "\x1b[0m"

    format = "%(asctime)s | %(levelname)s | %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    
    
# Создание логгера
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# # Console handler с многоуровневным log уровнем
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(CustomFormatter())
# Добавление console handler к логгеру
logger.addHandler(console_handler)
