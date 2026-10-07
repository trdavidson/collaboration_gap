from typing import Optional
import enum
import logging
import random
import re
import numpy as np


class ColorType(enum.Enum):
    DEFAULT = ""         # No ANSI code, uses terminal default
    BLACK = "\033[30m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

    @staticmethod
    def get_color(index: int) -> 'ColorType':
        """
        Get a color based on the index, cycling through available colors.

        Args:
            index (int): The index to determine the color.

        Returns:
            ColorType: The corresponding color.
        """
        colors = list(ColorType)[2:]  # Exclude DEFAULT and RESET
        return colors[index % len(colors)]


def print_c(message: str, color: ColorType = ColorType.DEFAULT) -> None:
    """
    Print a message with optional color formatting.

    Args:
        message (str): The message to print.
        color (ColorType): The color to format the message with.
                           Defaults to ColorType.DEFAULT (no color).
    """
    if color == ColorType.DEFAULT:
        print(message)
    else:
        print(f"{color.value}{message}{ColorType.RESET.value}")


def log_c(
        message: str,
        logger_instance: logging.Logger,
        color: ColorType = ColorType.DEFAULT,
        level: int = logging.INFO
) -> None:
    """
    Log a message with optional color formatting and specified logging level.

    Args:
        message (str): The message to log.
        logger_instance (logging.Logger): The logger instance to use.
        color (ColorType): The color to format the message with.
        level (int): The logging level to use (e.g., logging.INFO,
                     logging.WARNING, logging.ERROR). Defaults to logging.INFO.
    """
    if color != ColorType.DEFAULT:
        formatted_message = f"{color.value}{message}{ColorType.RESET.value}"
    else:
        # If no color is specified, use the message as is
        formatted_message = message

    logger_instance.log(level, formatted_message)


def set_random_seed(seed: Optional[int] = None, verbose=False) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (Optional[int]): The seed to set. None reseeds from system
            entropy, i.e. non-reproducibly.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        random.seed()
        np.random.seed()

    if seed is not None and verbose:
        print_c(f"Random seed set to {seed}", ColorType.GREEN)


# Some hosted deployments write the date with an underscore
# (gpt-4.1-mini_2025-04-14) where the providers' public APIs use a hyphen
# (gpt-4.1-mini-2025-04-14), so one model can appear under two spellings
# depending on where a result was produced.
_DEPLOYMENT_DATE_SUFFIX = re.compile(r'[-_](\d{4}-\d{2}-\d{2})$')


def normalize_deployment_name(name: Optional[str]) -> Optional[str]:
    """
    Rewrite a deployment name to a single canonical spelling.

    The trailing date is separated with a hyphen, so both spellings of one
    model compare equal.

    Args:
        name: A deployment name, or None.

    Returns:
        The canonical spelling, or the input unchanged if it carries no date
        suffix. None passes through.
    """
    if not isinstance(name, str):
        return name
    return _DEPLOYMENT_DATE_SUFFIX.sub(r'-\1', name)


def deployment_names_match(left: Optional[str], right: Optional[str]) -> bool:
    """
    Whether two deployment names refer to the same model, ignoring which
    separator precedes the date.

    Args:
        left: A deployment name, or None.
        right: A deployment name, or None.

    Returns:
        True if the canonical spellings are equal.
    """
    return normalize_deployment_name(left) == normalize_deployment_name(right)
