"""Simple logger for tracking MCMC sampling.

Classes:
    LoggerSettings: Data class for logger settings
    MCMCLogger: Custom Logger class
"""

import logging
import sys
from dataclasses import dataclass

from . import output


# ==================================================================================================
@dataclass
class LoggerSettings:
    """Configuration settings for MCMC logging.

    Attributes:
        do_printing: Whether to enable console output. Defaults to True.
        logfile_path: Path to the log file. If None, no file logging is performed.
        write_mode: File write mode ('w' for overwrite, 'a' for append). Defaults to 'w'.
    """

    do_printing: bool = True
    logfile_path: str = None
    write_mode: str = "w"


# ==================================================================================================
class MCMCLogger:
    """Logger for MCMC sampling with console and file output support.

    Methods:
        log_header: Log formatted header for MCMC output
        log_outputs: Log all outputs
        info: Pass info message to underlying Python logger
        debug: Pass debug message to underlying Python logger
        error: Pass error message to underlying Python logger
        exception: Pass exception message to underlying Python logger
    """

    def __init__(self, logger_settings: LoggerSettings) -> None:
        """Initialize the MCMC logger with specified settings.

        Args:
            logger_settings: Logger settings
        """
        self._logfile_path = logger_settings.logfile_path
        self._pylogger = logging.getLogger(__name__)
        self._pylogger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")

        if not self._pylogger.hasHandlers():
            if logger_settings.do_printing:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self._pylogger.addHandler(console_handler)

            if self._logfile_path is not None:
                self._logfile_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(
                    self._logfile_path, mode=logger_settings.write_mode
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.INFO)
                self._pylogger.addHandler(file_handler)

    def log_header(self, outputs: tuple[output.MCMCOutput]) -> None:
        """Log a formatted header for MCMC output columns.

        Args:
            outputs: Tuple of MCMC output objects to include in the header.
        """
        log_header_str = f"| {'Iteration':<12}| {'Time':<12}| "
        for out in outputs:
            if out.log:
                log_header_str += f"{out.str_id}| "
        self.info(log_header_str)
        self.info("-" * (len(log_header_str) - 1))

    def log_outputs(self, outputs: output.MCMCOutput, iteration: int, time: float) -> None:
        """Log formatted MCMC output values for the current iteration.

        Args:
            outputs: MCMC output objects containing values to log.
            iteration: Current iteration number.
            time: Elapsed time for this iteration.
        """
        output_str = f"| {iteration:<12.3e}| {time:<12.3e}| "
        for out in outputs:
            if out.log:
                value_str = f"{out.value:{out.str_format}}"
            output_str += f"{value_str}| "
        self.info(output_str)

    def info(self, message: str) -> None:
        """Log an info-level message.

        Args:
            message: The message to log.
        """
        self._pylogger.info(message)

    def debug(self, message: str) -> None:
        """Log a debug-level message.

        Args:
            message: The message to log.
        """
        self._pylogger.debug(message)

    def exception(self, message: str) -> None:
        """Log an exception with traceback information.

        Args:
            message: The message to log.
        """
        self._pylogger.exception(message)

    def error(self, message: str) -> None:
        """Log an error.

        Args:
            message (str): The message to log.
        """
        self._pylogger.error(message)
