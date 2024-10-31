import logging
import sys
from dataclasses import dataclass

from . import output


# ==================================================================================================
@dataclass
class LoggerSettings:
    do_printing: bool = True
    logfile_path: str = None
    write_mode: str = "w"


# ==================================================================================================
class MCMCLogger:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, logger_settings: LoggerSettings) -> None:
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

    # ----------------------------------------------------------------------------------------------
    def log_header(self, outputs: tuple[output.MCMCOutput]) -> None:
        log_header_str = f"| {'Iteration':<12}| {'Time':<12}| "
        for out in outputs:
            if out.log:
                log_header_str += f"{out.str_id}| "
        self.info(log_header_str)
        self.info("-" * (len(log_header_str) - 1))

    # ----------------------------------------------------------------------------------------------
    def log_outputs(self, outputs: output.MCMCOutput, iteration: int, time: float) -> None:
        output_str = f"| {iteration:<12.3e}| {time:<12.3e}| "
        for out in outputs:
            if out.log:
                value_str = f"{out.value:{out.str_format}}"
            output_str += f"{value_str}| "
        self.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def info(self, message: str) -> None:
        self._pylogger.info(message)

    # ----------------------------------------------------------------------------------------------
    def debug(self, message: str) -> None:
        self._pylogger.debug(message)

    # ----------------------------------------------------------------------------------------------
    def exception(self, message: str) -> None:
        self._pylogger.exception(message)
