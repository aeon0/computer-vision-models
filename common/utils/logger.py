import logging
import io
import os
import sys
import re
import warnings
import tensorflow as tf


class Logger:
    """Manage logging"""
    _logger_level = logging.DEBUG
    _formatter = logging.Formatter('[%(asctime)s] %(levelname)-10s %(message)s')
    _log_contents = io.StringIO()
    _current_log_file_path = "info.log"
    _output = ""  # intercepted output from stdout and stderr
    logger = None
    string_handler = None
    file_handler = None
    console_handler = None

    @staticmethod
    def setup_logger():
        """
        Setup logger for StringIO, console and file handler
        """
        if Logger.logger is not None:
            print("WARNING: logger was setup already, deleting all previously existing handlers")
            for hdlr in Logger.logger.handlers[:]:  # remove all old handlers
                Logger.logger.removeHandler(hdlr)

        # Create the logger
        Logger.logger = tf.get_logger()
        for hdlr in Logger.logger.handlers:
            Logger.logger.removeHandler(hdlr)
        Logger.logger.setLevel(Logger._logger_level)
        Logger.logger.propagate = False  # otherwise tensorflow duplicates each logging to the console

        # Setup the StringIO handler
        Logger._log_contents = io.StringIO()
        Logger.string_handler = logging.StreamHandler(Logger._log_contents)
        Logger.string_handler.setLevel(Logger._logger_level)

        # Setup the console handler
        Logger.console_handler = logging.StreamHandler(sys.stdout)
        Logger.console_handler.setLevel(Logger._logger_level)

        # Setup the file handler
        Logger.file_handler = logging.FileHandler(Logger._current_log_file_path, 'a')
        Logger.file_handler.setLevel(Logger._logger_level)

        # Optionally add a formatter
        Logger.string_handler.setFormatter(Logger._formatter)
        Logger.console_handler.setFormatter(Logger._formatter)
        Logger.file_handler.setFormatter(Logger._formatter)

        # Add the console handler to the logger
        Logger.logger.addHandler(Logger.string_handler)
        Logger.logger.addHandler(Logger.console_handler)
        Logger.logger.addHandler(Logger.file_handler)

    @staticmethod
    def write(buf):
        """
        Override the write() function for stdout / stderr to intercept it and use logger instead
        :param buf: a string passed to stdout
        """
        # For each new line character a seperate log entry should be generated
        # this replicates the stdout / stderr output into the logs
        for i, line in enumerate(buf.splitlines(True)):
            Logger._output += line
            if re.match(r'^.*[\r\n]$', line):
                Logger.flush()

    @staticmethod
    def flush():
        """
        Override flush method of stdout / stderr
        each time flush is called a new log entry should be generated
        apart from the user calling flush, write() calls flush after each new line character
        """
        # strip and remove \b (backspace) from output string as keras likes to add these a lot
        Logger._output = Logger._output.strip().replace("\b", "")
        if Logger._output != "":
            Logger.logger.info(Logger._output)
            Logger._output = ""

    @staticmethod
    def set_log_file(path, mode: str='a'):
        """
        Set the path of the log file
        :param path: path + name of the new log file
        :param mode: mode e.g. 'a' => append (default), 'w' => write
        """
        Logger._current_log_file_path = path
        Logger.logger.removeHandler(Logger.file_handler)

        Logger.file_handler = logging.FileHandler(Logger._current_log_file_path, mode)
        Logger.file_handler.setLevel(Logger._logger_level)
        Logger.logger.addHandler(Logger.file_handler)

    @staticmethod
    def remove_file_logger():
        """
        Remove the file logger to not write output to a log file
        """
        Logger.logger.removeHandler(Logger.file_handler)
        if os.path.exists(Logger._current_log_file_path):
            try:
                os.remove(Logger._current_log_file_path)
            except PermissionError:
                warnings.warn("Could not remove info.log, permission denied")

    @staticmethod
    def get_contents():
        """
        Get current contents of the logger
        :return: list of log strings
        """
        return Logger._log_contents.getvalue()

    @staticmethod
    def get_log_file_path() -> str:
        """
        :return: path to the current log file
        """
        return Logger._current_log_file_path

    @staticmethod
    def set_level(lvl):
        """
        Set logging level
        :param lvl: logging level
        """
        Logger._logger_level = lvl
        Logger.setup_logger()

    @staticmethod
    def init():
        """
        Initialize the logger, only needed to be called once as it is a static class
        """
        Logger.setup_logger()

        # redirect stderr & stdout to logger (implements write method)
        sys.stderr = Logger
        sys.stdout = Logger
