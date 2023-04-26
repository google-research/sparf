"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import logging

import coloredlogs


def create_logger(log_file: str=None):
    logger = logging.getLogger()
    logging.getLogger('matplotlib.font_manager').disabled = True
    # remove the annoying warnings from matplotlib
    logger.handlers.clear()
    logger.setLevel(level=logging.INFO)
    logger.propagate = False

    format_str = '[%(asctime)s] [%(levelname).4s] %(message)s'

    stream_handler = logging.StreamHandler()
    colored_formatter = coloredlogs.ColoredFormatter(format_str)
    stream_handler.setFormatter(colored_formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class Logger:
    def __init__(self, log_file: str=None, local_rank=-1):
        if local_rank == 0 or local_rank == -1:
            self.logger = create_logger(log_file=log_file)
        else:
            self.logger = None

    def debug(self, message: str):
        if self.logger is not None:
            self.logger.debug(message)

    def info(self, message: str):
        if self.logger is not None:
            self.logger.info(message)

    def warning(self, message: str):
        if self.logger is not None:
            self.logger.warning(message)

    def error(self, message: str):
        if self.logger is not None:
            self.logger.error(message)

    def critical(self, message: str):
        if self.logger is not None:
            self.logger.critical(message)
