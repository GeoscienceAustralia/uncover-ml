import logging
import time

from pathlib import Path

from uncoverml.config import Config


# I know that this is sketchy, will switch to logging on next refactor
def write_progress_to_file(type, message, current_config: Config):
    write_file_path = Path(current_config.output_dir) / f'{type}_progress.txt'
    with open(write_file_path, 'a') as write_file:
        write_file.write(f'{message}\n')
