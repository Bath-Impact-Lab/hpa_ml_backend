import logging
import os
import inspect
from datetime import datetime

def setup_logger():
    # Create a logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Timestamp for log filename
    log_filename = datetime.now().strftime("logs/log_%Y%m%d_%H%M%S.log")
    
    # Setup basic configuration for logging
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def log_message(level, message):
   # Get the current frame and extract the calling function's line number
    frame = inspect.currentframe().f_back
    line_number = frame.f_lineno
    function_name = frame.f_code.co_name
    filename = frame.f_code.co_filename

    # Custom message with line number info
    log_msg = f"{message} | Called from {filename}:{function_name}, line {line_number}"

    # Log based on the provided level
    if level == 'info':
        logging.info(log_msg)
    elif level == 'warning':
        logging.warning(log_msg)
    elif level == 'error':
        logging.error(log_msg)
    elif level == 'debug':
        logging.debug(log_msg)
    else:
        logging.info(log_msg)

    pass