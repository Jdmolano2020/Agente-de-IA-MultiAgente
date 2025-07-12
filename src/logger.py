import logging
import os
from datetime import datetime


def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"procesamiento_{datetime.now().strftime('%Y%m%d')}.log")
    
    logger = logging.getLogger("multi_agente_ia")
    logger.setLevel(logging.INFO)
    
    # Formato del log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    
    # Evitar agregar handlers duplicados
    if not logger.handlers:
        # Handler para archivo
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

