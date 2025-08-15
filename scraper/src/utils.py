import os
import json
import logging
from datetime import datetime
import hashlib
from logging.handlers import RotatingFileHandler
try:
    from config import Config
except ImportError as e:
    logging.error(f"Failed to import Config: {e}")
    raise

def setup_directories():
    logger = logging.getLogger('article_scraper.utils')
    try:
        directories = [Config.LOGS_DIR, "data", "data/temp"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logger.info("Directories set up successfully")
    except Exception as e:
        logger.error(f"Failed to set up directories: {e}")
        raise

def setup_logging():
    try:
        setup_directories()
        logger = logging.getLogger('article_scraper')
        logger.setLevel(logging.INFO)
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        file_handler = RotatingFileHandler(
            Config.MAIN_LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info("Logging setup completed")
        logger.debug(f"Config.CHECKPOINT_FILE: {Config.CHECKPOINT_FILE}")
        return logger
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise

def log_failed_article(idx, url, reason):
    logger = logging.getLogger('article_scraper.utils')
    try:
        timestamp = datetime.now().isoformat()
        with open(Config.FAILED_ARTICLES_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}\t{idx}\t{url}\t{reason}\n")
        logger.info(f"Logged failed article {idx}: {reason}")
    except Exception as e:
        logger.error(f"Failed to log failed article {idx}: {e}")

def update_stats(stats_dict):
    logger = logging.getLogger('article_scraper.utils')
    try:
        if os.path.exists(Config.STATS_FILE):
            with open(Config.STATS_FILE, 'r', encoding='utf-8') as f:
                existing_stats = json.load(f)
        else:
            existing_stats = {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'scraping_failed': 0,
                'summarization_failed': 0,
                'embedding_failed': 0,
                'pinecone_failed': 0,
                'ollama_connection_failed': 0,
                'last_updated': None
            }
        
        for key, value in stats_dict.items():
            if key in existing_stats:
                existing_stats[key] += value
            else:
                existing_stats[key] = value
        
        existing_stats['last_updated'] = datetime.now().isoformat()
        
        with open(Config.STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_stats, f, indent=2, ensure_ascii=False)
            
        logger.info("Stats updated successfully")
    except Exception as e:
        logger.error(f"Failed to update stats: {e}")

def clean_text(text):
    logger = logging.getLogger('article_scraper.utils')
    if not text:
        logger.debug("Empty text provided for cleaning")
        return ""
    text = " ".join(text.split())
    text = text.replace('\x00', '')
    logger.debug("Text cleaned successfully")
    return text.strip()

def create_unique_id(idx, url):
    logger = logging.getLogger('article_scraper.utils')
    try:
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
        unique_id = f"article_{idx}_{url_hash}"
        logger.debug(f"Created unique ID: {unique_id}")
        return unique_id
    except Exception as e:
        logger.error(f"Failed to create unique ID for article {idx}: {e}")
        raise

def save_checkpoint(page_num: int, article_idx: int):
    logger = logging.getLogger('article_scraper.utils')
    try:
        checkpoint_path = Config.CHECKPOINT_FILE
        logger.debug(f"Saving checkpoint to {checkpoint_path}")
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump({'page_num': page_num, 'article_idx': article_idx}, f)
        logger.info(f"Checkpoint saved: page {page_num}, article {article_idx}")
    except AttributeError as e:
        logger.error(f"Config attribute error in save_checkpoint: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise