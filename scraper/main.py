# main.py

import asyncio
import logging
from datetime import datetime
import os
import json

from playwright.async_api import async_playwright
from config import Config
from src.utils import setup_logging, setup_directories, update_stats, save_checkpoint
from src.scraper import scrape_single_article, get_article_links_from_search
from src.ollama_service import OllamaService
from src.pinecone_client import PineconeClient

logger = setup_logging()

class ArticleProcessor:
    def __init__(self):
        self.ollama_service = OllamaService()
        self.pinecone_client = PineconeClient()
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'scraping_failed': 0,
            'summarization_failed': 0,
            'embedding_failed': 0,
            'pinecone_failed': 0,
            'ollama_connection_failed': 0
        }
        self.semaphore = asyncio.Semaphore(Config.CONCURRENCY_LIMIT)

    async def initialize(self):
        logger.info("Initializing Article Processor...")
        try:
            Config.validate()
            logger.info("Configuration validated successfully")
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

        if not await self.ollama_service.check_connection():
            logger.error("Failed to connect to Ollama server")
            self.stats['ollama_connection_failed'] += 1
            return False

        logger.info("Connected to Ollama server successfully")

        if not await self.ollama_service.ensure_models_available():
            logger.error("Failed to ensure Ollama models are available")
            return False

        logger.info("All required Ollama models are available")
        return True

    async def process_single_article(self, playwright, idx, url):
        async with self.semaphore:
            logger.info(f"Processing article {idx}")
            self.stats['total_processed'] += 1

            scrape_result = await scrape_single_article(playwright, idx, url)
            if not scrape_result['success']:
                self.stats['scraping_failed'] += 1
                self.stats['failed'] += 1
                return False
            article_data = scrape_result['data']
            article_data['url'] = url

            summarized_data = await self.ollama_service.summarize_article(article_data)
            if not summarized_data:
                self.stats['summarization_failed'] += 1
                self.stats['failed'] += 1
                return False

            await asyncio.sleep(Config.WAIT_BETWEEN_ARTICLES)

            embedding_data = await self.ollama_service.create_embeddings(summarized_data)
            if not embedding_data:
                self.stats['embedding_failed'] += 1
                self.stats['failed'] += 1
                return False

            pinecone_result = await self.pinecone_client.store_article_embedding(
                embedding_data, idx, url
            )
            if not pinecone_result['success']:
                self.stats['pinecone_failed'] += 1
                self.stats['failed'] += 1
                return False

            self.stats['successful'] += 1
            return True

    async def process_articles_batch(self, playwright, article_links, start_idx):
        indexed_urls = list(enumerate(article_links, start=start_idx))
        tasks = [self.process_single_article(playwright, idx, url) for idx, url in indexed_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def print_stats_summary(self):
        logger.info("="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Articles Processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info("-"*40)
        logger.info(f"Scraping Failures: {self.stats['scraping_failed']}")
        logger.info(f"Summarization Failures: {self.stats['summarization_failed']}")
        logger.info(f"Embedding Failures: {self.stats['embedding_failed']}")
        logger.info(f"Pinecone Failures: {self.stats['pinecone_failed']}")
        logger.info(f"Ollama Connection Failures: {self.stats['ollama_connection_failed']}")
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("="*60)

async def main():
    start_time = datetime.now()
    logger.info(f"Starting article scraping and processing pipeline at {start_time}")
    setup_directories()
    processor = ArticleProcessor()

    if not await processor.initialize():
        logger.error("Failed to initialize services. Exiting.")
        return

    try:
        if os.path.exists(Config.CHECKPOINT_FILE):
            with open(Config.CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            article_global_index = checkpoint.get('article_idx', 1)
            start_page = checkpoint.get('page_num', 1)
            logger.info(f"Resuming from checkpoint: page {start_page}, article {article_global_index}")
        else:
            article_global_index = 1
            start_page = 1
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        article_global_index = 1
        start_page = 1

    try:
        async with async_playwright() as playwright:
            page_num = start_page
            while True:
                if page_num == 1:
                    search_url = Config.BASE_SEARCH_URL
                else:
                    search_url = f"{Config.BASE_SEARCH_URL}&page={page_num}"

                logger.info(f"\n{'='*60}")
                logger.info(f"PROCESSING SEARCH RESULTS PAGE {page_num}")
                logger.info(f"{'='*60}")

                page_article_links = await get_article_links_from_search(playwright, search_url)
                if not page_article_links:
                    logger.info(f"No articles found on page {page_num}. Ending pipeline.")
                    break

                logger.info(f"Found {len(page_article_links)} articles on page {page_num}")

                await processor.process_articles_batch(
                    playwright, page_article_links, article_global_index
                )

                article_global_index += len(page_article_links)
                save_checkpoint(page_num, article_global_index)
                update_stats(processor.stats)
                processor.print_stats_summary()

                try:
                    pinecone_stats = await processor.pinecone_client.get_index_stats()
                    if pinecone_stats:
                        logger.info(f"Pinecone Index Total Vectors: {pinecone_stats.get('total_vector_count', 'Unknown')}")
                except Exception as e:
                    logger.error(f"Could not fetch Pinecone stats: {e}")

                logger.info(f"Waiting {Config.WAIT_BETWEEN_PAGES} seconds before next page...")
                await asyncio.sleep(Config.WAIT_BETWEEN_PAGES)
                page_num += 1

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        save_checkpoint(page_num, article_global_index)
        raise
    except Exception as e:
        logger.error(f"Fatal error in main pipeline: {e}")
        save_checkpoint(page_num, article_global_index)
        raise
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("\n" + "="*60)
        logger.info("FINAL PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Start Time: {start_time}")
        logger.info(f"End Time: {end_time}")
        logger.info(f"Total Duration: {duration}")
        processor.print_stats_summary()
        final_stats = {
            **processor.stats,
            'pipeline_duration_seconds': duration.total_seconds(),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
        update_stats(final_stats)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        raise
