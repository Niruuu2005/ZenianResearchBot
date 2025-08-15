# ollama_service.py

import logging
from typing import Dict, Optional
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime

from config import Config
from src.embedding_service import EmbeddingService  # Local embedding service

logger = logging.getLogger('article_scraper.ollama')

class OllamaService:
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.timeout = Config.OLLAMA_TIMEOUT
        self.embedding_service = EmbeddingService()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def check_connection(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        logger.info("Ollama connection check successful")
                        return True
            logger.error(f"Ollama connection check failed: HTTP {response.status}")
            return False
        except Exception as e:
            logger.error(f"Ollama connection check failed: {e}")
            return False

    async def ensure_models_available(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch Ollama models: HTTP {response.status}")
                        return False
                    data = await response.json()
                    available_models = [m['name'] for m in data.get('models', [])]
                    if not any(self.model in am for am in available_models):
                        await self._pull_model(self.model, session)
                    return True
        except Exception as e:
            logger.error(f"Error checking models: {e}")
            return False

    async def _pull_model(self, model_name: str, session: aiohttp.ClientSession):
        try:
            logger.info(f"Pulling model: {model_name}")
            async with session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300
            ) as response:
                if response.status == 200:
                    logger.info(f"Successfully pulled model: {model_name}")
                else:
                    logger.error(f"Failed to pull model {model_name}: {response.status}")
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def summarize_article(self, article_data: Dict) -> Optional[Dict]:
        """
        Summarize article to ~150 words using Ollama.
        Output should contain ONLY the summary text.
        """
        try:
            content_to_summarize = self._prepare_content(article_data)
            if not content_to_summarize:
                logger.warning("No content to summarize")
                return None

            prompt = (
                "Provide a concise summary of the following research article in about 150 words. "
                "Do NOT include any introductory or concluding phrases such as "
                "'Here’s a summary' or 'In conclusion'. "
                "Return only the plain summary text without any additional commentary or labels:\n\n"
                f"{content_to_summarize}\n\nSummary:"
            )

            summary = await self._generate_text(prompt, self.model)
            if not summary:
                logger.error("Failed to generate summary")
                return None

            cleaned_summary = summary.strip()
            for prefix in [
                "Here’s a summary of the research article:",
                "Here’s a 150-word summary of the article:",
                "Here's a summary of the research article:",
                "Here's a 150-word summary of the article:"
            ]:
                if cleaned_summary.startswith(prefix):
                    cleaned_summary = cleaned_summary[len(prefix):].strip()

            logger.info(f"Successfully summarized article: {article_data['title'][:50]}...")

            return {
                'summary': cleaned_summary,
                'link': article_data.get('url', ''),
                'timestamp': datetime.utcnow().isoformat(),
                'title': article_data.get('title', '')  # Added title here
            }

        except Exception as e:
            logger.error(f"Error summarizing article: {e}")
            return None

    async def create_embeddings(self, summarized_data: Dict) -> Optional[Dict]:
        """Create embeddings for the summary only"""
        try:
            summary_text = summarized_data.get('summary', '')
            if not summary_text:
                logger.warning("No summary text to create embeddings for")
                return None

            embeddings = await asyncio.to_thread(
                self.embedding_service.create_embeddings, summary_text
            )
            if not embeddings:
                logger.error("Failed to generate embeddings locally")
                return None

            logger.info("Successfully created embeddings for summary")
            metadata = {
                'link': summarized_data.get('link', ''),
                'summary': summarized_data.get('summary', ''),
                'timestamp': summarized_data.get('timestamp', ''),
                'title': summarized_data.get('title', '')  # Add title here
            }
            return {
                'embeddings': embeddings,
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"Error creating embeddings locally: {e}")
            return None

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_text(self, prompt: str, model: str) -> Optional[str]:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": Config.TEMPERATURE,
                        "num_predict": Config.MAX_TOKENS_FOR_SUMMARY,
                        "top_p": Config.TOP_P,
                        "top_k": Config.TOP_K
                    }
                }
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('response', '').strip()
                    else:
                        logger.error(f"Ollama generation failed: HTTP {response.status}")
                        return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout generating text with model {model}")
            return None
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None

    def _prepare_content(self, article_data: Dict) -> str:
        parts = []
        if article_data.get('title'):
            parts.append(f"Title: {article_data['title']}")
        if article_data.get('abstract'):
            parts.append(f"Abstract: {article_data['abstract']}")
        if article_data.get('content'):
            content = article_data['content']
            if len(content) > 3000:
                content = content[:3000] + "..."
            parts.append(f"Content: {content}")
        return "\n\n".join(parts)
