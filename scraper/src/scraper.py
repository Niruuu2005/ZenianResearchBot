import logging
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config
from src.utils import clean_text, log_failed_article

logger = logging.getLogger('article_scraper.scraper')

async def safe_get_text(page, selectors):
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            text = await locator.inner_text(timeout=5000)
            if text.strip():
                return text
        except Exception:
            continue
    return ''

async def safe_get_texts(page, selectors):
    for selector in selectors:
        try:
            locs = page.locator(selector)
            count = await locs.count()
            texts = [await locs.nth(i).inner_text(timeout=5000) for i in range(count)]
            if texts:
                return texts
        except Exception:
            continue
    return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def extract_article_data(page):
    try:
        title_selectors = ['.c-article-title', 'h1', '[class*="title"]', '[class*="header"]']
        identifier_selectors = ['.c-article-identifiers__item', '[class*="doi"]', '[class*="identifier"]']
        abstract_selectors = ['div#Abs1-content p', '[id*="abstract"] p', '[class*="abstract"] p']
        content_selectors = ['.u-serif.js-main-column', '[class*="content"]', '[class*="article-body"]']
        
        title = clean_text(await safe_get_text(page, title_selectors))
        identifiers = [clean_text(s) for s in await safe_get_texts(page, identifier_selectors)]
        
        paragraphs = await page.eval_on_selector_all(
            ','.join(abstract_selectors),
            'elements => elements.map(e => e.innerText.trim())'
        )
        abstract_text = clean_text("\n\n".join(paragraphs))
        
        main_content = clean_text(await safe_get_text(page, content_selectors))
        
        if not title:
            logger.warning("No title extracted")
            return None
            
        return {
            'title': title,
            'identifiers': identifiers,
            'abstract': abstract_text,
            'content': main_content,
            'full_text': f"{title}\n\n{abstract_text}\n\n{main_content}".strip()
        }
    except Exception as e:
        logger.error(f"Error extracting article data: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def scrape_single_article(playwright, idx, url):
    browser = None
    context = None
    page = None
    try:
        logger.info(f"[Article {idx}] Starting scrape: {url}")
        browser = await playwright.chromium.launch(
            headless=Config.HEADLESS_MODE, 
            timeout=Config.LAUNCH_TIMEOUT
        )
        context = await browser.new_context(user_agent=Config.USER_AGENT)
        
        async def block_resources(route):
            if route.request.resource_type in ["image", "font", "stylesheet"]:
                await route.abort()
            else:
                await route.continue_()
        
        page = await context.new_page()
        await page.route("**/*", block_resources)
        try:
            await page.goto(url, wait_until='networkidle', timeout=Config.SCRAPING_TIMEOUT)
        except PlaywrightTimeoutError:
            logger.warning(f"[Article {idx}] Networkidle timeout, falling back to domcontentloaded")
            await page.goto(url, wait_until='domcontentloaded', timeout=Config.SCRAPING_TIMEOUT)
        
        article_data = await extract_article_data(page)
        
        if article_data and article_data['title']:
            logger.info(f"[Article {idx}] Successfully scraped: {article_data['title'][:50]}...")
            return {
                'success': True,
                'data': article_data,
                'url': url,
                'idx': idx
            }
        else:
            logger.warning(f"[Article {idx}] No valid data extracted")
            log_failed_article(idx, url, "No valid data extracted")
            return {'success': False, 'idx': idx, 'url': url}
            
    except PlaywrightTimeoutError:
        logger.error(f"[Article {idx}] Timeout error")
        log_failed_article(idx, url, "Timeout error")
        return {'success': False, 'idx': idx, 'url': url}
    except Exception as e:
        logger.error(f"[Article {idx}] Error: {e}")
        log_failed_article(idx, url, str(e))
        return {'success': False, 'idx': idx, 'url': url}
    finally:
        if page:
            try:
                await page.close()
                logger.debug(f"[Article {idx}] Page closed successfully")
            except Exception as e:
                logger.error(f"[Article {idx}] Error closing page: {e}")
        if context:
            try:
                await context.close()
                logger.debug(f"[Article {idx}] Context closed successfully")
            except Exception as e:
                logger.error(f"[Article {idx}] Error closing context: {e}")
        if browser:
            try:
                await browser.close()
                logger.debug(f"[Article {idx}] Browser closed successfully")
            except Exception as e:
                logger.error(f"[Article {idx}] Error closing browser: {e}")

async def get_article_links_from_search(playwright, search_url):
    browser = None
    context = None
    page = None
    try:
        logger.info(f"Fetching search results: {search_url}")
        browser = await playwright.chromium.launch(
            headless=Config.HEADLESS_MODE,
            timeout=Config.LAUNCH_TIMEOUT
        )
        context = await browser.new_context(user_agent=Config.USER_AGENT)
        page = await context.new_page()
        try:
            await page.goto(search_url, wait_until='networkidle', timeout=Config.SCRAPING_TIMEOUT)
        except PlaywrightTimeoutError:
            logger.warning(f"Search page timeout, falling back to domcontentloaded")
            await page.goto(search_url, wait_until='domcontentloaded', timeout=Config.SCRAPING_TIMEOUT)
        
        article_links = await page.eval_on_selector_all(
            'a.app-card-open__link, [class*="article-link"], [href*="/article/"], [href*="/chapter/"]',
            'elements => elements.map(el => el.href)'
        )
        article_links = list(set(article_links))[:20]  # Remove duplicates, limit to 20
        logger.info(f"Found {len(article_links)} article links")
        return article_links
    except Exception as e:
        logger.error(f"Error fetching search page {search_url}: {e}")
        return []
    finally:
        if page:
            try:
                await page.close()
                logger.debug("Search page closed successfully")
            except Exception as e:
                logger.error(f"Error closing search page: {e}")
        if context:
            try:
                await context.close()
                logger.debug("Search context closed successfully")
            except Exception as e:
                logger.error(f"Error closing search context: {e}")
        if browser:
            try:
                await browser.close()
                logger.debug("Search browser closed successfully")
            except Exception as e:
                logger.error(f"Error closing search browser: {e}")