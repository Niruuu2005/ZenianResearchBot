import os
import logging
import random
import re

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
import openai

from embedding_service import EmbeddingService
from pinecone_client import PineconeQueryClient

# ===== Load env =====
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if TELEGRAM_TOKEN is None or TELEGRAM_TOKEN == "":
    raise ValueError("Missing TELEGRAM_TOKEN in your .env file.")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("query_bot")

# ===== Init services =====
embedding_service = EmbeddingService()
pinecone_client = PineconeQueryClient()

# ===== Greeting patterns =====
GREETINGS = [
    "hello", "hi", "hey", "heyy", "hii", "yo", "sup",
    "good morning", "good afternoon", "good evening",
    "bye", "goodbye", "see you", "take care"
]

def is_greeting(query: str) -> bool:
    q_lower = query.lower()
    return any(
        re.fullmatch(r"[\W_]*(" + re.escape(g) + r")[\W_]*", q_lower) or g in q_lower
        for g in GREETINGS
    )

def escape_html(text: str) -> str:
    if text is None:
        return ""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

async def get_friendly_reply(user_text: str) -> str:
    if OPENAI_API_KEY:
        try:
            prompt = (
                "Reply in 1-2 short lines, warmly and casually, to this greeting:\n\n"
                f"{user_text}"
            )
            response = openai.resources.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a friendly Telegram chatbot."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
    return random.choice([
        "👋 Hi! How’s your day going?",
        "Hey there! How can I help you today?",
        "Hello! 😊 Ready to explore some research papers?",
        "Hi! Feel free to ask about research topics."
    ])

async def start(update, context):
    await update.message.reply_text(
        "Hi! 👋\nSend me a research topic and I’ll fetch the <b>top matching paper</b>.",
        parse_mode="HTML"
    )

def _get_attr_or_key(obj, name, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default

async def handle_query(update, context):
    query_text = update.message.text.strip()
    if not query_text:
        await update.message.reply_text("Please send a valid query text.", parse_mode="HTML")
        return

    if is_greeting(query_text):
        reply = await get_friendly_reply(query_text)
        await update.message.reply_text(reply, parse_mode="HTML")
        return

    await update.message.reply_text("🔎 Searching for the most relevant paper...", parse_mode="HTML")

    vector = embedding_service.create_embeddings(query_text)
    if not vector:
        await update.message.reply_text("Failed to create embedding.", parse_mode="HTML")
        return

    matches = pinecone_client.query_top_k(vector, top_k=1)
    if not matches:
        await update.message.reply_text("No relevant papers found.", parse_mode="HTML")
        return

    top = matches[0]
    meta = _get_attr_or_key(top, "metadata", {}) or {}

    title = (meta.get("title") or "[No title available]").strip()
    summary_raw = (meta.get("summary") or "[No summary available]").strip()
    link_url = meta.get("link") or "N/A"

    if len(summary_raw) > 400:
        summary_raw = summary_raw[:400].rstrip() + "..."

    title_escaped = escape_html(title)
    summary = escape_html(summary_raw)
    link_escaped = escape_html(link_url)

    if link_url.startswith("http"):
        link_html = f'<a href="{link_escaped}">Open Article ↗</a>'
    else:
        link_html = link_escaped

    message = (
        f"📄 <b>Title:</b> {title_escaped}\n"
        f"📝 <b>Summary:</b> {summary}\n"
        f"🔗 {link_html}"
    )

    await update.message.reply_text(
        message,
        parse_mode="HTML",
        disable_web_page_preview=True
    )

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    logger.info("Telegram bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
