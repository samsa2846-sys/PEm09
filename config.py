"""
Configuration module for the Personal Assistant Telegram Bot.
Loads environment variables and provides configuration settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env file")

# ==================== API CONFIGURATION ====================
# Выбор провайдера AI: "yandex", "openai", "proxyapi"
API_PROVIDER = os.getenv("API_PROVIDER", "yandex").lower()

# Yandex Cloud Configuration
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")

# OpenAI Configuration (опционально, если нужен OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_PROXYAPI = os.getenv("USE_PROXYAPI", "false").lower() == "true"
PROXYAPI_BASE_URL = "https://api.proxyapi.ru/openai/v1"
OPENAI_BASE_URL = PROXYAPI_BASE_URL if USE_PROXYAPI else "https://api.openai.com/v1"

# Проверка наличия необходимых ключей
if API_PROVIDER == "yandex":
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        raise ValueError(
            "❌ Для работы с Yandex API нужны YANDEX_API_KEY и YANDEX_FOLDER_ID в .env файле!\n"
            "📖 Инструкция: YANDEX_SETUP.md"
        )
elif API_PROVIDER in ["openai", "proxyapi"]:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in .env file")

# Bot Modes
class BotMode:
    TEXT = "text"
    VOICE = "voice"
    VISION = "vision"
    RAG = "rag"

# Для технического консультанта RAG режим по умолчанию
DEFAULT_MODE = os.getenv("BOT_MODE", BotMode.RAG)

# Voice Configuration
class VoiceType:
    ALLOY = "alloy"      # Neutral
    ECHO = "echo"        # Male
    NOVA = "nova"        # Female
    FABLE = "fable"      # Male (British)
    ONYX = "onyx"        # Male (Deep)
    SHIMMER = "shimmer"  # Female (Warm)

DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", VoiceType.ALLOY)

# OpenAI Models (если используется OpenAI)
GPT_MODEL = "gpt-4o"
GPT_MINI_MODEL = "gpt-4o-mini"
WHISPER_MODEL = "whisper-1"
TTS_MODEL = "tts-1"
VISION_MODEL = "gpt-4o"
DALLE_MODEL = "dall-e-3"

# Yandex Models
YANDEX_GPT_MODEL = os.getenv("YANDEX_GPT_MODEL", "yandexgpt-lite")  # или "yandexgpt"
YANDEX_TTS_VOICE = os.getenv("YANDEX_TTS_VOICE", "alena")  # alena, filipp, ermil, jane, omazh, zahar, dasha

# DALL-E Configuration
DALLE_DEFAULT_SIZE = "1024x1024"  # Options: 1024x1024, 1024x1792, 1792x1024
DALLE_DEFAULT_QUALITY = "standard"  # Options: standard, hd
DALLE_DEFAULT_STYLE = "vivid"  # Options: vivid, natural

# Database Configuration
DB_PATH = BASE_DIR / os.getenv("DB_PATH", "data/embeddings.db")

# Data paths
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EMBEDDINGS_DB = DATA_DIR / "embeddings.db"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "bot.log"

# RAG Configuration
# Для технической документации увеличены параметры
RAG_CHUNK_SIZE = 1500  # Увеличено для целостности таблиц
RAG_CHUNK_OVERLAP = 300  # Больше overlap для связности
RAG_TOP_K = 5  # Больше результатов для полноты ответа

# OpenAI Settings
TEMPERATURE = 0.7
MAX_TOKENS = 1500

# User session settings
MAX_HISTORY_LENGTH = 10  # Maximum number of messages to keep in history

