"""
Text-to-Speech Service.
Handles text to voice conversion.
"""

from pathlib import Path
from typing import Optional

from utils.logging import logger
from config import VoiceType, DEFAULT_VOICE, API_PROVIDER, YANDEX_TTS_VOICE

# Условный импорт в зависимости от провайдера
if API_PROVIDER == "yandex":
    from services.yandex_client import yandex_speechkit_client as tts_client
else:
    from services.openai_client import openai_client as tts_client

# Маппинг голосов OpenAI -> Yandex
VOICE_MAPPING = {
    VoiceType.ALLOY: "alena",
    VoiceType.ECHO: "filipp",
    VoiceType.NOVA: "jane",
    VoiceType.FABLE: "ermil",
    VoiceType.ONYX: "zahar",
    VoiceType.SHIMMER: "dasha"
}


async def generate_voice_response(
    text: str,
    voice: str = DEFAULT_VOICE
) -> Path:
    """
    Generate voice response from text.
    
    Args:
        text: Text to convert to speech
        voice: Voice type to use
    
    Returns:
        Path to generated audio file
    """
    try:
        # Validate voice type
        valid_voices = [
            VoiceType.ALLOY,
            VoiceType.ECHO,
            VoiceType.NOVA,
            VoiceType.FABLE,
            VoiceType.ONYX,
            VoiceType.SHIMMER
        ]
        
        if voice not in valid_voices:
            logger.warning(f"Invalid voice '{voice}', using default")
            voice = DEFAULT_VOICE
        
        # Generate speech
        logger.debug(f"Generating voice response with voice: {voice}")
        
        if API_PROVIDER == "yandex":
            # Конвертируем голос OpenAI в Yandex формат
            yandex_voice = VOICE_MAPPING.get(voice, YANDEX_TTS_VOICE)
            audio_path = await tts_client.generate_speech(text, voice=yandex_voice)
        else:
            audio_path = await tts_client.generate_speech(text, voice=voice)
        
        logger.info(f"Voice response generated: {audio_path}")
        return audio_path
        
    except Exception as e:
        logger.error(f"Error generating voice response: {e}")
        raise


def get_voice_info(voice: str) -> dict:
    """
    Get information about a voice type.
    
    Args:
        voice: Voice identifier
    
    Returns:
        Dictionary with voice information
    """
    voices = {
        VoiceType.ALLOY: {
            "name": "Alloy",
            "type": "Нейтральный",
            "description": "Сбалансированный голос"
        },
        VoiceType.ECHO: {
            "name": "Echo",
            "type": "Мужской",
            "description": "Четкий мужской голос"
        },
        VoiceType.NOVA: {
            "name": "Nova",
            "type": "Женский",
            "description": "Энергичный женский голос"
        },
        VoiceType.FABLE: {
            "name": "Fable",
            "type": "Мужской (британский)",
            "description": "Британский акцент"
        },
        VoiceType.ONYX: {
            "name": "Onyx",
            "type": "Мужской (глубокий)",
            "description": "Глубокий мужской голос"
        },
        VoiceType.SHIMMER: {
            "name": "Shimmer",
            "type": "Женский (теплый)",
            "description": "Теплый женский голос"
        }
    }
    
    return voices.get(voice, voices[VoiceType.ALLOY])


def get_available_voices() -> str:
    """
    Get formatted list of available voices.
    
    Returns:
        Formatted string with voice information
    """
    voices = [
        VoiceType.ALLOY,
        VoiceType.ECHO,
        VoiceType.NOVA,
        VoiceType.FABLE,
        VoiceType.ONYX,
        VoiceType.SHIMMER
    ]
    
    result = "📢 Доступные голоса:\n\n"
    for voice in voices:
        info = get_voice_info(voice)
        result += f"• {info['name']} ({voice})\n"
        result += f"  Тип: {info['type']}\n"
        result += f"  {info['description']}\n\n"
    
    return result

