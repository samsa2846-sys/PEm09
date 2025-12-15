"""
Speech-to-Text Service.
Handles voice message transcription.
"""

from pathlib import Path
from typing import Union

from utils.logging import logger
from utils.helpers import convert_ogg_to_wav, cleanup_file
from config import API_PROVIDER

# Условный импорт в зависимости от провайдера
if API_PROVIDER == "yandex":
    from services.yandex_client import yandex_speechkit_client as stt_client
else:
    from services.openai_client import openai_client as stt_client


async def transcribe_voice_message(audio_path: Union[str, Path]) -> str:
    """
    Transcribe a voice message to text.
    
    Args:
        audio_path: Path to audio file (OGG or WAV)
    
    Returns:
        Transcribed text
    """
    audio_path = Path(audio_path)
    wav_path = None
    
    try:
        # Transcribe using appropriate service
        if API_PROVIDER == "yandex":
            # Yandex SpeechKit принимает OGG напрямую, конвертация не нужна
            logger.debug(f"Transcribing with Yandex SpeechKit: {audio_path}")
            text = await stt_client.transcribe_audio(audio_path)
        else:
            # OpenAI Whisper требует WAV - конвертируем OGG в WAV
            if audio_path.suffix.lower() == '.ogg':
                logger.debug(f"Converting OGG to WAV: {audio_path}")
                wav_path = convert_ogg_to_wav(audio_path)
                transcription_path = wav_path
            else:
                transcription_path = audio_path
            
            text = await stt_client.transcribe_audio(transcription_path)
        
        logger.info(f"Transcription completed: {len(text)} characters")
        return text
        
    except Exception as e:
        logger.error(f"Error in voice transcription: {e}")
        raise
        
    finally:
        # Cleanup converted file if created
        if wav_path and wav_path != audio_path:
            cleanup_file(wav_path)

