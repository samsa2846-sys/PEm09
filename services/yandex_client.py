"""
Yandex Cloud Client.
Provides methods for YandexGPT, SpeechKit (STT/TTS), and Vision API.
"""

import requests
import base64
import json
from typing import List, Dict, Optional
from pathlib import Path

from config import (
    YANDEX_API_KEY,
    YANDEX_FOLDER_ID,
    YANDEX_GPT_MODEL,
    TEMPERATURE,
    MAX_TOKENS
)
from utils.logging import logger


class YandexGPTClient:
    """Client for Yandex GPT API operations."""
    
    def __init__(self):
        """Initialize the Yandex GPT client."""
        self.api_key = YANDEX_API_KEY.strip()
        self.folder_id = YANDEX_FOLDER_ID.strip()
        self.model_uri = f"gpt://{self.folder_id}/{YANDEX_GPT_MODEL}"
        
        logger.info(f"Yandex GPT client initialized with model: {YANDEX_GPT_MODEL}")
    
    async def generate_text_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """
        Generate text response using Yandex GPT.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Response randomness (0-1)
            max_tokens: Maximum tokens in response
        
        Returns:
            Generated text response
        """
        try:
            # Преобразуем формат OpenAI в формат Yandex
            yandex_messages = self._convert_messages_format(messages)
            
            url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "modelUri": self.model_uri,
                "completionOptions": {
                    "temperature": temperature,
                    "maxTokens": max_tokens
                },
                "messages": yandex_messages
            }
            
            logger.debug(f"Sending request to Yandex GPT")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Yandex GPT API error ({response.status_code}): {error_text}")
                raise RuntimeError(f"Yandex GPT API error: {error_text}")
            
            result = response.json()
            answer = result["result"]["alternatives"][0]["message"]["text"]
            
            logger.info(f"Generated response: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating text response: {e}")
            raise
    
    def _convert_messages_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert OpenAI message format to Yandex format.
        
        OpenAI format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        Yandex format: [{"role": "system", "text": "..."}, {"role": "user", "text": "..."}]
        """
        yandex_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Yandex поддерживает только system и user роли
            if role == "assistant":
                role = "user"  # Или можно добавить префикс к тексту
            
            yandex_messages.append({
                "role": role,
                "text": content
            })
        
        return yandex_messages


class YandexSpeechKitClient:
    """Client for Yandex SpeechKit (STT and TTS)."""
    
    def __init__(self):
        """Initialize the Yandex SpeechKit client."""
        self.api_key = YANDEX_API_KEY.strip()
        self.folder_id = YANDEX_FOLDER_ID.strip()
        
        logger.info("Yandex SpeechKit client initialized")
    
    async def transcribe_audio(
        self,
        audio_file_path: Path,
        language: str = "ru-RU"
    ) -> str:
        """
        Transcribe audio file to text using Yandex SpeechKit.
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (ru-RU, en-US, etc.)
        
        Returns:
            Transcribed text
        """
        try:
            logger.debug(f"Transcribing audio: {audio_file_path}")
            
            # Читаем аудио файл
            with open(audio_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
            }
            
            # Определяем формат аудио
            audio_format = "oggopus" if audio_file_path.suffix.lower() == ".ogg" else "lpcm"
            
            params = {
                "folderId": self.folder_id,
                "lang": language,
                "format": audio_format,
            }
            
            # Отправляем аудио как бинарные данные
            response = requests.post(
                url,
                headers=headers,
                params=params,
                data=audio_data,
                timeout=60
            )
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Yandex SpeechKit STT error ({response.status_code}): {error_text}")
                raise RuntimeError(f"Speech recognition error: {error_text}")
            
            result = response.json()
            transcribed_text = result.get("result", "")
            
            logger.info(f"Audio transcribed: {len(transcribed_text)} characters")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    async def generate_speech(
        self,
        text: str,
        voice: str = "alena",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate speech from text using Yandex SpeechKit TTS.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (alena, filipp, ermil, jane, omazh, zahar, dasha)
            output_path: Path to save audio file
        
        Returns:
            Path to generated audio file
        """
        try:
            logger.debug(f"Generating speech with voice: {voice}")
            
            url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
            }
            
            data = {
                "text": text,
                "lang": "ru-RU",
                "voice": voice,
                "folderId": self.folder_id,
                "format": "oggopus",  # Telegram поддерживает OGG
            }
            
            response = requests.post(url, headers=headers, data=data, timeout=60)
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Yandex SpeechKit TTS error ({response.status_code}): {error_text}")
                raise RuntimeError(f"Speech synthesis error: {error_text}")
            
            # Default output path
            if output_path is None:
                from config import DATA_DIR
                import uuid
                output_path = DATA_DIR / f"tts_{uuid.uuid4()}.ogg"
            
            # Сохраняем аудио
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Speech generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise


class YandexVisionClient:
    """Client for Yandex Vision API."""
    
    def __init__(self):
        """Initialize the Yandex Vision client."""
        self.api_key = YANDEX_API_KEY.strip()
        self.folder_id = YANDEX_FOLDER_ID.strip()
        
        logger.info("Yandex Vision client initialized")
    
    async def analyze_image(
        self,
        image_path: Optional[Path] = None,
        image_url: Optional[str] = None,
        prompt: str = "Опиши это изображение подробно. Что ты видишь?"
    ) -> str:
        """
        Analyze an image using Yandex Vision API (OCR + Classification).
        Note: Yandex Vision only supports TEXT_DETECTION and CLASSIFICATION.
        For full image description, YandexGPT Pro with vision would be needed.
        
        Args:
            image_path: Local path to image
            image_url: URL to image
            prompt: Analysis prompt (used only for context, not sent to API)
        
        Returns:
            Image analysis result
        """
        try:
            logger.debug("Analyzing image with Yandex Vision (OCR + Classification)")
            
            # Подготовка изображения
            if image_path:
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
            elif image_url:
                # Скачиваем изображение по URL
                response = requests.get(image_url, timeout=30)
                image_data = base64.b64encode(response.content).decode("utf-8")
            else:
                raise ValueError("Either image_path or image_url must be provided")
            
            url = "https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze"
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "folderId": self.folder_id,
                "analyze_specs": [
                    {
                        "content": image_data,
                        "features": [
                            {"type": "TEXT_DETECTION"},
                            {"type": "CLASSIFICATION"},
                            {"type": "FACE_DETECTION"}  # Добавляем детекцию лиц
                        ]
                    }
                ]
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Yandex Vision API error ({response.status_code}): {error_text}")
                raise RuntimeError(f"Vision API error: {error_text}")
            
            result = response.json()
            
            # Извлекаем текст и классификацию
            analysis_text = self._format_vision_result(result, prompt)
            
            logger.info(f"Image analyzed: {len(analysis_text)} characters")
            return analysis_text
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
    
    def _format_vision_result(self, result: dict, original_prompt: str) -> str:
        """Format Yandex Vision result into readable text."""
        output_parts = []
        
        try:
            results = result.get("results", [])
            if not results:
                return "Не удалось распознать изображение."
            
            for res in results:
                # Текст с изображения
                text_detection = res.get("results", [])
                for detection in text_detection:
                    if detection.get("textDetection"):
                        pages = detection["textDetection"].get("pages", [])
                        for page in pages:
                            blocks = page.get("blocks", [])
                            for block in blocks:
                                lines = block.get("lines", [])
                                for line in lines:
                                    words = line.get("words", [])
                                    line_text = " ".join([w.get("text", "") for w in words])
                                    if line_text:
                                        output_parts.append(line_text)
                
                # Классификация
                    if detection.get("classification"):
                        properties = detection["classification"].get("properties", [])
                        if properties:
                            output_parts.append("\nКлассификация изображения:")
                            for prop in properties[:3]:  # Топ-3
                                name = prop.get("name", "")
                                probability = prop.get("probability", 0)
                                output_parts.append(f"- {name}: {probability:.2%}")
            
            if not output_parts:
                return (
                    "На изображении не обнаружено текста или объектов для распознавания.\n\n"
                    "ℹ️ Yandex Vision специализируется на распознавании текста (OCR) "
                    "и классификации изображений. Для детального описания изображений "
                    "используйте OpenAI GPT-4 Vision."
                )
            
            return "\n".join(output_parts)
            
        except Exception as e:
            logger.error(f"Error formatting vision result: {e}")
            return "Ошибка при обработке результатов распознавания."


# Global client instances
yandex_gpt_client = YandexGPTClient()
yandex_speechkit_client = YandexSpeechKitClient()
yandex_vision_client = YandexVisionClient()


