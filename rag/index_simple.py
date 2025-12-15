"""
Simple document index without vector embeddings.
Uses keyword-based search - perfect for Yandex API without embeddings.
"""

import re
from pathlib import Path
from typing import List, Dict
from utils.logging import logger
from config import DATA_DIR, DOCUMENTS_DIR, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP


class SimpleDocumentIndex:
    """Simple document index using keyword search."""
    
    def __init__(self):
        """Initialize the simple index."""
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []
        self.persist_directory = DATA_DIR / "simple_index"
        self.persist_directory.mkdir(exist_ok=True)
        
        logger.info("Simple document index initialized")
    
    def index_documents_directory(self, force_reindex: bool = False) -> int:
        """
        Index all documents in the documents directory.
        
        Args:
            force_reindex: Force re-indexing even if index exists
        
        Returns:
            Number of chunks indexed
        """
        try:
            # Проверяем, есть ли уже индекс
            index_file = self.persist_directory / "chunks.txt"
            if index_file.exists() and not force_reindex:
                logger.info("Loading existing index...")
                return self._load_index()
            
            # Находим все документы
            documents = list(DOCUMENTS_DIR.glob('*'))
            documents = [d for d in documents if d.is_file() and d.suffix in ['.pdf', '.txt', '.md']]
            
            if not documents:
                logger.warning("No documents found to index")
                return 0
            
            logger.info(f"Indexing {len(documents)} documents...")
            
            total_chunks = 0
            for doc_path in documents:
                chunks = self._process_document(doc_path)
                total_chunks += len(chunks)
                logger.info(f"Indexed {doc_path.name}: {len(chunks)} chunks")
            
            # Сохраняем индекс
            self._save_index()
            
            logger.info(f"Total chunks indexed: {total_chunks}")
            return total_chunks
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return 0
    
    def _process_document(self, doc_path: Path) -> List[str]:
        """Process a single document and split into chunks."""
        try:
            # Извлекаем текст
            if doc_path.suffix == '.pdf':
                text = self._extract_pdf_text(doc_path)
            else:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Разбиваем на чанки
            chunks = self._split_into_chunks(text)
            
            # Сохраняем с метаданными
            for chunk in chunks:
                self.chunks.append(chunk)
                self.metadata.append({'source': doc_path.name})
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {doc_path}: {e}")
            return []
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            from PyPDF2 import PdfReader
            
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        Similar to LangChain's RecursiveCharacterTextSplitter.
        """
        # Разбиваем на предложения
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Если добавление предложения не превысит лимит - добавляем
            if len(current_chunk) + len(sentence) <= RAG_CHUNK_SIZE:
                current_chunk += " " + sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Начинаем новый чанк с overlap
                if len(current_chunk) > RAG_CHUNK_OVERLAP:
                    # Берем последние N символов для overlap
                    overlap_text = current_chunk[-RAG_CHUNK_OVERLAP:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
        
        # Добавляем последний чанк
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _save_index(self):
        """Save index to disk."""
        try:
            chunks_file = self.persist_directory / "chunks.txt"
            metadata_file = self.persist_directory / "metadata.txt"
            
            # Сохраняем чанки
            with open(chunks_file, 'w', encoding='utf-8') as f:
                for chunk in self.chunks:
                    # Используем разделитель, который вряд ли встретится в тексте
                    f.write(chunk + "\n###CHUNK_SEPARATOR###\n")
            
            # Сохраняем метаданные
            import json
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False)
            
            logger.info(f"Index saved: {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _load_index(self) -> int:
        """Load index from disk."""
        try:
            chunks_file = self.persist_directory / "chunks.txt"
            metadata_file = self.persist_directory / "metadata.txt"
            
            if not chunks_file.exists():
                return 0
            
            # Загружаем чанки
            with open(chunks_file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.chunks = [c.strip() for c in content.split("###CHUNK_SEPARATOR###") if c.strip()]
            
            # Загружаем метаданные
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = [{'source': 'Unknown'}] * len(self.chunks)
            
            logger.info(f"Index loaded: {len(self.chunks)} chunks")
            return len(self.chunks)
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return 0
    
    def get_all_chunks(self) -> List[str]:
        """Get all indexed chunks."""
        return self.chunks
    
    def keyword_retrieve(self, question: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant chunks using keyword matching.
        
        Args:
            question: User's question
            top_k: Number of top chunks to return
        
        Returns:
            List of most relevant chunks
        """
        try:
            # Извлекаем ключевые слова из вопроса
            q_words = set(re.findall(r'\w+', question.lower()))
            
            # Убираем стоп-слова
            stop_words = {
                'что', 'как', 'где', 'когда', 'кто', 'какой', 'какая', 'какие',
                'это', 'для', 'или', 'и', 'в', 'на', 'с', 'по', 'из', 'у',
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'
            }
            q_words = {w for w in q_words if len(w) > 2 and w not in stop_words}
            
            if not q_words:
                logger.warning("No meaningful keywords found in question")
                return []
            
            # Подсчитываем совпадения
            scored = []
            for chunk in self.chunks:
                c_words = set(re.findall(r'\w+', chunk.lower()))
                
                # Базовый score - количество совпадающих слов
                score = len(q_words & c_words)
                
                # Бонус за точное совпадение фразы
                if question.lower() in chunk.lower():
                    score += 10
                
                if score > 0:
                    scored.append((score, chunk))
            
            # Сортируем по убыванию score
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # Возвращаем топ-K чанков
            result = [chunk for _, chunk in scored[:top_k]]
            
            logger.info(f"Found {len(result)} relevant chunks out of {len(self.chunks)} total")
            return result
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval: {e}")
            return []
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_documents": len(set(m['source'] for m in self.metadata)),
            "total_chunks": len(self.chunks),
            "persist_directory": str(self.persist_directory),
            "index_type": "simple_keyword"
        }
    
    def clear(self):
        """Clear the index."""
        self.chunks = []
        self.metadata = []
        logger.info("Index cleared")


# Global instance
simple_index = SimpleDocumentIndex()


