"""
RAG Query Handler.
Handles queries against the knowledge base with context-aware responses.
"""

from typing import List, Dict, Optional

from utils.logging import logger
from config import RAG_TOP_K, API_PROVIDER

# Динамический импорт в зависимости от провайдера
if API_PROVIDER == "yandex":
    from rag.index_simple import simple_index as knowledge_index
    from services.yandex_client import yandex_gpt_client as ai_client
else:
    from rag.index import vector_index as knowledge_index
    from services.openai_client import openai_client as ai_client


async def query_knowledge_base(
    query: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Query the knowledge base and generate response.
    
    Args:
        query: User's query
        conversation_history: Previous conversation messages
    
    Returns:
        Generated response based on retrieved context
    """
    try:
        # Search for relevant documents
        logger.debug(f"Searching knowledge base for: {query} (provider: {API_PROVIDER})")
        
        if API_PROVIDER == "yandex":
            # Keyword-based search for Yandex
            relevant_chunks = knowledge_index.keyword_retrieve(query, top_k=RAG_TOP_K)
            
            if not relevant_chunks:
                logger.warning("No relevant documents found, using fallback")
                return await _fallback_response(query, conversation_history)
            
            context = "\n\n".join(relevant_chunks)
        else:
            # Vector-based search for OpenAI
            results = knowledge_index.similarity_search_with_score(query, k=RAG_TOP_K)
            
            if not results:
                logger.warning("No relevant documents found, using fallback")
                return await _fallback_response(query, conversation_history)
            
            # Prepare context from retrieved documents
            context = _prepare_context(results)
        
        # Generate response with context
        response = await _generate_rag_response(
            query=query,
            context=context,
            conversation_history=conversation_history
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        # Fallback to regular GPT response
        return await _fallback_response(query, conversation_history)


def _prepare_context(results: List[tuple]) -> str:
    """
    Prepare context from search results.
    
    Args:
        results: List of (document, score) tuples
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content.strip()
        
        context_parts.append(
            f"[Источник {i}: {source}]\n{content}\n"
        )
    
    return "\n".join(context_parts)


async def _generate_rag_response(
    query: str,
    context: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Generate response using RAG context.
    
    Args:
        query: User's query
        context: Retrieved context from knowledge base
        conversation_history: Previous conversation
    
    Returns:
        Generated response
    """
    # Build prompt with context
    system_prompt = """Ты - технический консультант-эксперт по люкам и дождеприемникам компании "ЛИТЛИДЕР".

ТВОЯ РОЛЬ:
- Помогаешь инженерам, проектировщикам, прорабам и снабженцам подбирать продукцию
- Расшифровываешь технические маркировки типа ТМ(Д400)-2-7-9-60
- Предоставляешь точные технические характеристики, веса, размеры
- Объясняешь различия между типами продукции (плавающие/обычные люки, классы нагрузки)
- Помогаешь с подбором оборудования для конкретных условий эксплуатации

ВАЖНЫЕ ПРАВИЛА:
1. Используй ТОЛЬКО информацию из технического каталога ниже
2. Для технических характеристик цитируй точные данные (вес, размеры, маркировку)
3. Если в каталоге нет информации - честно скажи об этом
4. Всегда указывай артикул/маркировку продукции
5. Объясняй технические термины понятным языком
6. Структурируй ответы: характеристики, применение, преимущества

КОНТЕКСТ ИЗ ТЕХНИЧЕСКОГО КАТАЛОГА:
{context}

Используй этот контекст для точного и профессионального ответа."""
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": system_prompt.format(context=context)
        }
    ]
    
    # Add conversation history if available
    if conversation_history:
        # Limit history to avoid token limits
        recent_history = conversation_history[-6:]  # Last 3 exchanges
        messages.extend(recent_history)
    
    # Add current query
    messages.append({
        "role": "user",
        "content": query
    })
    
    # Generate response
    response = await ai_client.generate_text_response(messages)
    
    return response


async def _fallback_response(
    query: str,
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Fallback to regular GPT response when RAG fails.
    
    Args:
        query: User's query
        conversation_history: Previous conversation
    
    Returns:
        Generated response
    """
    logger.info("Using fallback response (no RAG context)")
    
    system_message = {
        "role": "system",
        "content": """Ты - технический консультант по люкам и дождеприемникам компании "ЛИТЛИДЕР". 
        
База знаний не содержит информации по этому вопросу.
Если это вопрос о продукции - извинись и попроси пользователя загрузить технический каталог.
Если это общий вопрос - можешь ответить, но предупреди, что ответ не из каталога."""
    }
    
    messages = [system_message]
    
    if conversation_history:
        messages.extend(conversation_history[-6:])
    
    messages.append({
        "role": "user",
        "content": query
    })
    
    response = await ai_client.generate_text_response(messages)
    
    return f"⚠️ Технический каталог не содержит информации по этому вопросу.\n\n{response}\n\n💡 Убедитесь, что файл Litlider_Katalog_VCHSHG_2025.pdf загружен в систему."


async def add_document_to_knowledge_base(file_path: str) -> dict:
    """
    Add a document to the knowledge base.
    
    Args:
        file_path: Path to document file
    
    Returns:
        Dictionary with status and details
    """
    try:
        from pathlib import Path
        from rag.loader import document_loader
        
        # Load document
        file_path = Path(file_path)
        documents = document_loader.load_document(file_path)
        
        # Add to index
        # Note: This function currently only works with vector index
        # For Yandex, re-index the entire directory instead
        if API_PROVIDER == "yandex":
            logger.warning("For Yandex, please re-index the entire directory instead of adding single documents")
            return {
                "success": False,
                "error": "Use directory re-indexing for Yandex",
                "message": "Для Yandex переиндексируйте всю директорию"
            }
        
        knowledge_index.add_documents(documents)
        
        logger.info(f"Added {file_path.name} to knowledge base")
        
        return {
            "success": True,
            "file": file_path.name,
            "chunks": len(documents),
            "message": f"Документ {file_path.name} успешно добавлен ({len(documents)} фрагментов)"
        }
        
    except Exception as e:
        logger.error(f"Error adding document to knowledge base: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Ошибка при добавлении документа: {e}"
        }


def get_knowledge_base_stats() -> dict:
    """
    Get statistics about the knowledge base.
    
    Returns:
        Dictionary with statistics
    """
    return knowledge_index.get_stats()

