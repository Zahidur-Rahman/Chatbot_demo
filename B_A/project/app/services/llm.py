from app.config import settings
import logging
from langchain_mistralai.chat_models import ChatMistralAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm():
    try:
        logger.info(f"Loading Mistral LLM model: {settings.mistral_model}")
        llm = ChatMistralAI(
            api_key=settings.mistral_api_key,
            model=settings.mistral_model,
            temperature=0.2,  # Lower temperature for more focused responses
            max_tokens=1000,  # Increased for handling context
            top_p=0.95  # For better response quality
        )
        logger.info("Mistral LLM loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"LLM loading failed: {str(e)}")
        raise

llm = get_llm()