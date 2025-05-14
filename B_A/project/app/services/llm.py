from app.config import settings
import logging
from langchain_mistralai.chat_models import ChatMistralAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm():
    try:
        logger.info("Loading Mistral LLM via LangChain")
        llm = ChatMistralAI(
            api_key=settings.mistral_api_key,
            model=settings.mistral_model,
            temperature=0.2,
            max_tokens=200
        )
        return llm
    except Exception as e:
        logger.error(f"LLM loading failed: {str(e)}")
        raise

llm = get_llm()