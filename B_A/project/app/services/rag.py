from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_mistralai.embeddings import MistralAIEmbeddings
from app.config import settings
from app.services.llm import get_llm
from typing import List, Optional
import logging
import time
import os
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        self.llm = get_llm()
        self.embeddings = MistralAIEmbeddings(
            model=settings.mistral_embedding_model,
            mistral_api_key=settings.mistral_api_key
        )
        
        # Fixed URLs and PDFs for context
        self.context_urls = [
            "https://en.wikipedia.org/wiki/Large_language_model",
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Bangladesh%E2%80%93India_cricket_rivalry"
        ]
        
        # Add your PDF paths here
        self.pdf_paths = [
                "/home/zahid/Desktop/B_RAG_API_practice_2/B_A/project/docs/mcp.pdf",
        ]
        
        # Initialize vector store with both URLs and PDFs
        self.vector_store = self._initialize_vector_store()
        

    def _initialize_vector_store(self):
        """Initialize vector store with both URLs and PDFs"""
        try:
            all_docs = []
            
            # Process URLs
            if self.context_urls:
                logger.info(f"Loading documents from URLs: {self.context_urls}")
                web_loader = WebBaseLoader(self.context_urls)
                all_docs.extend(web_loader.load())
            
            # Process PDFs
            if self.pdf_paths:
                logger.info(f"Loading documents from PDFs: {self.pdf_paths}")
                for pdf_path in self.pdf_paths:
                    if os.path.exists(pdf_path):
                        pdf_loader = PyPDFLoader(pdf_path)
                        all_docs.extend(pdf_loader.load())
                    else:
                        logger.warning(f"PDF file not found: {pdf_path}")
            
            if not all_docs:
                raise ValueError("No documents loaded from URLs or PDFs")
            
            texts = self.text_splitter.split_documents(all_docs)
            logger.info(f"Created {len(texts)} text chunks")
            
            vector_store = FAISS.from_documents(texts, self.embeddings)
            logger.info("Vector store initialized successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(self, question: str, k: int = 4) -> dict:
        """Query the RAG system with a question"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        prompt_template = """You are a helpful AI assistant. Follow these guidelines:
        1. If the question is directly related to the provided context (about cricket rivalry, LLMs, or AI), use the context to answer.
        2. If the question is NOT related to the context topics, IGNORE the context completely and use your general knowledge.
        3. When using general knowledge, start with "Based on my general knowledge:" and provide a direct, helpful answer.
        4. Do not mention the context if it's not relevant to the question.
        5. Be clear and concise in your responses.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={
                        "k": k,
                        "fetch_k": k * 2
                    }
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            logger.info(f"Processing question: {question}")
            result = qa_chain.invoke({"query": question})
            
            return {
                "answer": result["result"],
                "sources": [doc.metadata for doc in result["source_documents"]]
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise