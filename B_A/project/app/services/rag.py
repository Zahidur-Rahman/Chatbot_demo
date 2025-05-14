from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import MistralAIEmbeddings
from app.config import settings
from app.services.llm import get_llm
from typing import List, Optional

class RAGService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        
        self.llm = get_llm()
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=settings.mistral_api_key
        )
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self, urls: Optional[List[str]] = None):
        urls = urls or ["https://en.wikipedia.org/wiki/Large_language_model"]
        loader = WebBaseLoader(urls)
        docs = loader.load()
        texts = self.text_splitter.split_documents(docs)
        return FAISS.from_documents(texts, self.embeddings)

    def query(self, question: str, k: int = 2) -> dict:
        prompt_template = """Answer concisely based on the context:
        Context: {context}
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }