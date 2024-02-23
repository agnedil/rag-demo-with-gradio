import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List

from langchain_community.llms import Replicate    # importing from langchain depricated; use langchain_community for several modules here
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


class ElevatedRagChain:
    '''
    Class ElevatedRagChain integrates various components from the langchain library to build
    an advanced retrieval-augmented generation (RAG) system designed to process documents
    by reading in, chunking, embedding, and adding their chunk embeddings to FAISS vector store
    for efficient retrieval. It uses the embeddings to retrieve relevant document chunks
    in response to user queries.
    The chunks are retrieved using an ensemble retriever (BM25 retriever + FAISS retriver)
    and passed through a Cohere reranker before being used as context
    for generating answers using a Llama 2 large language model (LLM). 
    '''
    def __init__(self) -> None:
        '''
        Initialize the class with predefined model, embedding function, weights, and top_k value
        '''
        self.llama2_70b   = 'meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48'
        self.embed_func   = CohereEmbeddings(model="embed-english-light-v3.0")
        self.bm25_weight  = 0.6
        self.faiss_weight = 0.4
        self.top_k        = 5


    def add_pdfs_to_vectore_store(
            self,
            pdf_links: List,
            chunk_size: int=1500,
            ) -> None:
        '''
        Processes PDF documents by loading, chunking, embedding, and adding them to a FAISS vector store.
        Build an advanced RAG system  
        Args:
            pdf_links (List): list of URLs pointing to the PDF documents to be processed
            chunk_size (int, optional): size of text chunks to split the documents into, defaults to 1500
        '''        
        # load pdfs
        self.raw_data = [ OnlinePDFLoader(doc).load()[0] for doc in pdf_links ]

        # chunk text
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        self.split_data    = self.text_splitter.split_documents(self.raw_data)

        # add chunks to BM25 retriever
        self.bm25_retriever   = BM25Retriever.from_documents(self.split_data)
        self.bm25_retriever.k = self.top_k

        # embed and add chunks to vectore store
        self.vector_store     = FAISS.from_documents(self.split_data, self.embed_func)
        self.faiss_retriever  = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        print("All PDFs processed and added to vectore store.")
        
        # build advanced RAG system
        self.build_elevated_rag_system()
        print("RAG system is built successfully.")


    def build_elevated_rag_system(self) -> None:
        '''
        Build an advanced RAG system from different components:
        * BM25 retriever
        * FAISS vector store retriever
        * Llama 2 model
        '''
        # combine BM25 and FAISS retrievers into an ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[self.bm25_weight, self.faiss_weight]
        )

        # use reranker to improve retrieval quality
        self.reranker = CohereRerank(top_n=5)
        self.rerank_retriever = ContextualCompressionRetriever(    # combine ensemble retriever and reranker
            base_retriever=self.ensemble_retriever,
            base_compressor=self.reranker,
        )

        # define prompt template for the language model
        RAG_PROMPT_TEMPLATE = """\
        Use the following context to provide a detailed technical answer the user's question.
        Do not use an introduction similar to "Based on the provided documents, ...", just answer the question.
        If you don't know the answer, please respond with "I don't know".

        Context:
        {context}

        User's question:
        {question}
        """
        self.rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.str_output_parser = StrOutputParser()

        # parallel execution of context retrieval and question passing
        self.entry_point_and_elevated_retriever = RunnableParallel(
            {
                "context" : self.rerank_retriever,
                "question" : RunnablePassthrough()
            }
        )

        # initialize Llama 2 model with specific parameters
        self.llm = Replicate(
            model=self.llama2_70b,
            model_kwargs={"temperature": 0.5,"top_p": 1, "max_new_tokens":1000}
        )

        # chain components to form final elevated RAG system using LangChain Expression Language (LCEL)
        self.elevated_rag_chain = self.entry_point_and_elevated_retriever | self.rag_prompt | self.llm #| self.str_output_parser
