from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import re

load_dotenv()

clean_text = 1

def limpar_texto(texto):
    texto = re.sub(r'­\n', '', texto)      # hifenização invisível
    texto = re.sub(r'-\n', '', texto)       # hifenização normal
    texto = re.sub(r'\d+\s*•\s*JK\s*\n', '', texto)   # "430 • JK"
    texto = re.sub(r'JK\s*•\s*\d+\s*\n', '', texto)   # "JK • 183"
    texto = re.sub(r'(?<!\n)\n(?!\n)', ' ', texto)     # quebras no meio de frase
    texto = re.sub(r' +', ' ', texto)
    return texto.strip()

def initialize_kb(embeddings_model):
    vector_store = Chroma(
        collection_name="embeddings_jk",
        embedding_function=embeddings_model,
        persist_directory="./chroma_db"
    )

    if vector_store._collection.count() == 0:
        # Só carrega PDF e processa se realmente precisar

        arquivo = 'jk_couto_2ed.pdf'
        loader = PyMuPDFLoader(f"./kb/raw/{arquivo}")
        all_pages = loader.load()

        if clean_text == 0:
            for doc in all_pages:
                doc.page_content = limpar_texto(doc.page_content)

        text_splitter = SemanticChunker(
            embeddings=embeddings_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85
        )

        chunks = text_splitter.split_documents(all_pages)
        print(f"Documento dividido em {len(chunks)} partes.")

        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            page = chunk.metadata.get('page', None)
            chunk.metadata['page_display'] = f"p. {int(page) + 1}" if page is not None else "p. ?"
        
        vector_store.add_documents(documents=chunks)
    else:
        # Execuções seguintes: recupera do disco para o BM25
        print("KB já existente, carregando do disco.")
        result = vector_store.get()
        chunks = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(result["documents"], result["metadatas"])
        ]
        print(f"{len(chunks)} chunks recuperados do disco.")

    return vector_store, chunks
