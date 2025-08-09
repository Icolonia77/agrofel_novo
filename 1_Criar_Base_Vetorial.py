# 1_Criar_Base_Vetorial.py — v2
import os
import re
import time
import json
from typing import List, Dict, Any

import google.generativeai as genai
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    Docx2txtLoader, PyPDFLoader
)
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ---- CONFIG ----------------------------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("A chave GOOGLE_API_KEY não foi encontrada no arquivo .env")
genai.configure(api_key=api_key)

CAMINHO_BASE_PROJETO = os.getenv("AGROFEL_BASE_DIR", "C:/Users/Usuario/Documents/agrofel_novo/")  # editável por env
PASTA_BULAS = os.path.join(CAMINHO_BASE_PROJETO, "documentos/")
CAMINHO_INDEX_FAISS = os.path.join(CAMINHO_BASE_PROJETO, "faiss_index_agrofel")
CAMINHO_META_JSON = os.path.join(CAMINHO_BASE_PROJETO, "faiss_index_agrofel_metadata.json")  # para auditoria
OS_TRY_TABLES = os.getenv("AGROFEL_PARSE_TABLES", "1") == "1"   # tentar extração de tabela

# ---- Helpers ---------------------------------------------------
def clean_text(txt: str) -> str:
    if not txt: return txt
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    # quebra hifenização comum: "apli-\ncação" -> "aplicação"
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
    return txt.strip()

def try_extract_tables_from_pdf(path_pdf: str) -> List[Document]:
    """Tenta extrair tabelas de PDF com camelot/tabula; retorna cada tabela como Document separado."""
    docs = []
    try:
        import camelot  # pip install camelot-py[cv]
        tables = camelot.read_pdf(path_pdf, pages="all", flavor="lattice")
        for i, t in enumerate(tables):
            txt = t.df.to_csv(index=False)
            md = {"source": os.path.basename(path_pdf), "page": int(t.parsing_report.get("page", -1)), "type": "table"}
            docs.append(Document(page_content=txt, metadata=md))
        return docs
    except Exception:
        # fallback para tabula
        try:
            import tabula  # pip install tabula-py (requer Java)
            dfs = tabula.read_pdf(path_pdf, pages="all", multiple_tables=True)
            for i, df in enumerate(dfs):
                txt = df.to_csv(index=False)
                md = {"source": os.path.basename(path_pdf), "page": -1, "type": "table"}
                docs.append(Document(page_content=txt, metadata=md))
        except Exception:
            pass
    return docs

def load_all_documents() -> List[Document]:
    if not os.path.exists(PASTA_BULAS) or not os.listdir(PASTA_BULAS):
        raise FileNotFoundError(f"A pasta '{PASTA_BULAS}' não existe ou está vazia.")

    docs: List[Document] = []
    arquivos = [f for f in os.listdir(PASTA_BULAS) if f.lower().endswith((".pdf", ".docx"))]
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo PDF/DOCX encontrado em '{PASTA_BULAS}'.")

    for nome in sorted(arquivos):
        caminho = os.path.join(PASTA_BULAS, nome)
        print(f"Processando: {nome}")
        if nome.lower().endswith(".pdf"):
            # PyPDFLoader retorna por página, com metadata.page
            loader = PyPDFLoader(caminho)
            dpages = loader.load()
            for d in dpages:
                d.page_content = clean_text(d.page_content)
                d.metadata["source"] = nome
                d.metadata["type"] = d.metadata.get("type", "page")
            docs.extend(dpages)
            # Tabelas (opcional)
            if OS_TRY_TABLES:
                docs.extend(try_extract_tables_from_pdf(caminho))
        else:
            loader = Docx2txtLoader(caminho)
            dlist = loader.load()
            # Docx2txt não entrega páginas; marcamos como "doc"
            for d in dlist:
                d.page_content = clean_text(d.page_content)
                d.metadata["source"] = nome
                d.metadata["type"] = d.metadata.get("type", "doc")
                d.metadata["page"] = d.metadata.get("page", -1)
            docs.extend(dlist)

    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    # Divisores que respeitam títulos/seções quando possível
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=180,
        separators=[ "\n### ", "\n## ", "\n# ", "\n\n", "\n", " ", "" ]
    )
    chunks = splitter.split_documents(docs)
    # Adiciona chunk_id sequencial por arquivo para facilitar citações
    counters: Dict[str, int] = {}
    for d in chunks:
        src = d.metadata.get("source", "desconhecido")
        counters[src] = counters.get(src, 0) + 1
        d.metadata["chunk_id"] = counters[src]
    return chunks

# ---- Main ------------------------------------------------------
def criar_base_de_conhecimento():
    print(f"Iniciando ingestão em '{PASTA_BULAS}'...")
    t0 = time.time()

    docs = load_all_documents()
    print(f"Arquivos carregados: {len(set([d.metadata['source'] for d in docs]))} | documentos brutos: {len(docs)}")

    chunks = chunk_documents(docs)
    print(f"Chunks criados: {len(chunks)} (≈{round(sum(len(c.page_content) for c in chunks)/len(chunks))} chars/chunk)")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # IMPORTANTE: normalize_L2=True → distância coseno em FAISS
    vectordb = FAISS.from_documents(chunks, embeddings, normalize_L2=True)
    vectordb.save_local(CAMINHO_INDEX_FAISS)

    # Salva um JSON leve com metadados para auditoria
    meta = [
        {"source": c.metadata.get("source"), "page": c.metadata.get("page", -1),
         "chunk_id": c.metadata.get("chunk_id"), "type": c.metadata.get("type")}
        for c in chunks
    ]
    with open(CAMINHO_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSUCESSO! Índice FAISS salvo em '{CAMINHO_INDEX_FAISS}'. Metadados: '{CAMINHO_META_JSON}'.")
    print(f"Tempo total: {round(time.time() - t0, 1)}s")

if __name__ == "__main__":
    criar_base_de_conhecimento()
