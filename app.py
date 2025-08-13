# app.py
import os
import re
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_core.pydantic_v1 import BaseModel, Field

import google.generativeai as genai

# -------------------- CONFIG --------------------
load_dotenv()
st.set_page_config(page_title="Assistente de Campo Agrofel", page_icon="🌿", layout="wide")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY não encontrado no .env/Secrets.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

AGROFEL_INDEX_DIR = os.getenv("AGROFEL_INDEX_DIR", "faiss_index_agrofel")
UNKNOWN = "Não especificado"

# -------------------- SCHEMA --------------------
class ProdutoExtraido(BaseModel):
    nome_produto: str = Field(default=UNKNOWN)
    culturas: List[str] = Field(default_factory=list)
    pragas: List[str] = Field(default_factory=list)
    ingrediente_ativo: str = Field(default=UNKNOWN)
    dose: str = Field(default=UNKNOWN)
    modo_aplicacao: str = Field(default=UNKNOWN)
    intervalo_seguranca: str = Field(default=UNKNOWN)
    modo_acao: str = Field(default=UNKNOWN)
    numero_max_aplicacoes: str = Field(default=UNKNOWN)  # <--- NOVO
    efeito_lavoura: str = Field(default=UNKNOWN)
    fontes: List[Dict[str, Any]] = Field(default_factory=list)  # [{"source":"arquivo.pdf","page":12}, ...]

# -------------------- LOAD COMPONENTS --------------------
@st.cache_resource(show_spinner="Carregando índice/LLM...")
def _load_components_cached():
    return load_components()

def load_components():
    if not os.path.exists(AGROFEL_INDEX_DIR):
        raise FileNotFoundError(
            f"Índice FAISS não encontrado em '{AGROFEL_INDEX_DIR}'. "
            f"Rode 'python 1_Criar_Base_Vetorial.py' antes."
        )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db: FAISS = FAISS.load_local(AGROFEL_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    # Reconstrói docs p/ BM25 e mantém todos em memória
    ids = list(db.index_to_docstore_id.values())
    all_docs: List[Document] = []
    for _id in ids:
        d = db.docstore.search(_id)
        if d and d.page_content:
            all_docs.append(d)

    bm25 = BM25Retriever.from_documents(all_docs) if all_docs else None
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1)
    return db, bm25, llm, all_docs

# -------------------- RAG: recuperação --------------------
def gerar_consultas(query: str, llm) -> List[str]:
    prompt = ChatPromptTemplate.from_template(
        "Gere 4 variações curtas para busca em bulas (termos de cultura, praga e ingrediente ativo). "
        "Uma por linha.\nPergunta: {p}\nConsultas:"
    )
    out = (prompt | llm | StrOutputParser()).invoke({"p": query}).strip().split("\n")
    return [c.strip() for c in out if c.strip()][:4]

def rrf_fusion(rankings: List[List[Document]], k: int = 12) -> List[Document]:
    scores: Dict[str, float] = {}
    items: Dict[str, Document] = {}
    for lst in rankings:
        for r, doc in enumerate(lst, start=1):
            key = f"{doc.metadata.get('source','?')}|{doc.metadata.get('page',-1)}|{doc.metadata.get('chunk_id',-1)}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + r)
            items[key] = doc
    ordered = sorted(items.keys(), key=lambda k: scores[k], reverse=True)
    return [items[k] for k in ordered[:k]]

def recuperar_contexto(query: str, lock_to_source: Optional[str] = None) -> List[Document]:
    db, bm25, llm, _ = _load_components_cached()
    queries = gerar_consultas(query, llm)

    faiss_lists = []
    for q in queries:
        docs = db.max_marginal_relevance_search(q, k=8, fetch_k=40, lambda_mult=0.2)
        if lock_to_source:
            docs = [d for d in docs if d.metadata.get("source") == lock_to_source]
        faiss_lists.append(docs)

    bm25_lists = []
    if bm25:
        bm25.k = 6
        for q in queries:
            docs = bm25.get_relevant_documents(q)
            if lock_to_source:
                docs = [d for d in docs if d.metadata.get("source") == lock_to_source]
            bm25_lists.append(docs)

    return rrf_fusion(faiss_lists + bm25_lists, k=14)

def rerank_llm(query: str, docs: List[Document], top_k: int = 8) -> List[Document]:
    if not docs:
        return []
    _, _, llm, _ = _load_components_cached()
    snippet = "\n\n---\n\n".join(
        [f"[{i}] (src={d.metadata.get('source')}, page={d.metadata.get('page',-1)})\n{d.page_content[:1600]}"
         for i, d in enumerate(docs)]
    )
    prompt = f"""
Pergunta: "{query}"

Abaixo, trechos de bulas com índices. Selecione até {top_k} mais relevantes priorizando:
Dose, Intervalo de segurança (ou "período de carência"), **Número máximo de aplicações**,
**Modo de aplicação / Forma de aplicação / Modo de uso / Aplicação terrestre/aérea/costal/tratorizado /
Equipamento de aplicação**, Cultura, Praga e tabelas.
Retorne SOMENTE JSON com índices. Ex: [0,2,5]

Trechos:
{snippet}
"""
    try:
        ids_json = llm.invoke(prompt).content.strip()
        import json as _json
        idxs = _json.loads(ids_json)
        idxs = [i for i in idxs if isinstance(i, int) and 0 <= i < len(docs)]
        return [docs[i] for i in idxs][:top_k]
    except Exception:
        return docs[:top_k]

# -------------------- AUGMENT: headers + heurísticas --------------------
def _dedup_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page", -1), d.metadata.get("chunk_id", -1))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

def augment_with_headers(docs: List[Document]) -> List[Document]:
    """Garante que o cabeçalho (primeiro chunk / primeira página) das bulas esteja no contexto."""
    _, _, _, all_docs = _load_components_cached()
    if not docs or not all_docs:
        return docs
    sources = {d.metadata.get("source") for d in docs if d.metadata.get("source")}
    headers = []
    for d in all_docs:
        if d.metadata.get("source") in sources and (
            d.metadata.get("chunk_id") == 1 or d.metadata.get("page", 999) in (0, 1)
        ):
            headers.append(d)
    return _dedup_docs(headers + docs)

def _guess_product_name_from_docs(docs: List[Document]) -> Optional[str]:
    text = "\n".join([d.page_content for d in docs if d.page_content])
    # 1) Campo explícito
    m = re.search(r"(?:nome(?:\s+do)?\s+produto|nome\s+comercial|marca\s+comercial)\s*[:\-]\s*([^\n]{2,80})", text, flags=re.I)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    # 2) Linha título em destaque
    for line in text.splitlines():
        s = line.strip()
        if 3 <= len(s) <= 60 and (s.isupper() or s.istitle()) and not s.endswith(":"):
            return s
    # 3) Fallback: nome do arquivo
    src = docs[0].metadata.get("source") if docs else None
    if src:
        base = re.sub(r"\.(pdf|docx)$", "", src, flags=re.I)
        base = re.sub(r"[_\-]+", " ", base).strip()
        return base
    return None

def _extract_intervalo(text: str) -> Optional[str]:
    m = re.search(r"(?:intervalo\s+de\s+segurança|período\s+de\s+carência)\s*[:\-]?\s*([^\n]{1,80})", text, flags=re.I)
    return m.group(1).strip() if m else None

def _extract_dose(text: str) -> Optional[str]:
    m = re.search(r"(\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?\s*(?:mL|L|g|kg)\s*(?:i\.a\.\s*)?/ha)",
                  text, flags=re.I)
    return m.group(1).replace(",", ".").strip() if m else None

def _extract_modo_aplicacao(text: str) -> Optional[str]:
    """
    Extrai um resumo curto do modo de aplicação (terrestre/aérea/costal/tratorizado, via foliar, pré/pós-emergência)
    e volume de calda, a partir de cabeçalhos e palavras-chave.
    """
    t = text
    tl = t.lower()

    parts = []
    header_re = re.compile(
        r"(?:modo\s+de\s+aplica[cç][aã]o|forma\s+de\s+aplica[cç][aã]o|modo\s+de\s+uso|"
        r"equipamento\s+de\s+aplica[cç][aã]o)\s*[:\-]?\s*([^\n]{1,160})",
        flags=re.I
    )
    for m in header_re.finditer(t):
        frag = re.sub(r"\s+", " ", m.group(1)).strip(" .;")
        if frag:
            parts.append(frag)

    flags = []
    if re.search(r"aplica[cç][aã]o\s+terrestre", tl): flags.append("Terrestre")
    if re.search(r"aplica[cç][aã]o\s+a[ée]rea", tl): flags.append("Aérea")
    if "costal" in tl: flags.append("Costal")
    if re.search(r"tratorizad[oa]", tl): flags.append("Tratorizado")
    if re.search(r"turbo\s*atomizador|atomizador", tl): flags.append("Turbo atomizador")
    if re.search(r"via\s+foliar", tl): flags.append("Via foliar")
    if re.search(r"p[óo]s-?\s*emerg[êe]ncia", tl): flags.append("Pós-emergência")
    if re.search(r"pr[ée]-?\s*emerg[êe]ncia", tl): flags.append("Pré-emergência")

    vol = re.search(r"volume\s+de\s+calda\s*[:\-]?\s*([^\n]{1,80})", t, flags=re.I)
    vol_txt = vol.group(1).strip() if vol else None

    chunks = []
    if flags:
        seen = set(); ordered = []
        for f in flags:
            if f not in seen:
                seen.add(f); ordered.append(f)
        chunks.append(", ".join(ordered))
    if vol_txt:
        chunks.append(f"Volume de calda {vol_txt}")
    if parts:
        seen2 = set(); dedup = []
        for p in parts:
            p2 = p.lower()
            if p2 not in seen2:
                seen2.add(p2); dedup.append(p)
        extra = "; ".join(dedup)
        if extra:
            chunks.append(extra)

    out = " | ".join(chunks).strip()
    return out or None

def _extract_num_max_aplicacoes(text: str) -> Optional[str]:
    """
    Tenta capturar algo como:
    - "número máximo de aplicações: 2"
    - "máximo de 3 aplicações por ciclo"
    - "até 2 aplicações por safra"
    - "no máximo 4 aplicações"
    """
    patterns = [
        r"(?:n[úu]mero\s+m[aá]ximo\s+de\s+aplica[cç][õo]es|m[aá]ximo\s+de\s+aplica[cç][õo]es|limite\s+de\s+aplica[cç][õo]es)[^0-9]{0,20}(\d+)([^.\n]{0,30})",
        r"(?:at[ée]|no\s+m[aá]ximo\s+de)\s+(\d+)\s+aplica[cç][õo]es([^.\n]{0,30})",
        r"n[úu]mero\s+de\s+aplica[cç][õo]es[^0-9]{0,15}(\d+)([^.\n]{0,30})"
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            n = m.group(1)
            tail = (m.group(2) or "").strip()
            # se mencionar ciclo/safra/ano, adiciona
            hint = ""
            if re.search(r"(ciclo|safra|ano)", tail, flags=re.I):
                hint = " por ciclo/safra"
            return f"{n} aplicações{hint}"
    return None

def enrich_missing_fields(prod: ProdutoExtraido, docs: List[Document]) -> ProdutoExtraido:
    text = "\n".join([d.page_content for d in docs if d.page_content])
    if not prod.nome_produto or prod.nome_produto == UNKNOWN:
        hint = _guess_product_name_from_docs(docs)
        if hint:
            prod.nome_produto = hint
    if not prod.intervalo_seguranca or prod.intervalo_seguranca == UNKNOWN:
        iv = _extract_intervalo(text)
        if iv:
            prod.intervalo_seguranca = iv
    if not prod.dose or prod.dose == UNKNOWN:
        ds = _extract_dose(text)
        if ds:
            prod.dose = ds
    if not prod.modo_aplicacao or prod.modo_aplicacao == UNKNOWN:
        ma = _extract_modo_aplicacao(text)
        if ma:
            prod.modo_aplicacao = ma
    if not prod.numero_max_aplicacoes or prod.numero_max_aplicacoes == UNKNOWN:
        mx = _extract_num_max_aplicacoes(text)
        if mx:
            prod.numero_max_aplicacoes = mx
    return prod

# -------------------- EXTRAÇÃO E FORMATAÇÃO --------------------
def build_citations(docs: List[Document]) -> List[Dict[str, Any]]:
    cites = [{"source": d.metadata.get("source"), "page": d.metadata.get("page", -1)} for d in docs]
    seen = set(); uniq = []
    for c in cites:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq

def extrair_produto(query: str, docs: List[Document]) -> ProdutoExtraido:
    _, _, llm, _ = _load_components_cached()
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    fontes = build_citations(docs)

    prompt = f"""
Você é agrônomo. Extraia APENAS o objeto (schema ProdutoExtraido) com base nos TRECHOS.
- Priorize: Nome do produto (nome comercial), Cultura, Praga, Dose, Intervalo de segurança (ou "período de carência"),
  **Número máximo de aplicações**, Modo de aplicação, Modo de ação e Ingrediente ativo.
- Se o nome comercial não aparecer claramente, utilize o melhor indício (ex.: linha de título ou nome do arquivo sem extensão).
- Não invente; quando realmente não houver, use "{UNKNOWN}".
- Se houver múltiplos produtos, escolha o mais pertinente à pergunta.

PERGUNTA: "{query}"

TRECHOS:
{context}
"""
    prod: ProdutoExtraido = llm.with_structured_output(ProdutoExtraido).invoke(prompt)
    prod.fontes = fontes
    prod.dose = prod.dose.replace(" – ", "-").replace(" a ", " - ").strip()
    return prod

def _fmt_list(xs: List[str]) -> str:
    return ", ".join(xs) if xs else UNKNOWN

def formatar_markdown(prod: ProdutoExtraido) -> str:
    """
    Formato mais didático (linguagem simples). Não inventa valores;
    apenas explica o significado de cada campo para orientar a ação.
    """
    linhas = [
        f"**Nome do Produto:** {prod.nome_produto or UNKNOWN}  \n"
        f"_Como você verá na embalagem/bula._",

        f"**Cultura(s) atendida(s):** {_fmt_list(prod.culturas)}  \n"
        f"_Em quais lavouras este produto pode ser usado._",

        f"**Praga(s) alvo:** {_fmt_list(prod.pragas)}  \n"
        f"_O que o produto combate. Confira se a sua praga está nesta lista antes de aplicar._",

        f"**Ingrediente Ativo:** {prod.ingrediente_ativo or UNKNOWN}  \n"
        f"_Substância responsável pelo efeito. Útil para comparar produtos equivalentes._",

        f"**Dose (quanto aplicar):** {prod.dose or UNKNOWN}  \n"
        f"_Quantidade por hectare. Use medidores/balança. Não ultrapasse a faixa indicada na bula._",

        f"**Modo de Aplicação (como aplicar):** {prod.modo_aplicacao or UNKNOWN}  \n"
        f"_Ex.: terrestre/aérea/costal/tratorizado; via foliar; pré/pós-emergência; volume de calda. "
        f"Ajuste bicos e pressão conforme a bula._",

        f"**Intervalo de Segurança (para colher):** {prod.intervalo_seguranca or UNKNOWN}  \n"
        f"_Número de dias entre a última aplicação e a colheita. Respeite para evitar resíduos._",

        f"**Modo de Ação (como age):** {prod.modo_acao or UNKNOWN}  \n"
        f"_Ex.: contato, sistêmico, seletivo/não seletivo. Ajuda a planejar rotação de mecanismos de ação._",

        f"**Número Máximo de Aplicações:** {prod.numero_max_aplicacoes or UNKNOWN}  \n"
        f"_Quantas vezes pode aplicar por ciclo/safra segundo a bula. Não ultrapasse este limite._",

        f"**Efeito na Lavoura / Observações:** {prod.efeito_lavoura or UNKNOWN}  \n"
        f"_Informações sobre fitotoxicidade e cuidados especiais, quando houver._",
    ]

    if getattr(prod, "fontes", None):
        partes = []
        for c in prod.fontes[:6]:
            src = c.get("source", "?")
            page = c.get("page", None)
            if page is not None and page != -1:
                partes.append(f"{src}, p.{page}")
            else:
                partes.append(f"{src}")
        refs = "; ".join(partes)
        linhas.append(f"**Fontes (bula):** {refs}")

    return "\n\n".join(linhas)

# -------------------- API PRINCIPAL --------------------
def answer_question(query: str, lock_to_source: Optional[str] = None) -> Dict[str, Any]:
    docs_raw = recuperar_contexto(query, lock_to_source)
    docs = rerank_llm(query, docs_raw, top_k=8)
    docs = augment_with_headers(docs)
    if not docs:
        return {"ok": False, "msg": "Não encontrei trechos suficientes."}
    prod = extrair_produto(query, docs)
    prod = enrich_missing_fields(prod, docs)
    md = formatar_markdown(prod)
    lock_suggestion = prod.fontes[0]["source"] if prod.fontes else None
    return {"ok": True, "markdown": md, "produto": prod.dict(), "lock_suggestion": lock_suggestion}

def warmup():
    _load_components_cached()

# -------------------- UI (Streamlit) --------------------
st.title("🌿 Assistente de Campo Agrofel")
st.caption("RAG híbrido (FAISS+BM25+MMR+RRF) + extração estruturada com citações de bula.")

with st.sidebar:
    try:
        warmup()
        st.success("Índice carregado.")
        st.write(f"Dir índice:\n`{AGROFEL_INDEX_DIR}`")
    except Exception as e:
        st.error(f"Falha ao carregar índice: {e}")
        st.stop()

if "lock_source" not in st.session_state:
    st.session_state.lock_source = None

pergunta = st.text_area(
    "Qual praga está afetando sua lavoura e em qual cultura?",
    height=110,
    placeholder="Ex.: 'Produto para capim-amargoso (Digitaria insularis) em soja'"
)

c1, c2 = st.columns(2)
with c1:
    travar = st.checkbox(
        "Restringir à bula exibida anteriormente (travamento por produto)",
        value=bool(st.session_state.lock_source)
    )
with c2:
    st.text_input("Travado em:", value=st.session_state.lock_source or "", disabled=True)

if st.button("Buscar Sugestões"):
    if not pergunta.strip():
        st.warning("Digite a pergunta.")
    else:
        with st.spinner("Buscando nas bulas e extraindo informações..."):
            lock = st.session_state.lock_source if travar else None
            resp = answer_question(pergunta, lock_to_source=lock)

        if not resp.get("ok"):
            st.warning(resp.get("msg", "Falha ao responder."))
        else:
            st.subheader("Sugestão técnica")
            st.markdown(resp["markdown"])
            lock_sug = resp.get("lock_suggestion")
            st.markdown("---")
            if lock_sug and (not st.session_state.lock_source or st.session_state.lock_source != lock_sug):
                if st.button(f"🔒 Travar buscas nesta bula: {lock_sug}"):
                    st.session_state.lock_source = lock_sug
            elif not travar:
                st.session_state.lock_source = None




# # app.py
# import os
# import re
# from typing import List, Dict, Any, Optional

# import streamlit as st
# from dotenv import load_dotenv

# from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers import BM25Retriever
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema import Document
# from langchain_core.pydantic_v1 import BaseModel, Field

# import google.generativeai as genai

# # -------------------- CONFIG --------------------
# load_dotenv()
# st.set_page_config(page_title="Assistente de Campo Agrofel (v2)", page_icon="🌿", layout="wide")

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     st.error("GOOGLE_API_KEY não encontrado no .env/Secrets.")
#     st.stop()
# genai.configure(api_key=GOOGLE_API_KEY)

# # Ajuste aqui ou no .env (AGROFEL_INDEX_DIR). Pode ser caminho absoluto.
# AGROFEL_INDEX_DIR = os.getenv("AGROFEL_INDEX_DIR", "./index/faiss_index_agrofel")
# UNKNOWN = "Não especificado"

# # -------------------- SCHEMA --------------------
# class ProdutoExtraido(BaseModel):
#     nome_produto: str = Field(default=UNKNOWN)
#     culturas: List[str] = Field(default_factory=list)
#     pragas: List[str] = Field(default_factory=list)
#     ingrediente_ativo: str = Field(default=UNKNOWN)
#     dose: str = Field(default=UNKNOWN)
#     modo_aplicacao: str = Field(default=UNKNOWN)
#     intervalo_seguranca: str = Field(default=UNKNOWN)
#     modo_acao: str = Field(default=UNKNOWN)
#     efeito_lavoura: str = Field(default=UNKNOWN)
#     # Citações: [{"source":"arquivo.pdf","page":12}, ...]
#     fontes: List[Dict[str, Any]] = Field(default_factory=list)

# # -------------------- LOAD COMPONENTS --------------------
# @st.cache_resource(show_spinner="Carregando índice/LLM...")
# def _load_components_cached():
#     return load_components()

# def load_components():
#     if not os.path.exists(AGROFEL_INDEX_DIR):
#         raise FileNotFoundError(
#             f"Índice FAISS não encontrado em '{AGROFEL_INDEX_DIR}'. "
#             f"Rode 'python 1_Criar_Base_Vetorial.py' antes."
#         )

#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     db: FAISS = FAISS.load_local(AGROFEL_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

#     # Reconstroi docs para BM25 e mantém todos em memória
#     ids = list(db.index_to_docstore_id.values())
#     all_docs: List[Document] = []
#     for _id in ids:
#         d = db.docstore.search(_id)
#         if d and d.page_content:
#             all_docs.append(d)

#     bm25 = BM25Retriever.from_documents(all_docs) if all_docs else None
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1)

#     return db, bm25, llm, all_docs

# # -------------------- RAG: recuperação --------------------
# def gerar_consultas(query: str, llm) -> List[str]:
#     prompt = ChatPromptTemplate.from_template(
#         "Gere 4 variações curtas para busca em bulas (termos de cultura, praga e ingrediente ativo). "
#         "Uma por linha.\nPergunta: {p}\nConsultas:"
#     )
#     out = (prompt | llm | StrOutputParser()).invoke({"p": query}).strip().split("\n")
#     return [c.strip() for c in out if c.strip()][:4]

# def rrf_fusion(rankings: List[List[Document]], k: int = 12) -> List[Document]:
#     scores: Dict[str, float] = {}
#     items: Dict[str, Document] = {}
#     for lst in rankings:
#         for r, doc in enumerate(lst, start=1):
#             key = f"{doc.metadata.get('source','?')}|{doc.metadata.get('page',-1)}|{doc.metadata.get('chunk_id',-1)}"
#             scores[key] = scores.get(key, 0.0) + 1.0 / (60 + r)
#             items[key] = doc
#     ordered = sorted(items.keys(), key=lambda k: scores[k], reverse=True)
#     return [items[k] for k in ordered[:k]]

# def recuperar_contexto(query: str, lock_to_source: Optional[str] = None) -> List[Document]:
#     db, bm25, llm, _ = _load_components_cached()
#     queries = gerar_consultas(query, llm)

#     faiss_lists = []
#     for q in queries:
#         docs = db.max_marginal_relevance_search(q, k=8, fetch_k=40, lambda_mult=0.2)
#         if lock_to_source:
#             docs = [d for d in docs if d.metadata.get("source") == lock_to_source]
#         faiss_lists.append(docs)

#     bm25_lists = []
#     if bm25:
#         bm25.k = 6
#         for q in queries:
#             docs = bm25.get_relevant_documents(q)
#             if lock_to_source:
#                 docs = [d for d in docs if d.metadata.get("source") == lock_to_source]
#             bm25_lists.append(docs)

#     return rrf_fusion(faiss_lists + bm25_lists, k=14)

# def rerank_llm(query: str, docs: List[Document], top_k: int = 8) -> List[Document]:
#     if not docs:
#         return []
#     _, _, llm, _ = _load_components_cached()
#     snippet = "\n\n---\n\n".join(
#         [f"[{i}] (src={d.metadata.get('source')}, page={d.metadata.get('page',-1)})\n{d.page_content[:1600]}"
#          for i, d in enumerate(docs)]
#     )
#     prompt = f"""
# Pergunta: "{query}"

# Abaixo, trechos de bulas com índices. Selecione até {top_k} mais relevantes priorizando:
# Dose, Intervalo de segurança (ou "período de carência"), **Modo de aplicação / Forma de aplicação / Modo de uso /
# Aplicação terrestre/aérea/costal/tratorizado / Equipamento de aplicação**, Cultura, Praga e tabelas.
# Retorne SOMENTE JSON com índices. Ex: [0,2,5]

# Trechos:
# {snippet}
# """
#     try:
#         ids_json = llm.invoke(prompt).content.strip()
#         import json as _json
#         idxs = _json.loads(ids_json)
#         idxs = [i for i in idxs if isinstance(i, int) and 0 <= i < len(docs)]
#         return [docs[i] for i in idxs][:top_k]
#     except Exception:
#         return docs[:top_k]

# # -------------------- AUGMENT: headers + heurísticas --------------------
# def _dedup_docs(docs: List[Document]) -> List[Document]:
#     seen = set()
#     out: List[Document] = []
#     for d in docs:
#         key = (d.metadata.get("source"), d.metadata.get("page", -1), d.metadata.get("chunk_id", -1))
#         if key not in seen:
#             seen.add(key)
#             out.append(d)
#     return out

# def augment_with_headers(docs: List[Document]) -> List[Document]:
#     """Garante que o cabeçalho (primeiro chunk / primeira página) das bulas esteja no contexto."""
#     _, _, _, all_docs = _load_components_cached()
#     if not docs or not all_docs:
#         return docs
#     sources = {d.metadata.get("source") for d in docs if d.metadata.get("source")}
#     headers = []
#     for d in all_docs:
#         if d.metadata.get("source") in sources and (
#             d.metadata.get("chunk_id") == 1 or d.metadata.get("page", 999) in (0, 1)
#         ):
#             headers.append(d)
#     return _dedup_docs(headers + docs)

# def _guess_product_name_from_docs(docs: List[Document]) -> Optional[str]:
#     text = "\n".join([d.page_content for d in docs if d.page_content])
#     # 1) Campo explícito
#     m = re.search(r"(?:nome(?:\s+do)?\s+produto|nome\s+comercial|marca\s+comercial)\s*[:\-]\s*([^\n]{2,80})", text, flags=re.I)
#     if m:
#         return re.sub(r"\s+", " ", m.group(1)).strip()
#     # 2) Linha título em destaque
#     for line in text.splitlines():
#         s = line.strip()
#         if 3 <= len(s) <= 60 and (s.isupper() or s.istitle()) and not s.endswith(":"):
#             return s
#     # 3) Fallback: nome do arquivo
#     src = docs[0].metadata.get("source") if docs else None
#     if src:
#         base = re.sub(r"\.(pdf|docx)$", "", src, flags=re.I)
#         base = re.sub(r"[_\-]+", " ", base).strip()
#         return base
#     return None

# def _extract_intervalo(text: str) -> Optional[str]:
#     m = re.search(r"(?:intervalo\s+de\s+segurança|período\s+de\s+carência)\s*[:\-]?\s*([^\n]{1,80})", text, flags=re.I)
#     return m.group(1).strip() if m else None

# def _extract_dose(text: str) -> Optional[str]:
#     # Exemplos: 0,5–0,7 L/ha | 100-300 L/ha | 60 g i.a./ha | 1 L/ha
#     m = re.search(r"(\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?\s*(?:mL|L|g|kg)\s*(?:i\.a\.\s*)?/ha)",
#                   text, flags=re.I)
#     return m.group(1).replace(",", ".").strip() if m else None

# def _extract_modo_aplicacao(text: str) -> Optional[str]:
#     """
#     Extrai um resumo curto do modo de aplicação (terrestre/aérea/costal/tratorizado, via foliar, pré/pós-emergência)
#     e volume de calda, a partir de cabeçalhos e palavras-chave.
#     """
#     import re
#     t = text
#     tl = t.lower()

#     parts = []
#     header_re = re.compile(
#         r"(?:modo\s+de\s+aplica[cç][aã]o|forma\s+de\s+aplica[cç][aã]o|modo\s+de\s+uso|"
#         r"equipamento\s+de\s+aplica[cç][aã]o)\s*[:\-]?\s*([^\n]{1,160})",
#         flags=re.I
#     )
#     for m in header_re.finditer(t):
#         frag = re.sub(r"\s+", " ", m.group(1)).strip(" .;")
#         if frag:
#             parts.append(frag)

#     flags = []
#     if re.search(r"aplica[cç][aã]o\s+terrestre", tl): flags.append("Terrestre")
#     if re.search(r"aplica[cç][aã]o\s+a[ée]rea", tl): flags.append("Aérea")
#     if "costal" in tl: flags.append("Costal")
#     if re.search(r"tratorizad[oa]", tl): flags.append("Tratorizado")
#     if re.search(r"turbo\s*atomizador|atomizador", tl): flags.append("Turbo atomizador")
#     if re.search(r"via\s+foliar", tl): flags.append("Via foliar")
#     if re.search(r"p[óo]s-?\s*emerg[êe]ncia", tl): flags.append("Pós-emergência")
#     if re.search(r"pr[ée]-?\s*emerg[êe]ncia", tl): flags.append("Pré-emergência")

#     vol = re.search(r"volume\s+de\s+calda\s*[:\-]?\s*([^\n]{1,80})", t, flags=re.I)
#     vol_txt = vol.group(1).strip() if vol else None

#     chunks = []
#     if flags:
#         seen = set(); ordered = []
#         for f in flags:
#             if f not in seen:
#                 seen.add(f); ordered.append(f)
#         chunks.append(", ".join(ordered))
#     if vol_txt:
#         chunks.append(f"Volume de calda {vol_txt}")
#     if parts:
#         seen2 = set(); dedup = []
#         for p in parts:
#             p2 = p.lower()
#             if p2 not in seen2:
#                 seen2.add(p2); dedup.append(p)
#         extra = "; ".join(dedup)
#         if extra:
#             chunks.append(extra)

#     out = " | ".join(chunks).strip()
#     return out or None

# def enrich_missing_fields(prod: ProdutoExtraido, docs: List[Document]) -> ProdutoExtraido:
#     text = "\n".join([d.page_content for d in docs if d.page_content])
#     if not prod.nome_produto or prod.nome_produto == UNKNOWN:
#         hint = _guess_product_name_from_docs(docs)
#         if hint:
#             prod.nome_produto = hint
#     if not prod.intervalo_seguranca or prod.intervalo_seguranca == UNKNOWN:
#         iv = _extract_intervalo(text)
#         if iv:
#             prod.intervalo_seguranca = iv
#     if not prod.dose or prod.dose == UNKNOWN:
#         ds = _extract_dose(text)
#         if ds:
#             prod.dose = ds
#     if not prod.modo_aplicacao or prod.modo_aplicacao == UNKNOWN:
#         ma = _extract_modo_aplicacao(text)
#         if ma:
#             prod.modo_aplicacao = ma
#     return prod

# # -------------------- EXTRAÇÃO E FORMATAÇÃO --------------------
# def build_citations(docs: List[Document]) -> List[Dict[str, Any]]:
#     cites = [{"source": d.metadata.get("source"), "page": d.metadata.get("page", -1)} for d in docs]
#     seen = set(); uniq = []
#     for c in cites:
#         key = (c["source"], c["page"])
#         if key not in seen:
#             seen.add(key); uniq.append(c)
#     return uniq

# def extrair_produto(query: str, docs: List[Document]) -> ProdutoExtraido:
#     _, _, llm, _ = _load_components_cached()
#     context = "\n\n---\n\n".join([d.page_content for d in docs])
#     fontes = build_citations(docs)

#     prompt = f"""
# Você é agrônomo. Extraia APENAS o objeto (schema ProdutoExtraido) com base nos TRECHOS.
# - Priorize: Nome do produto (nome comercial), Cultura, Praga, Dose, Intervalo de segurança (ou "período de carência"),
#   Modo de aplicação, Modo de ação e Ingrediente ativo.
# - Se o nome comercial não aparecer claramente, utilize o melhor indício (ex.: linha de título ou nome do arquivo sem extensão).
# - Não invente; quando realmente não houver, use "{UNKNOWN}".
# - Se houver múltiplos produtos, escolha o mais pertinente à pergunta.

# PERGUNTA: "{query}"

# TRECHOS:
# {context}
# """
#     prod: ProdutoExtraido = llm.with_structured_output(ProdutoExtraido).invoke(prompt)
#     prod.fontes = fontes
#     # normalização leve
#     prod.dose = prod.dose.replace(" – ", "-").replace(" a ", " - ").strip()
#     return prod

# def formatar_markdown(prod: ProdutoExtraido) -> str:
#     linhas = [
#         f"**Nome do Produto:** {prod.nome_produto or UNKNOWN}",
#         f"**Cultura(s):** {', '.join(prod.culturas) if prod.culturas else UNKNOWN}",
#         f"**Praga(s):** {', '.join(prod.pragas) if prod.pragas else UNKNOWN}",
#         f"**Ingrediente Ativo:** {prod.ingrediente_ativo or UNKNOWN}",
#         f"**Dose:** {prod.dose or UNKNOWN}",
#         f"**Modo de Aplicação:** {prod.modo_aplicacao or UNKNOWN}",
#         f"**Intervalo de Segurança:** {prod.intervalo_seguranca or UNKNOWN}",
#         f"**Modo de Ação:** {prod.modo_acao or UNKNOWN}",
#         f"**Efeito na Lavoura:** {prod.efeito_lavoura or UNKNOWN}",
#     ]
#     if getattr(prod, "fontes", None):
#         partes = []
#         for c in prod.fontes[:6]:
#             src = c.get("source", "?")
#             page = c.get("page", None)
#             if page is not None and page != -1:
#                 partes.append(f"{src}, p.{page}")
#             else:
#                 partes.append(f"{src}")
#         refs = "; ".join(partes)
#         linhas.append(f"**Fontes (bula):** {refs}")
#     return "\n".join(linhas)

# # -------------------- API PRINCIPAL --------------------
# def answer_question(query: str, lock_to_source: Optional[str] = None) -> Dict[str, Any]:
#     docs_raw = recuperar_contexto(query, lock_to_source)
#     docs = rerank_llm(query, docs_raw, top_k=8)
#     docs = augment_with_headers(docs)  # garante cabeçalho com nome comercial
#     if not docs:
#         return {"ok": False, "msg": "Não encontrei trechos suficientes."}
#     prod = extrair_produto(query, docs)
#     prod = enrich_missing_fields(prod, docs)  # heurísticas de complemento
#     md = formatar_markdown(prod)
#     lock_suggestion = prod.fontes[0]["source"] if prod.fontes else None
#     return {"ok": True, "markdown": md, "produto": prod.dict(), "lock_suggestion": lock_suggestion}

# def warmup():
#     _load_components_cached()

# # -------------------- UI (Streamlit) --------------------
# st.title("🌿 Assistente de Campo Agrofel (v2)")
# st.caption("RAG híbrido (FAISS+BM25+MMR+RRF) + extração estruturada com citações de bula.")

# # status
# with st.sidebar:
#     try:
#         warmup()
#         st.success("Índice carregado.")
#         st.write(f"Dir índice:\n`{AGROFEL_INDEX_DIR}`")
#     except Exception as e:
#         st.error(f"Falha ao carregar índice: {e}")
#         st.stop()

# if "lock_source" not in st.session_state:
#     st.session_state.lock_source = None

# pergunta = st.text_area(
#     "Qual praga está afetando sua lavoura e em qual cultura?",
#     height=110,
#     placeholder="Ex.: 'Produto para capim-amargoso (Digitaria insularis) em soja'"
# )

# c1, c2 = st.columns(2)
# with c1:
#     travar = st.checkbox("Restringir à bula exibida anteriormente (travamento por produto)",
#                          value=bool(st.session_state.lock_source))
# with c2:
#     st.text_input("Travado em:", value=st.session_state.lock_source or "", disabled=True)

# if st.button("Buscar Sugestões"):
#     if not pergunta.strip():
#         st.warning("Digite a pergunta.")
#     else:
#         with st.spinner("Buscando nas bulas e extraindo informações..."):
#             lock = st.session_state.lock_source if travar else None
#             resp = answer_question(pergunta, lock_to_source=lock)

#         if not resp.get("ok"):
#             st.warning(resp.get("msg", "Falha ao responder."))
#         else:
#             st.subheader("Sugestão técnica")
#             st.markdown(resp["markdown"])
#             lock_sug = resp.get("lock_suggestion")
#             st.markdown("---")
#             if lock_sug and (not st.session_state.lock_source or st.session_state.lock_source != lock_sug):
#                 if st.button(f"🔒 Travar buscas nesta bula: {lock_sug}"):
#                     st.session_state.lock_source = lock_sug
#             elif not travar:
#                 st.session_state.lock_source = None



# # app.py
# import os
# import re
# from typing import List, Dict, Any, Optional, Tuple

# import streamlit as st
# from dotenv import load_dotenv

# from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers import BM25Retriever
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema import Document
# from langchain_core.pydantic_v1 import BaseModel, Field

# import google.generativeai as genai

# # -------------------- CONFIG --------------------
# load_dotenv()
# st.set_page_config(page_title="Assistente de Campo Agrofel (v2)", page_icon="🌿", layout="wide")

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     st.error("GOOGLE_API_KEY não encontrado no .env/Secrets.")
#     st.stop()
# genai.configure(api_key=GOOGLE_API_KEY)

# AGROFEL_INDEX_DIR = os.getenv("AGROFEL_INDEX_DIR", "./index/faiss_index_agrofel")
# UNKNOWN = "Não especificado"

# # -------------------- SCHEMA --------------------
# class ProdutoExtraido(BaseModel):
#     nome_produto: str = Field(default=UNKNOWN)
#     culturas: List[str] = Field(default_factory=list)
#     pragas: List[str] = Field(default_factory=list)
#     ingrediente_ativo: str = Field(default=UNKNOWN)
#     dose: str = Field(default=UNKNOWN)
#     modo_aplicacao: str = Field(default=UNKNOWN)
#     intervalo_seguranca: str = Field(default=UNKNOWN)
#     modo_acao: str = Field(default=UNKNOWN)
#     efeito_lavoura: str = Field(default=UNKNOWN)
#     fontes: List[Dict[str, Any]] = Field(default_factory=list)  # [{"source": "...", "page": 1}, ...]

# # -------------------- LOAD COMPONENTS --------------------
# _db = None
# _bm25 = None
# _llm = None
# _all_docs = None  # manter todos os docs na memória para pegar headers facilmente

# @st.cache_resource(show_spinner="Carregando índice/LLM...")
# def _load_components_cached():
#     return load_components()

# def load_components():
#     global _db, _bm25, _llm, _all_docs

#     if not os.path.exists(AGROFEL_INDEX_DIR):
#         raise FileNotFoundError(
#             f"Índice FAISS não encontrado em '{AGROFEL_INDEX_DIR}'. "
#             f"Rode 'python 1_Criar_Base_Vetorial.py' antes."
#         )
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     db: FAISS = FAISS.load_local(AGROFEL_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

#     # Reconstrói docs p/ BM25 e mantém todos em memória
#     ids = list(db.index_to_docstore_id.values())
#     all_docs: List[Document] = []
#     for _id in ids:
#         d = db.docstore.search(_id)
#         if d and d.page_content:
#             all_docs.append(d)

#     bm25 = BM25Retriever.from_documents(all_docs) if all_docs else None
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1)

#     _db, _bm25, _llm, _all_docs = db, bm25, llm, all_docs
#     return _db, _bm25, _llm, _all_docs

# # -------------------- RAG: recuperação --------------------
# def gerar_consultas(query: str, llm) -> List[str]:
#     prompt = ChatPromptTemplate.from_template(
#         "Gere 4 variações curtas para busca em bulas (termos de cultura, praga e ingrediente ativo). "
#         "Uma por linha.\nPergunta: {p}\nConsultas:"
#     )
#     out = (prompt | llm | StrOutputParser()).invoke({"p": query}).strip().split("\n")
#     return [c.strip() for c in out if c.strip()][:4]

# def rrf_fusion(rankings: List[List[Document]], k: int = 12) -> List[Document]:
#     scores: Dict[str, float] = {}
#     items: Dict[str, Document] = {}
#     for lst in rankings:
#         for r, doc in enumerate(lst, start=1):
#             key = f"{doc.metadata.get('source','?')}|{doc.metadata.get('page',-1)}|{doc.metadata.get('chunk_id',-1)}"
#             scores[key] = scores.get(key, 0.0) + 1.0 / (60 + r)
#             items[key] = doc
#     ordered = sorted(items.keys(), key=lambda k: scores[k], reverse=True)
#     return [items[k] for k in ordered[:k]]

# def recuperar_contexto(query: str, lock_to_source: Optional[str] = None) -> List[Document]:
#     db, bm25, llm, _ = _load_components_cached()
#     queries = gerar_consultas(query, llm)

#     faiss_lists = []
#     for q in queries:
#         docs = db.max_marginal_relevance_search(q, k=8, fetch_k=40, lambda_mult=0.2)
#         if lock_to_source:
#             docs = [d for d in docs if d.metadata.get("source") == lock_to_source]
#         faiss_lists.append(docs)

#     bm25_lists = []
#     if bm25:
#         bm25.k = 6
#         for q in queries:
#             docs = bm25.get_relevant_documents(q)
#             if lock_to_source:
#                 docs = [d for d in docs if d.metadata.get("source") == lock_to_source]
#             bm25_lists.append(docs)

#     return rrf_fusion(faiss_lists + bm25_lists, k=14)

# def rerank_llm(query: str, docs: List[Document], top_k: int = 8) -> List[Document]:
#     if not docs:
#         return []
#     _, _, llm, _ = _load_components_cached()
#     snippet = "\n\n---\n\n".join(
#         [f"[{i}] (src={d.metadata.get('source')}, page={d.metadata.get('page',-1)})\n{d.page_content[:1600]}"
#          for i, d in enumerate(docs)]
#     )
#     prompt = f"""
# Pergunta: "{query}"

# Abaixo, trechos de bulas com índices. Selecione até {top_k} mais relevantes priorizando:
# Dose, Intervalo de segurança (ou 'período de carência'), Cultura, Praga, Modo de ação/aplicação e tabelas.
# Retorne SOMENTE JSON com índices. Ex: [0,2,5]

# Trechos:
# {snippet}
# """
#     try:
#         ids_json = llm.invoke(prompt).content.strip()
#         import json as _json
#         idxs = _json.loads(ids_json)
#         idxs = [i for i in idxs if isinstance(i, int) and 0 <= i < len(docs)]
#         return [docs[i] for i in idxs][:top_k]
#     except Exception:
#         return docs[:top_k]

# # -------------------- AUGMENT: headers + heurísticas --------------------
# def _dedup_docs(docs: List[Document]) -> List[Document]:
#     seen = set()
#     out: List[Document] = []
#     for d in docs:
#         key = (d.metadata.get("source"), d.metadata.get("page", -1), d.metadata.get("chunk_id", -1))
#         if key not in seen:
#             seen.add(key)
#             out.append(d)
#     return out

# def augment_with_headers(docs: List[Document]) -> List[Document]:
#     """Garante que o cabeçalho (primeiro chunk / primeira página) das bulas esteja no contexto."""
#     _, _, _, all_docs = _load_components_cached()
#     if not docs or not all_docs:
#         return docs
#     sources = {d.metadata.get("source") for d in docs if d.metadata.get("source")}
#     headers = []
#     for d in all_docs:
#         if d.metadata.get("source") in sources and (
#             d.metadata.get("chunk_id") == 1 or d.metadata.get("page", 999) in (0, 1)
#         ):
#             headers.append(d)
#     return _dedup_docs(headers + docs)

# def _guess_product_name_from_docs(docs: List[Document]) -> Optional[str]:
#     text = "\n".join([d.page_content for d in docs if d.page_content])
#     # 1) Campo explícito
#     m = re.search(r"(?:nome(?:\s+do)?\s+produto|nome\s+comercial|marca\s+comercial)\s*[:\-]\s*([^\n]{2,80})", text, flags=re.I)
#     if m:
#         return re.sub(r"\s+", " ", m.group(1)).strip()
#     # 2) Linha título em destaque
#     for line in text.splitlines():
#         s = line.strip()
#         if 3 <= len(s) <= 60 and (s.isupper() or s.istitle()) and not s.endswith(":"):
#             return s
#     # 3) Fallback: nome do arquivo
#     src = docs[0].metadata.get("source") if docs else None
#     if src:
#         base = re.sub(r"\.(pdf|docx)$", "", src, flags=re.I)
#         base = re.sub(r"[_\-]+", " ", base).strip()
#         return base
#     return None

# def _extract_intervalo(text: str) -> Optional[str]:
#     m = re.search(r"(?:intervalo\s+de\s+segurança|período\s+de\s+carência)\s*[:\-]?\s*([^\n]{1,80})", text, flags=re.I)
#     return m.group(1).strip() if m else None

# def _extract_dose(text: str) -> Optional[str]:
#     # Exemplos: 0,5–0,7 L/ha | 100-300 L/ha | 60 g i.a./ha | 1 L/ha
#     m = re.search(r"(\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?\s*(?:mL|L|g|kg)\s*(?:i\.a\.\s*)?/ha)",
#                   text, flags=re.I)
#     return m.group(1).replace(",", ".").strip() if m else None

# def enrich_missing_fields(prod: ProdutoExtraido, docs: List[Document]) -> ProdutoExtraido:
#     text = "\n".join([d.page_content for d in docs if d.page_content])
#     if not prod.nome_produto or prod.nome_produto == UNKNOWN:
#         hint = _guess_product_name_from_docs(docs)
#         if hint:
#             prod.nome_produto = hint
#     if not prod.intervalo_seguranca or prod.intervalo_seguranca == UNKNOWN:
#         iv = _extract_intervalo(text)
#         if iv:
#             prod.intervalo_seguranca = iv
#     if not prod.dose or prod.dose == UNKNOWN:
#         ds = _extract_dose(text)
#         if ds:
#             prod.dose = ds
#     return prod

# # -------------------- EXTRAÇÃO E FORMATAÇÃO --------------------
# def build_citations(docs: List[Document]) -> List[Dict[str, Any]]:
#     cites = [{"source": d.metadata.get("source"), "page": d.metadata.get("page", -1)} for d in docs]
#     seen = set(); uniq = []
#     for c in cites:
#         key = (c["source"], c["page"])
#         if key not in seen:
#             seen.add(key); uniq.append(c)
#     return uniq

# def extrair_produto(query: str, docs: List[Document]) -> ProdutoExtraido:
#     _, _, llm, _ = _load_components_cached()
#     context = "\n\n---\n\n".join([d.page_content for d in docs])
#     fontes = build_citations(docs)

#     prompt = f"""
# Você é agrônomo. Extraia APENAS o objeto (schema ProdutoExtraido) com base nos TRECHOS.
# - Priorize: Nome do produto (nome comercial), Cultura, Praga, Dose, Intervalo de segurança (ou "período de carência"),
#   Modo de aplicação, Modo de ação e Ingrediente ativo.
# - Se o nome comercial não aparecer claramente, utilize o melhor indício (ex.: linha de título ou nome do arquivo sem extensão).
# - Não invente; quando realmente não houver, use "{UNKNOWN}".
# - Se houver múltiplos produtos, escolha o mais pertinente à pergunta.

# PERGUNTA: "{query}"

# TRECHOS:
# {context}
# """
#     prod: ProdutoExtraido = llm.with_structured_output(ProdutoExtraido).invoke(prompt)
#     prod.fontes = fontes
#     # normalização leve
#     prod.dose = prod.dose.replace(" – ", "-").replace(" a ", " - ").strip()
#     return prod

# def formatar_markdown(prod: ProdutoExtraido) -> str:
#     linhas = [
#         f"**Nome do Produto:** {prod.nome_produto or UNKNOWN}",
#         f"**Cultura(s):** {', '.join(prod.culturas) if prod.culturas else UNKNOWN}",
#         f"**Praga(s):** {', '.join(prod.pragas) if prod.pragas else UNKNOWN}",
#         f"**Ingrediente Ativo:** {prod.ingrediente_ativo or UNKNOWN}",
#         f"**Dose:** {prod.dose or UNKNOWN}",
#         f"**Modo de Aplicação:** {prod.modo_aplicacao or UNKNOWN}",
#         f"**Intervalo de Segurança:** {prod.intervalo_seguranca or UNKNOWN}",
#         f"**Modo de Ação:** {prod.modo_acao or UNKNOWN}",
#         f"**Efeito na Lavoura:** {prod.efeito_lavoura or UNKNOWN}",
#     ]

#     if getattr(prod, "fontes", None):
#         partes = []
#         for c in prod.fontes[:6]:
#             src = c.get("source", "?")
#             page = c.get("page", None)
#             if page is not None and page != -1:
#                 partes.append(f"{src}, p.{page}")
#             else:
#                 partes.append(f"{src}")
#         refs = "; ".join(partes)
#         linhas.append(f"**Fontes (bula):** {refs}")

#     return "\n".join(linhas)

# # -------------------- API PRINCIPAL --------------------
# def answer_question(query: str, lock_to_source: Optional[str] = None) -> Dict[str, Any]:
#     docs_raw = recuperar_contexto(query, lock_to_source)
#     docs = rerank_llm(query, docs_raw, top_k=8)
#     docs = augment_with_headers(docs)  # garante cabeçalho com nome comercial
#     if not docs:
#         return {"ok": False, "msg": "Não encontrei trechos suficientes."}
#     prod = extrair_produto(query, docs)
#     prod = enrich_missing_fields(prod, docs)  # heurísticas de complemento
#     md = formatar_markdown(prod)
#     lock_suggestion = prod.fontes[0]["source"] if prod.fontes else None
#     return {"ok": True, "markdown": md, "produto": prod.dict(), "lock_suggestion": lock_suggestion}

# def warmup():
#     _load_components_cached()

# # -------------------- UI (Streamlit) --------------------
# st.title("🌿 Assistente de Campo Agrofel (v2)")
# st.caption("RAG híbrido (FAISS+BM25+MMR+RRF) + extração estruturada com citações de bula.")

# # status
# with st.sidebar:
#     try:
#         warmup()
#         st.success("Índice carregado.")
#         st.write(f"Dir índice: `{AGROFEL_INDEX_DIR}`")
#     except Exception as e:
#         st.error(f"Falha ao carregar índice: {e}")
#         st.stop()

# if "lock_source" not in st.session_state:
#     st.session_state.lock_source = None

# pergunta = st.text_area(
#     "Qual praga está afetando sua lavoura e em qual cultura?",
#     height=110,
#     placeholder="Ex.: 'Produto para capim-amargoso (Digitaria insularis) em soja'"
# )

# c1, c2 = st.columns(2)
# with c1:
#     travar = st.checkbox("Restringir à bula exibida anteriormente (travamento por produto)",
#                          value=bool(st.session_state.lock_source))
# with c2:
#     st.text_input("Travado em:", value=st.session_state.lock_source or "", disabled=True)

# if st.button("Buscar Sugestões"):
#     if not pergunta.strip():
#         st.warning("Digite a pergunta.")
#     else:
#         with st.spinner("Buscando nas bulas e extraindo informações..."):
#             lock = st.session_state.lock_source if travar else None
#             resp = answer_question(pergunta, lock_to_source=lock)

#         if not resp.get("ok"):
#             st.warning(resp.get("msg", "Falha ao responder."))
#         else:
#             st.subheader("Sugestão técnica")
#             st.markdown(resp["markdown"])
#             lock_sug = resp.get("lock_suggestion")
#             st.markdown("---")
#             if lock_sug and (not st.session_state.lock_source or st.session_state.lock_source != lock_sug):
#                 if st.button(f"🔒 Travar buscas nesta bula: {lock_sug}"):
#                     st.session_state.lock_source = lock_sug
#             elif not travar:
#                 st.session_state.lock_source = None















# # # app.py (Versão com Melhor Extração de Nomes de Produtos)
# # import streamlit as st
# # import os
# # import smtplib
# # from email.message import EmailMessage
# # from urllib.parse import quote

# # import google.generativeai as genai
# # from dotenv import load_dotenv

# # # Imports para LangChain e Pydantic
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# # from langchain_community.vectorstores import FAISS
# # from langchain.prompts import ChatPromptTemplate
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_core.pydantic_v1 import BaseModel, Field

# # # --- CONFIGURAÇÃO INICIAL DA PÁGINA E VARIÁVEIS DE AMBIENTE ---
# # st.set_page_config(page_title="Assistente Agrofel", page_icon="🌿", layout="wide")

# # # Lógica para carregar a chave de API de forma segura
# # load_dotenv()
# # api_key = os.getenv("GOOGLE_API_KEY")
# # if not api_key:
# #     try:
# #         api_key = st.secrets["GOOGLE_API_KEY"]
# #     except (FileNotFoundError, KeyError):
# #         st.error("Chave de API do Google não encontrada! Configure-a no arquivo .env ou nos segredos do Streamlit.")
# #         st.stop()
# # genai.configure(api_key=api_key)


# # # --- DEFINIÇÃO DA FERRAMENTA PARA ANÁLISE SEMÂNTICA (REVISTA) ---
# # class AnalisePergunta(BaseModel):
# #     """Schema para analisar a pergunta de um agricultor."""
# #     cultura: str | None = Field(description="A cultura agrícola mencionada, como 'soja' ou 'milho'. Inclui sinônimos como 'plantação', 'lavoura'. Null se não mencionada.")
# #     praga: str | None = Field(description="A praga mencionada, como 'lagarta', 'guanxuma' ou 'ferrugem'. Inclui termos relacionados como 'doença', 'inseto', 'patógeno'. Null se não mencionada.")
# #     termo_produto: bool = Field(description="True se a pergunta mencionar termos como 'produto', 'veneno', 'tratamento' ou 'solução'.")


# # # --- FUNÇÕES DE LÓGICA (BACKEND DA APLICAÇÃO) ---

# # @st.cache_resource(show_spinner="A carregar base de conhecimento...")
# # def carregar_base_conhecimento():
# #     """ Carrega o índice FAISS pré-construído do repositório. """
# #     CAMINHO_INDEX_FAISS = "faiss_index_agrofel"
# #     if not os.path.exists(CAMINHO_INDEX_FAISS):
# #         st.error(f"ERRO CRÍTICO: A base de conhecimento pré-construída ('{CAMINHO_INDEX_FAISS}') não foi encontrada.")
# #         return None, None
# #     try:
# #         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# #         db = FAISS.load_local(CAMINHO_INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
# #         llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1) # Temperatura baixa para precisão
# #         return db, llm
# #     except Exception as e:
# #         st.error(f"Ocorreu um erro ao carregar a base de conhecimento: {e}")
# #         return None, None

# # def _buscar_e_gerar_recomendacao(query: str, db, llm, aviso_contexto: str | None = None):
# #     """ Executa o fluxo de RAG com Transformação de Consulta. """
# #     template_transformacao = "Você é um especialista em agronomia. Transforme a pergunta do utilizador em 3 consultas de busca concisas e variadas para uma base de dados vetorial de bulas de produtos. Responda apenas com as consultas, uma por linha.\n\nPergunta Original: {pergunta}\n\nConsultas de Busca:"
# #     prompt_transformacao = ChatPromptTemplate.from_template(template_transformacao)
# #     cadeia_transformacao = prompt_transformacao | llm | StrOutputParser()
# #     consultas_geradas = cadeia_transformacao.invoke({"pergunta": query}).strip().split('\n')
    
# #     todos_chunks = []
# #     for consulta in consultas_geradas:
# #         # Prioriza buscas que contenham termos relacionados a nomes de produtos
# #         if "nome" in consulta.lower() or "produto" in consulta.lower():
# #             todos_chunks.extend(db.similarity_search(consulta, k=6))  # Mais resultados para buscas por nomes
# #         else:
# #             todos_chunks.extend(db.similarity_search(consulta, k=3))
    
# #     unique_chunks = {doc.page_content: doc for doc in todos_chunks}.values()
# #     if not unique_chunks: return "NAO_ENCONTRADO"

# #     contexto_final = "\n\n---\n\n".join([doc.page_content for doc in unique_chunks])
    
# #     aviso_prompt = f"{aviso_contexto}\n\n" if aviso_contexto else ""

# #     # ★★★ INSTRUÇÕES REVISADAS PARA MELHOR EXTRAÇÃO DE NOMES ★★★
# #     prompt_geracao_final = f"""{aviso_prompt}Você é um consultor especialista da Agrofel. Sua tarefa é extrair informações técnicas detalhadas sobre produtos agrícolas a partir dos TRECHOS RELEVANTES DAS BULAS fornecidos.

# # PERGUNTA ORIGINAL: "{query}"

# # TRECHOS RELEVANTES:
# # ---
# # {contexto_final}
# # ---

# # INSTRUÇÕES CRUCIAIS:
# # 1. **NOME DO PRODUTO É OBRIGATÓRIO**: Sempre extraia o nome comercial completo do produto. 
# #    - Procure por: marcas registradas, nomes entre aspas, ou frases como "produto XYZ"
# #    - Se não encontrar explicitamente, INFIRA usando ingredientes ativos + fabricante (ex: "Glifosato 480 - Syngenta")
# #    - Só use "Não especificado" se for IMPOSSÍVEL identificar
   
# # 2. Para cada produto relevante, extraia TODOS os campos abaixo

# # 3. Formato estrito de saída (um campo por linha sempre):
# #    **Nome do Produto:** [Nome comercial completo. Ex: "Herbex 480 EC"]
# #    **Cultura(s):** [Culturas indicadas]
# #    **Praga(s):** [Pragas controladas]
# #    **Ingrediente Ativo:** [Princípio ativo + concentração]
# #    **Dose:** [Faixa de dosagem]
# #    **Modo de Aplicação:** [Instruções de aplicação]
# #    **Intervalo de Segurança:** [Período de carência]
# #    **Modo de Ação:** [Mecanismo de ação]
# #    **Efeito na Lavoura:** [Benefícios principais]

# # 4. SEPARE múltiplos produtos com "---" (3 hífens)

# # 5. Se nenhum produto for adequado: "NAO_ENCONTRADO"

# # 6. NUNCA invente nomes ou informações!
# # """

# #     resposta_final = llm.invoke(prompt_geracao_final).content

# #     # ★★★ PÓS-PROCESSAMENTO PARA VERIFICAR NOMES FALTANTES ★★★
# #     if "Não especificado" in resposta_final and "Nome do Produto" in resposta_final:
# #         st.warning("Refinando extração de nomes...")
# #         prompt_refinamento = f"""
# #         Você encontrou produtos relevantes mas faltam nomes comerciais. 
# #         ANALISE NOVAMENTE os trechos e INFIRA nomes com base em:
# #         - Ingredientes ativos mencionados
# #         - Fabricantes/selos ("Marca ABC")
# #         - Nomes entre aspas
# #         - Contexto de frases como "o produto XYZ"
        
# #         Trechos Originais:
# #         {contexto_final}
        
# #         Resposta Anterior (para corrigir):
# #         {resposta_final}
        
# #         INSTRUÇÃO: Mantenha o mesmo formato mas PREENCHA TODOS os nomes faltantes!
# #         """
# #         resposta_final = llm.invoke(prompt_refinamento).content

# #     return resposta_final

# # def obter_resposta_assistente(query: str, db, llm):
# #     """
# #     Orquestra o fluxo de trabalho com Roteador Melhorado
# #     """
# #     # ETAPA 1: Análise Semântica
# #     llm_com_ferramenta = llm.bind_tools([AnalisePergunta])
# #     analise = llm_com_ferramenta.invoke(query)
    
# #     # Fallback direto se análise falhar
# #     if not analise.tool_calls:
# #         return _buscar_e_gerar_recomendacao(
# #             query, db, llm, 
# #             aviso_contexto="🔍 Buscando soluções para sua consulta..."
# #         )

# #     dados_analise = analise.tool_calls[0]['args']
# #     cultura = dados_analise.get("cultura")
# #     praga = dados_analise.get("praga")
# #     termo_produto = dados_analise.get("termo_produto", False)

# #     # ETAPA 2: Roteamento Inteligente (REVISTO)
# #     aviso = None
    
# #     # Caso 1: Pergunta muito vazia
# #     if not praga and not cultura and not termo_produto:
# #         return "Por favor, seja mais específico na sua pergunta. Informe a cultura e/ou a praga que deseja controlar. Ex: 'Preciso de produto para lagarta na soja'."
    
# #     # Caso 2: Menções parciais com termo de produto
# #     if termo_produto:
# #         if praga and not cultura:
# #             aviso = f"🔍 Buscando produtos para '{praga}'. Verifique o registro para sua cultura."
# #         elif cultura and not praga:
# #             aviso = f"🔍 Buscando produtos recomendados para '{cultura}'. Especifique a praga para recomendações mais precisas."
# #         elif not praga and not cultura:
# #             aviso = "🔍 Buscando produtos agrícolas gerais. Especifique cultura/praga para recomendações direcionadas."
    
# #     # Caso 3: Temos pelo menos um elemento relevante
# #     return _buscar_e_gerar_recomendacao(
# #         query, db, llm, 
# #         aviso_contexto=aviso
# #     )

# # # --- Funções de Notificação (mantidas) ---
# # def enviar_email_confirmacao(pergunta, recomendacao):
# #     try:
# #         email_vendedor, email_remetente, senha_remetente = st.secrets["EMAIL_VENDEDOR"], st.secrets["EMAIL_REMETENTE"], st.secrets["SENHA_REMETENTE"]
# #         corpo_email = f'<html><body><p>Olá,</p><p>Um cliente solicitou um pedido através do <b>Assistente de Campo Agrofel</b>.</p><hr><h3>Detalhes da Solicitação:</h3><p><b>Pergunta do Cliente:</b><br>{pergunta}</p><p><b>Produtos Sugeridos e Confirmados:</b></p><div style="background-color:#f0f0f0; border-left: 5px solid #4CAF50; padding: 10px; font-family: monospace; white-space: pre-wrap;">{recomendacao.replace(chr(10), "<br>")}</div><br><p>Por favor, entre em contato com o cliente para dar seguimento.</p><p>Atenciosamente,<br>Assistente de Campo Agrofel</p></body></html>'
# #         msg = EmailMessage()
# #         msg['Subject'], msg['From'], msg['To'] = "Novo Pedido de Cliente - Assistente de Campo Agrofel", email_remetente, email_vendedor
# #         msg.add_alternative(corpo_email, subtype='html')
# #         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
# #             smtp.login(email_remetente, senha_remetente)
# #             smtp.send_message(msg)
# #         st.success("Pedido confirmado! Seu consultor Agrofel foi notificado por email.")
# #         st.balloons()
# #     except Exception as e:
# #         st.error(f"Ocorreu um erro ao enviar o email: {e}")

# # def gerar_link_whatsapp(pergunta, recomendacao):
# #     numero_whatsapp = "5519989963385" # Substitua pelo número real
# #     texto_base = f"""Olá! Usei o Assistente de Campo Agrofel e gostaria de falar com um especialista.\n\n*Minha pergunta foi:*\n"{pergunta}"\n\n*A recomendação foi:*\n{recomendacao}\n\nAguardo contato."""
# #     texto_formatado = quote(texto_base)
# #     return f"https://wa.me/{numero_whatsapp}?text={texto_formatado}"

# # # --- INTERFACE DO USUÁRIO ---
# # st.title("🌿 Assistente de Campo Agrofel")
# # st.markdown("Bem-vindo! Descreva seu problema com pragas na lavoura e encontrarei a melhor solução para você.")

# # db, llm = carregar_base_conhecimento()

# # if 'recomendacao' not in st.session_state: st.session_state.recomendacao = ""
# # if 'pergunta' not in st.session_state: st.session_state.pergunta = ""

# # with st.form("pergunta_form"):
# #     pergunta_usuario = st.text_area("Qual praga está afetando sua lavoura e em qual cultura?", height=100, placeholder="Ex: Guanxuma na soja ou 'Produto para ferrugem asiática'")
# #     submitted = st.form_submit_button("Buscar Sugestões")

# # if submitted and pergunta_usuario:
# #     if db is not None:
# #         with st.spinner("Analisando as melhores soluções para você..."):
# #             st.session_state.pergunta = pergunta_usuario
# #             recomendacao_gerada = obter_resposta_assistente(pergunta_usuario, db, llm)
# #             st.session_state.recomendacao = recomendacao_gerada
# #     else:
# #         st.error("A base de conhecimento não pôde ser carregada.")

# # if st.session_state.recomendacao:
# #     # Caso especial para resposta de pergunta vaga
# #     if "Por favor, seja mais específico" in st.session_state.recomendacao:
# #         st.info(st.session_state.recomendacao)
# #     elif "NAO_ENCONTRADO" in st.session_state.recomendacao:
# #         st.warning("Não encontrei produtos específicos para sua solicitação em nossa base de dados.")
# #     else:
# #         st.subheader("Encontrei esta sugestão para você:")
# #         st.markdown(st.session_state.recomendacao)
    
# #     # Lógica dos botões de ação
# #     st.markdown("---")
# #     if "NAO_ENCONTRADO" in st.session_state.recomendacao or "Por favor, seja mais específico" in st.session_state.recomendacao:
# #         st.info("Gostaria de falar com um de nossos consultores para obter ajuda personalizada?")
# #         link_whatsapp_sem_produto = gerar_link_whatsapp(st.session_state.pergunta, "Nenhuma recomendação automática foi gerada.")
# #         st.link_button("🗣️ Falar com um Humano via WhatsApp", link_whatsapp_sem_produto, use_container_width=True)
# #     else:
# #         st.markdown("O que você gostaria de fazer?")
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             if st.button("✅ Confirmar Pedido", type="primary", use_container_width=True):
# #                 enviar_email_confirmacao(st.session_state.pergunta, st.session_state.recomendacao)
# #                 st.session_state.recomendacao = "" # Limpa estado após ação
# #         with col2:
# #             link_whatsapp_com_produto = gerar_link_whatsapp(st.session_state.pergunta, st.session_state.recomendacao)
# #             st.link_button("🗣️ Falar com um Humano via WhatsApp", link_whatsapp_com_produto, use_container_width=True)





