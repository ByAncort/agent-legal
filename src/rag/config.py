import pathlib

# --- Configuración (idéntica al bloque 0 del notebook) -----------------------
RAIZ          = pathlib.Path(__file__).resolve().parents[2]
DATA_RAW      = RAIZ / "data" / "raw"                           # contratos .txt o .pdf
MODEL_EMB     = "intfloat/multilingual-e5-small"                # embeddings multilingües, CPU
MODEL_RERANK  = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"    # re-ranker multilingüe
MODEL_LLM     = "deepseek-chat"      # o "deepseek-reasoner" para razonamiento explícito
DEEPSEEK_URL  = "https://api.deepseek.com"

MAX_CHARS     = 4000   # techo por chunk (~1.000 tokens)
TOP_K_FUSION  = 30     # candidatos que salen de la fusión RRF
TOP_K_CONTEXT = 5      # fragmentos que entran al prompt
USE_RERANKER  = True

# La familia E5 exige estos prefijos: la consulta y el documento se codifican distinto.
PREFIJO_QUERY = "query: "
PREFIJO_DOC   = "passage: "
