"""Índices denso y BM25, fusión RRF y re-ranking (notebook §3–§7).

Única diferencia con el notebook: `doc_id` opcional para filtrar por contrato antes de rankear.
Sin filtro, el resultado es el mismo que en clausefinder.ipynb.
"""
import functools
import re

import numpy as np

from .config import (MODEL_EMB, MODEL_RERANK, PREFIJO_DOC, PREFIJO_QUERY, TOP_K_CONTEXT,
                     TOP_K_FUSION, USE_RERANKER)
from .corpus import chunks


def tokenizar(texto):
    return re.findall(r"[a-z0-9]+", texto.lower())


# --- Modelos e índices: se construyen una vez por proceso, al primer uso -----

@functools.lru_cache(maxsize=1)
def modelo_emb():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_EMB)


@functools.lru_cache(maxsize=1)
def matriz_emb():
    return modelo_emb().encode(
        [PREFIJO_DOC + c["texto_indexado"] for c in chunks()],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


@functools.lru_cache(maxsize=1)
def indice_bm25():
    from rank_bm25 import BM25Okapi
    return BM25Okapi([tokenizar(c["texto_indexado"]) for c in chunks()])


@functools.lru_cache(maxsize=1)
def reranker():
    if not USE_RERANKER:
        return None
    from sentence_transformers import CrossEncoder
    return CrossEncoder(MODEL_RERANK)


def precargar():
    '''Carga modelos e índices de una vez (útil al arrancar el servidor MCP).'''
    matriz_emb(); indice_bm25(); reranker()


def embeber(textos, prefijo=PREFIJO_QUERY):
    '''Vectores e5 normalizados; los reutiliza la memoria a largo plazo del crew.'''
    return modelo_emb().encode([prefijo + t for t in textos], normalize_embeddings=True)


# --- Búsqueda ---------------------------------------------------------------

def _ordenar(puntajes, k, doc_id):
    if doc_id is None:
        return [int(i) for i in np.argsort(-puntajes)[:k]]
    validos = np.array([i for i, c in enumerate(chunks()) if c["doc_id"] == doc_id], dtype=int)
    return [int(i) for i in validos[np.argsort(-puntajes[validos])][:k]]


def buscar_denso(pregunta, k=TOP_K_FUSION, doc_id=None):
    q = modelo_emb().encode([PREFIJO_QUERY + pregunta], normalize_embeddings=True)[0]
    puntajes = matriz_emb() @ q            # coseno, porque los vectores están normalizados
    return _ordenar(puntajes, k, doc_id)


def buscar_bm25(pregunta, k=TOP_K_FUSION, doc_id=None):
    return _ordenar(indice_bm25().get_scores(tokenizar(pregunta)), k, doc_id)


def fusion_rrf(rankings, k=60):
    acumulado = {}
    for ranking in rankings:
        for posicion, idx in enumerate(ranking):
            acumulado[idx] = acumulado.get(idx, 0.0) + 1.0 / (k + posicion + 1)
    return sorted(acumulado, key=acumulado.get, reverse=True)


def recuperar(pregunta, k=TOP_K_FUSION, doc_id=None):
    return fusion_rrf([buscar_denso(pregunta, k, doc_id), buscar_bm25(pregunta, k, doc_id)])[:k]


def reordenar(pregunta, indices, k=TOP_K_CONTEXT):
    modelo = reranker()
    if modelo is None or not indices:
        return indices[:k]
    pares = [(pregunta, chunks()[i]["texto_indexado"]) for i in indices]
    puntajes = modelo.predict(pares)
    orden = np.argsort(-puntajes)
    return [indices[i] for i in orden[:k]]


def pipeline_retrieval(pregunta, doc_id=None, k=TOP_K_CONTEXT):
    return reordenar(pregunta, recuperar(pregunta, doc_id=doc_id), k=k)


def armar_contexto(indices):
    bloques = []
    for n, idx in enumerate(indices, 1):
        c = chunks()[idx]
        bloques.append(
            f"[F{n}] documento: {c['doc_id']} | sección: {c['seccion']} {c['epigrafe']} | "
            f"página: {c['pagina']}\n{c['texto']}"
        )
    return "\n\n---\n\n".join(bloques)


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Top-k de ClauseFinder para una pregunta.")
    parser.add_argument("pregunta")
    parser.add_argument("--doc", default=None, help="filtrar por doc_id")
    parser.add_argument("-k", type=int, default=TOP_K_CONTEXT)
    args = parser.parse_args()

    for pos, idx in enumerate(pipeline_retrieval(args.pregunta, doc_id=args.doc, k=args.k), 1):
        c = chunks()[idx]
        print(f"{pos}. {c['doc_id']:<24} [{c['seccion']}] {c['epigrafe'][:40]}")
