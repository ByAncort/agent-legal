"""Corpus cargado una sola vez por proceso: documentos y chunks.

No carga modelos: las tools que solo leen o verifican texto (leer_seccion, verificar_cita)
arrancan rápido sin tocar torch.
"""
import functools

from .chunking import construir_chunks
from .ingest import cargar_documentos


@functools.lru_cache(maxsize=1)
def documentos():
    return cargar_documentos()


@functools.lru_cache(maxsize=1)
def chunks():
    return construir_chunks(documentos())


def doc_ids():
    return [d["doc_id"] for d in documentos()]


def secciones(doc_id):
    '''Números de sección del contrato, en orden de aparición y sin repetir.'''
    vistos = []
    for c in chunks():
        if c["doc_id"] == doc_id and c["seccion"] not in vistos:
            vistos.append(c["seccion"])
    return vistos


def indices_de_seccion(doc_id, seccion):
    '''Un artículo largo puede estar partido en varios chunks: se devuelven todos.'''
    return [i for i, c in enumerate(chunks()) if c["doc_id"] == doc_id and c["seccion"] == seccion]
