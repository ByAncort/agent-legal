"""Formato de salida común a todas las configuraciones (agente único y crew): una determinación
por categoría del checklist. Es lo que se compara contra CUAD."""
import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from src.categorias import CATEGORIAS

ClaveCategoria = Literal[tuple(c.clave for c in CATEGORIAS)]


class Determinacion(BaseModel):
    categoria: ClaveCategoria
    presente: bool
    seccion: str | None = Field(None, description="Número de artículo donde está la cláusula.")
    cita: str | None = Field(None, description="Texto literal del contrato que lo respalda.")
    justificacion: str = Field("", description="Una o dos frases, en español.")


class RespuestaRevision(BaseModel):
    doc_id: str
    determinaciones: list[Determinacion]


FORMATO_RESPUESTA = """{
  "doc_id": "<doc_id>",
  "determinaciones": [
    {"categoria": "<clave>", "presente": true, "seccion": "<n>", "cita": "<texto literal>",
     "justificacion": "<1-2 frases>"},
    {"categoria": "<clave>", "presente": false, "seccion": null, "cita": null,
     "justificacion": "<qué se buscó y por qué se descarta>"}
  ]
}"""

RE_BLOQUE_JSON = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)


def extraer_json(texto):
    '''Toma el JSON de la respuesta final: bloque ```json``` o, si no hay, del primer "{" al último "}".'''
    m = RE_BLOQUE_JSON.search(texto)
    crudo = m.group(1) if m else texto[texto.find("{"): texto.rfind("}") + 1]
    return json.loads(crudo)


def parsear_respuesta(texto, doc_id):
    '''RespuestaRevision o None. Las categorías que falten se cuentan como "no determinadas".'''
    try:
        datos = extraer_json(texto)
        datos.setdefault("doc_id", doc_id)
        return RespuestaRevision.model_validate(datos)
    except Exception:
        return None
