"""Demostración de la memoria a corto plazo (ventana de turnos) del chat de consulta.

La segunda pregunta solo se entiende con la primera ("esa cláusula"). Se corre la misma
conversación con ventana k=4 y sin memoria (k=0) y se guardan ambas transcripciones.

    python -m eval.demo_memoria_corta
"""
import json
import sys

from src.chat.consulta import ConsultaContrato
from src.rag.config import RAIZ

DOC = "MSA_MeritLife_2020"
CONVERSACION = [
    "¿El contrato limita la responsabilidad del contratista? ¿En qué sección?",
    "¿Esa cláusula tiene alguna excepción?",
    "¿Y qué dice la sección que mencionaste en la excepción?",
]


def conversar(k):
    chat = ConsultaContrato(DOC, k=k)
    turnos = []
    for pregunta in CONVERSACION:
        respuesta, pasos = chat.preguntar(pregunta)
        turnos.append({"pregunta": pregunta, "respuesta": respuesta,
                       "tools": [(a.tool, a.tool_input) for a, _ in pasos]})
        print(f"\n[k={k}] > {pregunta}\n{respuesta}\n   tools: {[t for t, _ in turnos[-1]['tools']]}")
    return turnos


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    salida = {"doc_id": DOC, "con_memoria_k4": conversar(4), "sin_memoria_k0": conversar(0)}
    ruta = RAIZ / "eval" / "results" / "demo_memoria_corta.json"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(salida, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n→ {ruta.relative_to(RAIZ).as_posix()}")
