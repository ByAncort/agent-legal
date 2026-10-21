"""Puntaje de una revisión contra las etiquetas expertas de CUAD (eval/ground_truth.json).

Por contrato × categoría:
- detección: VP / FP / FN / VN de "la cláusula está presente";
- fidelidad: la cita entregada existe literalmente en la sección que el sistema dice (verificación
  determinística, la misma tool que usa el Verificador);
- ubicación: en los VP, la sección citada es una de las que contienen el span de CUAD.
Una categoría que el sistema no respondió cuenta como "ausente" y se informa aparte.
"""
import json

from src.categorias import CATEGORIAS
from src.rag.config import RAIZ
from src.tools.core import ejecutar

VERDAD = json.loads((RAIZ / "eval" / "ground_truth.json").read_text(encoding="utf-8"))


def puntuar(respuesta, doc_id):
    '''respuesta: RespuestaRevision como dict (o None si el sistema no entregó nada válido).'''
    dadas = {d["categoria"]: d for d in (respuesta or {}).get("determinaciones", [])}
    filas = []
    for cat in CATEGORIAS:
        esperado = VERDAD[doc_id][cat.clave]
        d = dadas.get(cat.clave)
        predicho = bool(d and d["presente"])
        fila = {"doc_id": doc_id, "categoria": cat.clave, "esperado": esperado["presente"],
                "predicho": predicho, "respondida": d is not None,
                "cita_fiel": None, "seccion_correcta": None}
        if predicho:
            if d.get("cita") and d.get("seccion"):
                v = ejecutar("verificar_cita", {"cita": d["cita"], "doc_id": doc_id,
                                                "seccion": str(d["seccion"])})
                fila["cita_fiel"] = bool(v.get("cita_existe"))
            else:
                fila["cita_fiel"] = False
            if esperado["presente"]:
                fila["seccion_correcta"] = str(d.get("seccion")) in esperado["secciones"]
        filas.append(fila)
    return filas


def _tasa(numerador, denominador):
    return round(numerador / denominador, 3) if denominador else None


def agregar(filas):
    vp = sum(f["predicho"] and f["esperado"] for f in filas)
    fp = sum(f["predicho"] and not f["esperado"] for f in filas)
    fn = sum(not f["predicho"] and f["esperado"] for f in filas)
    vn = sum(not f["predicho"] and not f["esperado"] for f in filas)
    precision, recall = _tasa(vp, vp + fp), _tasa(vp, vp + fn)
    f1 = (round(2 * precision * recall / (precision + recall), 3)
          if precision and recall else 0.0)
    citas = [f["cita_fiel"] for f in filas if f["cita_fiel"] is not None]
    secciones = [f["seccion_correcta"] for f in filas if f["seccion_correcta"] is not None]
    return {
        "vp": vp, "fp": fp, "fn": fn, "vn": vn,
        "precision": precision, "recall": recall, "f1": f1,
        "exactitud": _tasa(vp + vn, len(filas)),
        "fidelidad_cita": _tasa(sum(citas), len(citas)),
        "seccion_correcta": _tasa(sum(secciones), len(secciones)),
        "sin_responder": sum(not f["respondida"] for f in filas),
        # falsos positivos en el control negativo (ausente en los 4 contratos)
        "fp_control_negativo": sum(f["predicho"] for f in filas if f["categoria"] == "liquidated_damages"),
    }
