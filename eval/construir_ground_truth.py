"""Construye eval/ground_truth.json desde las etiquetas expertas de CUAD.

    python -m eval.construir_ground_truth

Fuente: data/cuad/master_clauses.csv (The Atticus Project, CUAD v1, CC BY 4.0).
Para cada contrato × categoría guarda si la cláusula está presente, los spans anotados por los
abogados y en qué secciones de NUESTRO chunking aparece cada span (para medir también si el
sistema cita la sección correcta).
"""
import ast
import csv
import json
import sys

from src.categorias import CATEGORIAS
from src.rag import corpus
from src.rag.config import RAIZ
from src.rag.verify import cita_existe

CSV_CUAD = RAIZ / "data" / "cuad" / "master_clauses.csv"
SALIDA   = RAIZ / "eval" / "ground_truth.json"

ARCHIVO_CUAD = {
    "MSA_MeritLife_2020":
        "MERITLIFEINSURANCECO_06_19_2020-EX-10.(XIV)-MASTER SERVICES AGREEMENT.PDF",
    "CoBranding_Healthcentral_1999":
        "HealthcentralCom_19991108_S-1A_EX-10.27_6623292_EX-10.27_Co-Branding Agreement.pdf",
    "Licencia_Mantenimiento_SparklingSpring_2002":
        "SPARKLINGSPRINGWATERHOLDINGSLTD_07_03_2002-EX-10.13-SOFTWARE LICENSE AND MAINTENANCE AGREEMENT.PDF",
    "Mantenimiento_XImage_Sagem_1999":
        "IMAGEWARESYSTEMSINC_12_20_1999-EX-10.22-MAINTENANCE AGREEMENT.PDF",
}


def _respuesta(fila, columna):
    # el CSV de CUAD escribe "X-Answer" salvo en una columna, que trae "X- Answer"
    return fila.get(f"{columna}-Answer", fila.get(f"{columna}- Answer", "")).strip()


def _presente(respuesta, spans):
    if respuesta in ("Yes", "No"):
        return respuesta == "Yes"
    return bool(respuesta) or bool(spans)        # Governing Law / Renewal Term traen texto libre


def _candidatos(tramo, largo=120):
    '''El tramo entero y, de respaldo, su comienzo y su final cortados en un espacio: los spans
    de CUAD a veces incluyen el pie de página que nuestra limpieza quitó, o una errata.'''
    tramo = " ".join(tramo.split())
    if len(tramo) <= largo:
        return [tramo]
    inicio = tramo[:largo].rsplit(" ", 1)[0]
    final = tramo[-largo:].split(" ", 1)[-1]
    return [tramo, inicio, final]


def _secciones(doc_id, span):
    texto = lambda s: "\n".join(corpus.chunks()[i]["texto"] for i in corpus.indices_de_seccion(doc_id, s))
    encontradas = set()
    # CUAD marca con <omitted> el texto que salta dentro de un mismo span
    for tramo in (t for t in span.split("<omitted>") if len(t.strip()) >= 15):
        for cand in _candidatos(tramo):
            secciones = {s for s in corpus.secciones(doc_id) if cita_existe(cand, texto(s))}
            if secciones:
                encontradas |= secciones
                break
    return sorted(encontradas, key=int)


def construir():
    filas = {f["Filename"]: f for f in csv.DictReader(open(CSV_CUAD, encoding="utf-8"))}
    verdad = {}
    for doc_id, archivo in ARCHIVO_CUAD.items():
        fila = filas[archivo]
        verdad[doc_id] = {}
        for cat in CATEGORIAS:
            spans = ast.literal_eval(fila[cat.cuad]) if fila[cat.cuad].strip() else []
            ubicados = [{"texto": s, "secciones": _secciones(doc_id, s)} for s in spans]
            verdad[doc_id][cat.clave] = {
                "presente": _presente(_respuesta(fila, cat.cuad), spans),
                "respuesta_cuad": _respuesta(fila, cat.cuad),
                "secciones": sorted({s for u in ubicados for s in u["secciones"]}, key=int),
                "spans": ubicados,
            }
    return verdad


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    verdad = construir()
    SALIDA.write_text(json.dumps(verdad, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'categoría':<28}" + "".join(f"{d[:14]:>16}" for d in verdad))
    for cat in CATEGORIAS:
        celdas = []
        for doc_id in verdad:
            e = verdad[doc_id][cat.clave]
            sin_ubicar = sum(1 for s in e["spans"] if not s["secciones"])
            marca = ("Sí §" + ",".join(e["secciones"])) if e["presente"] else "No"
            celdas.append(marca + (f" ?{sin_ubicar}" if sin_ubicar else ""))
        print(f"{cat.clave:<28}" + "".join(f"{c[:15]:>16}" for c in celdas))

    total = [e["presente"] for d in verdad.values() for e in d.values()]
    print(f"\n{sum(total)} presentes / {len(total) - sum(total)} ausentes → {SALIDA.relative_to(RAIZ)}")
