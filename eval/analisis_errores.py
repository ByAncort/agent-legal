"""Dónde falla cada configuración del Experimento 2: falsos positivos, falsos negativos y citas
infieles, con la justificación que dio el sistema.

    python -m eval.analisis_errores [--configs B0 B1 C2]
"""
import argparse
import json
import sys
from collections import defaultdict

from src.rag.config import RAIZ

DIR = RAIZ / "eval" / "results" / "exp2"


def errores(configs=None):
    por_config = defaultdict(list)
    for ruta in sorted(DIR.glob("*.json")):
        c = json.loads(ruta.read_text(encoding="utf-8"))
        if configs and c["config"] not in configs:
            continue
        dets = {d["categoria"]: d for d in c["respuesta"]["determinaciones"]}
        for f in c["filas"]:
            tipo = ("FP" if f["predicho"] and not f["esperado"] else
                    "FN" if f["esperado"] and not f["predicho"] else
                    "cita infiel" if f["cita_fiel"] is False else
                    "sección errónea" if f["seccion_correcta"] is False else None)
            if tipo:
                d = dets.get(f["categoria"], {})
                por_config[c["config"]].append(
                    (tipo, c["doc_id"], f["categoria"], d.get("seccion"), (d.get("justificacion") or "")[:160]))
    return por_config


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*")
    args = parser.parse_args()

    lineas = []
    for config, lista in errores(args.configs).items():
        lineas.append(f"\n### {config} ({len(lista)} errores)")
        for tipo, doc, cat, sec, just in sorted(lista):
            lineas.append(f"- **{tipo}** · {doc} · `{cat}` · §{sec} — {just}")
    texto = "\n".join(lineas)
    (RAIZ / "eval" / "results" / "errores_exp2.md").write_text(texto + "\n", encoding="utf-8")
    print(texto)
