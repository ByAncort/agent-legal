"""Vuelve a puntuar las corridas guardadas con la verificación de citas actual.

Las respuestas de los sistemas no cambian: solo se recalculan las métricas (eval/metricas.py).
Se usó tras corregir la normalización Unicode (U+037E en los PDF), que hacía fallar cualquier
cita con punto y coma.

    python -m eval.repuntuar
"""
import json
import sys

from eval.metricas import puntuar
from src.rag.config import RAIZ

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for carpeta in ("exp1", "exp2"):
        cambiadas = 0
        rutas = sorted((RAIZ / "eval" / "results" / carpeta).glob("*.json"))
        for ruta in rutas:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            doc = datos["doc_id"]
            nuevas = puntuar(datos.get("respuesta"), doc)
            if nuevas != datos["filas"]:
                cambiadas += 1
                datos["filas"] = nuevas
                ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{carpeta}: {cambiadas} de {len(rutas)} corridas con métricas actualizadas")
