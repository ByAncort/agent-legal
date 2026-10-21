"""Copia una corrida completa del crew (estado, trazas e informe) a eval/results/demo_crew/.

outputs/ es de trabajo y no se entrega; la evidencia que lee multiagente.ipynb vive en
eval/results/. Por defecto elige la corrida más reciente que tenga informe; si hay varias,
prefiere una con reintento (muestra el ciclo rechazo → devolución).

    python -m eval.guardar_demo [--corrida outputs/corridas/<carpeta>]
"""
import argparse
import json
import shutil
import sys

from src.rag.config import RAIZ

DESTINO = RAIZ / "eval" / "results" / "demo_crew"


def elegir():
    candidatas = []
    for estado in (RAIZ / "outputs" / "corridas").glob("*/state.json"):
        e = json.loads(estado.read_text(encoding="utf-8"))
        if e.get("informe"):
            candidatas.append((e["intentos"] > 0, estado.parent.name, estado.parent))
    if not candidatas:
        raise SystemExit("No hay corridas con informe en outputs/corridas/.")
    return max(candidatas)[2]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrida", default=None)
    args = parser.parse_args()

    origen = RAIZ / args.corrida if args.corrida else elegir()
    estado = json.loads((origen / "state.json").read_text(encoding="utf-8"))
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    shutil.copytree(origen, DESTINO)
    informes = sorted((RAIZ / "outputs" / "informes").glob(f"{estado['doc_id']}_*.md"))
    if informes:
        shutil.copy2(informes[-1], DESTINO / informes[-1].name)
    print(f"{origen.name} → {DESTINO.relative_to(RAIZ).as_posix()}/ "
          f"({len(list(DESTINO.iterdir()))} archivos)")
