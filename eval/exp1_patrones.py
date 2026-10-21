"""Experimento 1: patrones de ejecución de un agente único.

ReAct desde cero (regex) vs function calling nativo (JSON Schema) vs LangChain AgentExecutor.
Misma tarea, mismas tools, mismo LLM. Mide robustez del ciclo (errores de formato, de tool,
repeticiones, cortes), costo, latencia y calidad contra CUAD.

    python -m eval.exp1_patrones [--docs MSA_MeritLife_2020 CoBranding_Healthcentral_1999] [--runs 1]

Cada corrida queda en eval/results/exp1/; si el archivo ya existe se salta (se puede reanudar).
"""
import argparse
import json
import sys

from eval.metricas import agregar, puntuar
from src.agents_scratch.comun import costo_usd
from src.agents_scratch.function_calling import ejecutar_fc
from src.agents_scratch.react import ejecutar_react
from src.chat.revision import ejecutar_langchain
from src.rag.config import RAIZ

DIR = RAIZ / "eval" / "results" / "exp1"
PATRONES = {"react": ejecutar_react, "function_calling": ejecutar_fc, "langchain": ejecutar_langchain}


def correr(patrones, docs, runs):
    DIR.mkdir(parents=True, exist_ok=True)
    for run in range(1, runs + 1):
        for doc in docs:
            for patron in patrones:
                ruta = DIR / f"{patron}__{doc}__{run}.json"
                if ruta.exists():
                    continue
                print(f"→ {patron} · {doc} · corrida {run}", flush=True)
                c = PATRONES[patron](doc)
                datos = c.resumen()
                datos["costo_usd"] = round(costo_usd(c), 5)
                datos["filas"] = puntuar(c.respuesta, doc)
                ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")


def tabla():
    corridas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(DIR.glob("*.json"))]
    lineas = ["| Patrón | Corridas | Completa | Llamadas LLM | Tools | Err. formato | Err. tool | "
              "Repeticiones | Seg. | US$ | Precisión | Recall | F1 | Fidelidad cita |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for patron in PATRONES:
        cs = [c for c in corridas if c["patron"] == patron]
        if not cs:
            continue
        n = len(cs)
        prom = lambda k: sum(c[k] for c in cs) / n
        tools = sum(sum(1 for p in c["pasos"] if p["accion"]) for c in cs) / n
        m = agregar([f for c in cs for f in c["filas"]])
        completas = sum(c["termino"] == "respuesta_final" and c["respuesta_valida"] for c in cs)
        lineas.append(f"| {patron} | {n} | {completas}/{n} | {prom('llamadas_llm'):.1f} | {tools:.1f} | "
                      f"{prom('errores_formato'):.1f} | {prom('errores_tool'):.1f} | {prom('repeticiones'):.1f} | "
                      f"{prom('segundos'):.0f} | {prom('costo_usd'):.4f} | {m['precision']} | {m['recall']} | "
                      f"{m['f1']} | {m['fidelidad_cita']} |")
    return "\n".join(lineas)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", nargs="+", default=["MSA_MeritLife_2020", "CoBranding_Healthcentral_1999"])
    parser.add_argument("--patrones", nargs="+", default=list(PATRONES), choices=list(PATRONES))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--solo-tabla", action="store_true")
    args = parser.parse_args()

    if not args.solo_tabla:
        correr(args.patrones, args.docs, args.runs)
    resumen = tabla()
    (RAIZ / "eval" / "results" / "resumen_exp1.md").write_text(resumen + "\n", encoding="utf-8")
    print("\n" + resumen)
