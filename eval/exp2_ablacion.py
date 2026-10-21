"""Experimento 2: multiagente vs agente único, contra las etiquetas expertas de CUAD.

    B0      ClauseFinder Ev.1 (sin agentes)
    B1      un agente LangChain con las mismas tools
    C1      Auditor + Verificador
    C2      Auditor + Defensor + Verificador
    C3      C2 + reintento de los rechazados (Flow)
    C4      C2 con Process.hierarchical
    C3-mem  C3 con memoria a largo plazo (contratos en orden, memoria vacía al inicio)

    python -m eval.exp2_ablacion [--configs B0 B1 C1 C2 C3 C4 C3-mem] [--runs 1]

El Redactor no corre aquí (no afecta la detección); se evalúa aparte. Cada corrida queda en
eval/results/exp2/; si el archivo ya existe se salta, así que se puede interrumpir y reanudar.
"""
import argparse
import json
import sys

from eval.metricas import agregar, puntuar
from src.agents_scratch.comun import costo_usd
from src.rag.config import RAIZ

DIR = RAIZ / "eval" / "results" / "exp2"
DOCS = ["MSA_MeritLife_2020", "CoBranding_Healthcentral_1999",
        "Licencia_Mantenimiento_SparklingSpring_2002", "Mantenimiento_XImage_Sagem_1999"]
CONFIGS = ["B0", "B1", "C1", "C2", "C3", "C4", "C3-mem"]


class _Uso:
    def __init__(self, uso):
        self.tokens_entrada = uso["tokens_entrada"]
        self.tokens_entrada_cache = uso["tokens_entrada_cache"]
        self.tokens_salida = uso["tokens_salida"]


def una_corrida(config, doc):
    if config in ("B0", "B1"):
        from eval.baselines import ejecutar_b0
        from src.chat.revision import ejecutar_langchain
        c = ejecutar_b0(doc) if config == "B0" else ejecutar_langchain(doc)
        return {"respuesta": c.respuesta, "segundos": c.segundos, "llamadas_llm": c.llamadas_llm,
                "costo_usd": round(costo_usd(c), 5), "termino": c.termino,
                "tokens": {"entrada": c.tokens_entrada, "cache": c.tokens_entrada_cache, "salida": c.tokens_salida}}

    from src.crew.flow import respuesta_revision, revisar_contrato
    estado, carpeta = revisar_contrato(doc, config, redactar_informe=False)
    estados = [f.estado for f in estado.finales.values()]
    return {"respuesta": respuesta_revision(estado).model_dump(), "segundos": estado.segundos,
            "llamadas_llm": estado.uso["llamadas_llm"], "costo_usd": round(costo_usd(_Uso(estado.uso)), 5),
            "termino": "respuesta_final",
            "tokens": {"entrada": estado.uso["tokens_entrada"], "cache": estado.uso["tokens_entrada_cache"],
                       "salida": estado.uso["tokens_salida"]},
            "estados": {e: estados.count(e) for e in ("aprobado", "rechazado", "descartado", "ausente")},
            "rechazos_guardrail": estado.rechazos_guardrail, "reintentos": estado.intentos,
            "corrida": carpeta.relative_to(RAIZ).as_posix()}


def correr(configs, runs):
    from src.mcp_server import servidor_activo
    DIR.mkdir(parents=True, exist_ok=True)
    with servidor_activo():                         # un solo servidor MCP para todo el experimento
        for run in range(1, runs + 1):
            for config in configs:
                if config == "C3-mem":
                    pendientes = [d for d in DOCS if not (DIR / f"{config}__{d}__{run}.json").exists()]
                    if pendientes == DOCS:          # corrida nueva: memoria vacía
                        from src.crew.memoria import memoria
                        memoria().reset()
                for doc in DOCS:
                    ruta = DIR / f"{config}__{doc}__{run}.json"
                    if ruta.exists():
                        continue
                    print(f"\n######## {config} · {doc} · corrida {run}\n", flush=True)
                    datos = {"config": config, "doc_id": doc, "run": run, **una_corrida(config, doc)}
                    datos["filas"] = puntuar(datos["respuesta"], doc)
                    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")


def tabla():
    corridas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(DIR.glob("*.json"))]
    lineas = ["| Config | Corridas | Precisión | Recall | F1 | Exactitud | Fidelidad cita | Sección correcta | "
              "FP control (−) | Llamadas LLM | Seg. | US$ / contrato |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for config in CONFIGS:
        cs = [c for c in corridas if c["config"] == config]
        if not cs:
            continue
        m = agregar([f for c in cs for f in c["filas"]])
        n = len(cs)
        prom = lambda k: sum(c[k] for c in cs) / n
        lineas.append(f"| {config} | {n} | {m['precision']} | {m['recall']} | {m['f1']} | {m['exactitud']} | "
                      f"{m['fidelidad_cita']} | {m['seccion_correcta']} | {m['fp_control_negativo']} | "
                      f"{prom('llamadas_llm'):.0f} | {prom('segundos'):.0f} | {prom('costo_usd'):.4f} |")
    return "\n".join(lineas)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--solo-tabla", action="store_true")
    args = parser.parse_args()

    if not args.solo_tabla:
        correr(args.configs, args.runs)
    resumen = tabla()
    (RAIZ / "eval" / "results" / "resumen_exp2.md").write_text(resumen + "\n", encoding="utf-8")
    print("\n" + resumen)
