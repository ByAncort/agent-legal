"""Demostración de la memoria a largo plazo, en dos procesos separados.

    python -m eval.demo_memoria_larga guardar     # proceso 1: vacía la memoria y guarda lecciones
    python -m eval.demo_memoria_larga recuperar   # proceso 2: el Auditor del contrato siguiente las recupera

Las lecciones salen de una revisión real del Experimento 2 en la que el Defensor descartó un
falso positivo del Auditor. Como `recuperar` corre en otro proceso, lo que recupera viene de
LanceDB en disco (data/memory/), no de la RAM.
"""
import json
import sys

from src.crew.memoria import SCOPE_LECCIONES, guardar_lecciones, lecciones_para, memoria, textos_lecciones
from src.crew.schemas import HallazgoFinal
from src.rag.config import RAIZ

SALIDA = RAIZ / "eval" / "results" / "demo_memoria_larga.json"


def corrida_con_descarte():
    '''La corrida más reciente del crew con al menos un hallazgo descartado por el Defensor.'''
    for ruta in sorted((RAIZ / "eval" / "results" / "exp2").glob("C*__*.json"), reverse=True):
        r = json.loads(ruta.read_text(encoding="utf-8"))
        if r.get("estados", {}).get("descartado"):
            estado = json.loads((RAIZ / r["corrida"] / "state.json").read_text(encoding="utf-8"))
            return estado
    raise SystemExit("No hay corridas con descartes en eval/results/exp2/.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    accion = sys.argv[1] if len(sys.argv) > 1 else "guardar"

    if accion == "guardar":
        estado = corrida_con_descarte()
        finales = [HallazgoFinal.model_validate(f) for f in estado["finales"].values()]
        memoria().reset()
        lecciones, _ = textos_lecciones(estado["doc_id"], finales, estado["mensajes_guardrail"])
        n = guardar_lecciones(estado["doc_id"], finales, estado["mensajes_guardrail"])
        print(f"Memoria vaciada. Desde {estado['doc_id']} se guardaron {n[0]} lecciones y {n[1]} aprobados:")
        for l in lecciones:
            print("  -", l[:220])
        SALIDA.write_text(json.dumps({"origen": estado["doc_id"], "lecciones_guardadas": lecciones},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        datos = json.loads(SALIDA.read_text(encoding="utf-8"))
        siguiente = "Mantenimiento_XImage_Sagem_1999"
        texto = lecciones_para([c for c in ("ip_ownership_assignment", "exclusivity", "non_compete",
                                            "uncapped_liability")], siguiente)
        print(f"Nuevo proceso. El Auditor de {siguiente} recibe en su tarea:\n\n{texto or '(nada)'}")
        datos["recuperado_en_otro_proceso"] = {"contrato": siguiente, "texto_para_el_auditor": texto}
        SALIDA.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
