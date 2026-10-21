"""La tarea de revisión resuelta por UN agente LangChain (AgentExecutor).

Es el baseline de agente único del Experimento 2 y el tercer patrón del Experimento 1.
Devuelve una `Corrida`, igual que los agentes desde cero, para medirlos con la misma vara.

    python -m src.chat.revision --doc MSA_MeritLife_2020
"""
import json

from src.agents_scratch.comun import TOOLS_AGENTE_UNICO, Corrida, Paso, instrucciones_tarea, recortar
from src.chat.agente_langchain import ContadorUso, crear_agente
from src.esquemas import parsear_respuesta
from src.llm_factory import proveedor as config_proveedor
from src.observabilidad import salida, tramo


def ejecutar_langchain(doc_id, max_pasos=30, proveedor=None, verbose=False):
    corrida = Corrida(patron="langchain", doc_id=doc_id, modelo=config_proveedor(proveedor).modelo)
    uso = ContadorUso()
    agente = crear_agente(instrucciones_tarea(doc_id), TOOLS_AGENTE_UNICO,
                          max_iteraciones=max_pasos, proveedor=proveedor, verbose=verbose)
    try:
        with tramo(f"agente LangChain · {doc_id}", "AGENT", entrada=f"Revisar el contrato {doc_id}") as span:
            resultado = agente.invoke({"input": f"Revisa el contrato {doc_id}. Empieza."},
                                      config={"callbacks": [uso]})
            salida(span, resultado["output"])
    except Exception as e:
        corrida.texto_final = f"{type(e).__name__}: {e}"
        corrida.cerrar("error")
        return corrida

    vistas = set()
    for n, (accion, observacion) in enumerate(resultado["intermediate_steps"], 1):
        argumentos = accion.tool_input if isinstance(accion.tool_input, dict) else None
        firma = (accion.tool, json.dumps(argumentos, sort_keys=True, ensure_ascii=False))
        corrida.repeticiones += firma in vistas
        vistas.add(firma)
        if accion.tool == "_Exception":                  # salida del modelo que no se pudo interpretar
            corrida.errores_formato += 1
        elif '"error"' in str(observacion)[:40]:
            corrida.errores_tool += 1
        corrida.pasos.append(Paso(n, accion.log.strip()[:300], accion.tool, argumentos, recortar(observacion)))

    corrida.llamadas_llm, corrida.tokens_entrada = uso.llamadas, uso.entrada
    corrida.tokens_salida, corrida.tokens_entrada_cache = uso.salida, uso.cache
    corrida.texto_final = resultado["output"]
    respuesta = parsear_respuesta(corrida.texto_final, doc_id)
    if respuesta:
        corrida.respuesta, corrida.respuesta_valida = respuesta.model_dump(), True
    else:
        corrida.errores_formato += 1
    # AgentExecutor corta con este texto cuando agota max_iterations
    agotado = "stopped due to iteration limit" in corrida.texto_final.lower()
    corrida.cerrar("max_pasos" if agotado else "respuesta_final")
    return corrida


if __name__ == "__main__":
    import argparse
    import sys

    from src.agents_scratch.comun import costo_usd

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="MSA_MeritLife_2020")
    parser.add_argument("--max-pasos", type=int, default=30)
    args = parser.parse_args()

    c = ejecutar_langchain(args.doc, args.max_pasos)
    print(f"{c.termino} en {c.llamadas_llm} llamadas LLM, {len(c.pasos)} tools, {c.segundos} s · "
          f"errores de formato {c.errores_formato} · errores de tool {c.errores_tool} · US$ {costo_usd(c):.4f}")
    if c.respuesta:
        for d in c.respuesta["determinaciones"]:
            print(f"  {d['categoria']:<28} {'Sí §' + str(d['seccion']) if d['presente'] else 'No'}")
