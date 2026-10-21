"""Agente con function calling nativo, escrito desde cero.

Mismo ciclo que ReAct, pero sin parsing de texto: las tools se declaran con JSON Schema en
`tools=[...]` y el modelo devuelve las llamadas como datos estructurados (`message.tool_calls`).
Los argumentos se validan contra el mismo esquema Pydantic antes de ejecutar (src/tools/core.py).

    python -m src.agents_scratch.function_calling --doc MSA_MeritLife_2020
"""
import json

from src.agents_scratch.comun import (MAX_REPETICIONES, TOOLS_AGENTE_UNICO, Corrida, Paso,
                                      ejecutar_trazado, trazar_agente,
                                      instrucciones_tarea, recortar)
from src.esquemas import parsear_respuesta
from src.llm_factory import cliente_openai
from src.tools.core import a_json, ejecutar, esquema_openai


def ejecutar_fc(doc_id, max_pasos=30, proveedor=None, verbose=False):
    return trazar_agente("function calling", _ejecutar_fc, doc_id, max_pasos=max_pasos,
                         proveedor=proveedor, verbose=verbose)


def _ejecutar_fc(doc_id, max_pasos=30, proveedor=None, verbose=False):
    cliente, modelo = cliente_openai(proveedor)
    corrida = Corrida(patron="function_calling", doc_id=doc_id, modelo=modelo)
    tools = [esquema_openai(n) for n in TOOLS_AGENTE_UNICO]
    mensajes = [{"role": "system", "content": instrucciones_tarea(doc_id)},
                {"role": "user", "content": f"Revisa el contrato {doc_id}. Empieza."}]
    vistas = {}

    for n in range(1, max_pasos + 1):
        resp = cliente.chat.completions.create(model=modelo, messages=mensajes, tools=tools,
                                               tool_choice="auto", temperature=0.0, max_tokens=4000)
        corrida.contar_uso(resp.usage)
        mensaje = resp.choices[0].message
        razonamiento = (mensaje.content or "").strip()

        if not mensaje.tool_calls:                       # sin llamadas: es la respuesta final
            corrida.texto_final = razonamiento
            respuesta = parsear_respuesta(razonamiento, doc_id)
            if respuesta:
                corrida.respuesta, corrida.respuesta_valida = respuesta.model_dump(), True
                corrida.pasos.append(Paso(n, "", None, None, "respuesta final"))
                corrida.cerrar("respuesta_final")
                return corrida
            corrida.errores_formato += 1
            mensajes.append({"role": "assistant", "content": razonamiento})
            mensajes.append({"role": "user", "content": "Tu respuesta final no es el JSON pedido. "
                                                        "Escríbelo de nuevo, solo el JSON."})
            corrida.pasos.append(Paso(n, razonamiento[:300], None, None, "respuesta final inválida"))
            continue

        mensajes.append(mensaje.model_dump(exclude_none=True))
        for llamada in mensaje.tool_calls:               # puede pedir varias en paralelo
            nombre = llamada.function.name
            try:
                argumentos = json.loads(llamada.function.arguments or "{}")
            except json.JSONDecodeError as e:
                corrida.errores_formato += 1
                resultado = {"error": f"argumentos no son JSON válido: {e.msg}"}
                argumentos = None
            else:
                firma = (nombre, json.dumps(argumentos, sort_keys=True, ensure_ascii=False))
                vistas[firma] = vistas.get(firma, 0) + 1
                if vistas[firma] > 1:
                    corrida.repeticiones += 1
                if vistas[firma] > MAX_REPETICIONES + 1:
                    corrida.pasos.append(Paso(n, razonamiento, nombre, argumentos, "corte por repetición"))
                    corrida.cerrar("repeticion")
                    return corrida
                resultado = (ejecutar_trazado(nombre, argumentos) if nombre in TOOLS_AGENTE_UNICO else
                             {"error": f"herramienta no permitida {nombre!r}"})
            if isinstance(resultado, dict) and "error" in resultado:
                corrida.errores_tool += 1

            observacion = recortar(a_json(resultado))
            corrida.pasos.append(Paso(n, razonamiento, nombre, argumentos, observacion))
            mensajes.append({"role": "tool", "tool_call_id": llamada.id, "content": observacion})
            if verbose:
                print(f"[{n:>2}] {nombre}({json.dumps(argumentos, ensure_ascii=False)[:90]})")

    corrida.cerrar("max_pasos")
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

    c = ejecutar_fc(args.doc, args.max_pasos, verbose=True)
    print(f"\n{c.termino} en {c.llamadas_llm} llamadas LLM, {c.segundos} s · errores de formato "
          f"{c.errores_formato} · errores de tool {c.errores_tool} · US$ {costo_usd(c):.4f}")
    if c.respuesta:
        for d in c.respuesta["determinaciones"]:
            print(f"  {d['categoria']:<28} {'Sí §' + str(d['seccion']) if d['presente'] else 'No'}")
