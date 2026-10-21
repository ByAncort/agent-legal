"""Agente ReAct (Reason + Act) escrito desde cero.

El modelo escribe texto libre con el formato Thought / Action / Action Input y el programa lo
interpreta con expresiones regulares. Es el patrón visto en clase, con sus debilidades a la
vista: si el modelo se sale del formato o escribe un JSON inválido en Action Input, el paso se
pierde y hay que pedirle que lo corrija. Esos errores se cuentan (`errores_formato`).

    python -m src.agents_scratch.react --doc MSA_MeritLife_2020
"""
import json
import re

from src.agents_scratch.comun import (MAX_REPETICIONES, TOOLS_AGENTE_UNICO, Corrida, Paso,
                                      ejecutar_trazado, trazar_agente,
                                      instrucciones_tarea, recortar)
from src.esquemas import parsear_respuesta
from src.llm_factory import cliente_openai
from src.tools.core import descripcion, ejecutar, esquema_json

RE_FINAL   = re.compile(r"Final Answer:\s*(.*)", re.S)
RE_ACCION  = re.compile(r"Action:\s*`?([\w-]+)`?\s*\n+\s*Action Input:\s*(.*)", re.S)
RE_THOUGHT = re.compile(r"Thought:\s*(.*?)(?:\n\s*(?:Action|Final Answer):|\Z)", re.S)


def prompt_sistema(doc_id, tools=TOOLS_AGENTE_UNICO):
    catalogo = "\n".join(
        f"- {n}: {descripcion(n)}\n  Argumentos (JSON Schema): {json.dumps(esquema_json(n), ensure_ascii=False)}"
        for n in tools)
    return f"""{instrucciones_tarea(doc_id)}

Tienes acceso a estas herramientas:
{catalogo}

Usa EXACTAMENTE este formato, un paso por mensaje:

Thought: qué sabes y qué necesitas averiguar ahora
Action: nombre_de_la_herramienta
Action Input: {{"argumento": "valor"}}

El sistema responderá con "Observation: ..." (no la escribas tú). Repite Thought/Action/Action
Input las veces que haga falta. Cuando tengas las 10 categorías resueltas, escribe:

Thought: tengo todo lo necesario
Final Answer: <el JSON pedido>"""


def _primer_json(texto):
    '''El objeto JSON con que empieza Action Input (puede venir seguido de texto basura).'''
    return json.JSONDecoder().raw_decode(texto.strip())[0]


def ejecutar_react(doc_id, max_pasos=30, proveedor=None, verbose=False):
    return trazar_agente("ReAct", _ejecutar_react, doc_id, max_pasos=max_pasos,
                         proveedor=proveedor, verbose=verbose)


def _ejecutar_react(doc_id, max_pasos=30, proveedor=None, verbose=False):
    cliente, modelo = cliente_openai(proveedor)
    corrida = Corrida(patron="react", doc_id=doc_id, modelo=modelo)
    mensajes = [{"role": "system", "content": prompt_sistema(doc_id)},
                {"role": "user", "content": f"Revisa el contrato {doc_id}. Empieza."}]
    vistas = {}

    for n in range(1, max_pasos + 1):
        resp = cliente.chat.completions.create(
            model=modelo, messages=mensajes, temperature=0.0, max_tokens=4000,
            stop=["Observation:", "\nObservation"],     # que el modelo no invente la observación
        )
        corrida.contar_uso(resp.usage)
        texto = resp.choices[0].message.content or ""
        mensajes.append({"role": "assistant", "content": texto})
        thought = (RE_THOUGHT.search(texto).group(1).strip() if RE_THOUGHT.search(texto) else "")

        final = RE_FINAL.search(texto)
        if final:
            corrida.texto_final = final.group(1).strip()
            respuesta = parsear_respuesta(corrida.texto_final, doc_id)
            if respuesta:
                corrida.respuesta, corrida.respuesta_valida = respuesta.model_dump(), True
                corrida.pasos.append(Paso(n, thought, None, None, "Final Answer"))
                corrida.cerrar("respuesta_final")
                return corrida
            corrida.errores_formato += 1
            observacion = ("Tu Final Answer no es un JSON válido con el formato pedido. "
                           "Escríbelo de nuevo, solo el JSON.")
            corrida.pasos.append(Paso(n, thought, None, None, observacion))
            mensajes.append({"role": "user", "content": f"Observation: {observacion}"})
            continue

        accion = RE_ACCION.search(texto)
        if not accion:
            corrida.errores_formato += 1
            observacion = ("No encontré 'Action:' seguido de 'Action Input:'. Sigue el formato "
                           "Thought / Action / Action Input, o termina con Final Answer.")
            corrida.pasos.append(Paso(n, thought, None, None, observacion))
            mensajes.append({"role": "user", "content": f"Observation: {observacion}"})
            continue

        nombre, crudo = accion.group(1), accion.group(2)
        try:
            argumentos = _primer_json(crudo)
        except json.JSONDecodeError as e:
            corrida.errores_formato += 1
            observacion = f"Action Input no es JSON válido ({e.msg}). Escríbelo como objeto JSON en una línea."
            corrida.pasos.append(Paso(n, thought, nombre, None, observacion))
            mensajes.append({"role": "user", "content": f"Observation: {observacion}"})
            continue

        if nombre not in TOOLS_AGENTE_UNICO:
            resultado = {"error": f"herramienta no permitida {nombre!r}", "disponibles": TOOLS_AGENTE_UNICO}
        else:
            firma = (nombre, json.dumps(argumentos, sort_keys=True, ensure_ascii=False))
            vistas[firma] = vistas.get(firma, 0) + 1
            if vistas[firma] > 1:
                corrida.repeticiones += 1
            if vistas[firma] > MAX_REPETICIONES + 1:
                corrida.pasos.append(Paso(n, thought, nombre, argumentos, "corte por repetición"))
                corrida.cerrar("repeticion")
                return corrida
            resultado = ejecutar_trazado(nombre, argumentos)
        if isinstance(resultado, dict) and "error" in resultado:
            corrida.errores_tool += 1

        observacion = recortar(resultado)
        corrida.pasos.append(Paso(n, thought, nombre, argumentos, observacion))
        mensajes.append({"role": "user", "content": f"Observation: {observacion}"})
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

    c = ejecutar_react(args.doc, args.max_pasos, verbose=True)
    print(f"\n{c.termino} en {len(c.pasos)} pasos, {c.segundos} s · errores de formato "
          f"{c.errores_formato} · errores de tool {c.errores_tool} · US$ {costo_usd(c):.4f}")
    if c.respuesta:
        for d in c.respuesta["determinaciones"]:
            print(f"  {d['categoria']:<28} {'Sí §' + str(d['seccion']) if d['presente'] else 'No'}")
