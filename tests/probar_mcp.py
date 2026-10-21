"""Prueba de punta a punta del servidor MCP con un cliente MCP real (sin LLM).

    python -m tests.probar_mcp

Lanza el servidor si no está corriendo, lista las tools con sus anotaciones y las invoca:
una cita real, una alterada, un doc_id inválido y una búsqueda.
"""
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.mcp_server import servidor_activo

DOC = "MSA_MeritLife_2020"
LLAMADAS = [
    ("verificar_cita", {"cita": "Neither party shall assign, transfer, or subcontract this Agreement",
                        "doc_id": DOC, "seccion": "10"}),
    ("verificar_cita", {"cita": "either party shall assign, transfer, or subcontract this Agreement",
                        "doc_id": DOC, "seccion": "10"}),
    ("leer_seccion",   {"doc_id": "NO_EXISTE", "seccion": "1"}),
    ("buscar_clausulas", {"consulta": "limitation of liability", "doc_id": DOC, "k": 3}),
]


async def probar(url):
    async with streamablehttp_client(url) as (lectura, escritura, _):
        async with ClientSession(lectura, escritura) as sesion:
            await sesion.initialize()
            tools = (await sesion.list_tools()).tools
            print(f"{len(tools)} tools expuestas por MCP:")
            for t in tools:
                a = t.annotations
                print(f"  - {t.name:<17} readOnly={a.readOnlyHint!s:<5} "
                      f"params={list(t.inputSchema['properties'])}")
            print()
            for nombre, args in LLAMADAS:
                r = await sesion.call_tool(nombre, args)
                datos = json.loads(r.content[0].text)
                if "fragmentos" in datos:
                    datos = [f"§{f['seccion']} {f['epigrafe'][:40]}" for f in datos["fragmentos"]]
                print(f"{nombre}({', '.join(f'{k}={str(v)[:30]!r}' for k, v in args.items())})")
                print(f"  → {str(datos)[:220]}\n")


def probar_crewai(url):
    '''La misma tool, invocada como la invoca un agente de CrewAI (MCPNativeTool), desde un hilo
    sin event loop. Sin src/crew/parches.py falla con "no running event loop".'''
    import threading

    import src.crew.parches  # noqa: F401
    from crewai.mcp.client import MCPClient
    from crewai.mcp.transports.http import HTTPTransport
    from crewai.tools.mcp_native_tool import MCPNativeTool

    tool = MCPNativeTool(client_factory=lambda: MCPClient(transport=HTTPTransport(url=url)),
                         tool_name="verificar_cita", tool_schema={"description": "", "args_schema": None},
                         server_name="localhost")
    salida = {}
    hilo = threading.Thread(target=lambda: salida.update(r=tool._run(**LLAMADAS[0][1])))
    hilo.start()
    hilo.join()
    print(f"MCPNativeTool (CrewAI) desde un hilo → {salida.get('r', 'FALLÓ')[:90]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    with servidor_activo() as url:
        asyncio.run(probar(url))
        probar_crewai(url)
