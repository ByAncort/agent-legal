"""Servidor MCP de ClauseFinder: expone las tools de src/tools/core.py por streamable HTTP local.

    python -m src.mcp_server        # escucha en http://127.0.0.1:8765/mcp

Por qué HTTP y no stdio: CrewAI 1.15 abre un cliente y un transporte NUEVOS en cada llamada a una
tool MCP (`tool_resolver._resolve_native`). Con stdio eso relanza el proceso —y recarga los
modelos, ~50 s, contra un timeout de 30 s— en cada búsqueda. Un servidor de larga vida carga los
índices una vez y atiende todas las llamadas.

Seguridad:
- escucha solo en 127.0.0.1, nunca en 0.0.0.0;
- cada tool declara sus `ToolAnnotations` (solo lectura / escritura);
- `registrar_aprobados` no se expone: aprobar citas es del orquestador, no de un agente;
- el filtrado por agente lo hace el cliente (`create_static_tool_filter` en src/crew).
"""
import contextlib
import inspect
import os
import socket
import subprocess
import sys
import time
from typing import Annotated

from src.rag.config import RAIZ

HOST = "127.0.0.1"
PORT = int(os.environ.get("CLAUSEFINDER_MCP_PORT", "8765"))
# Los clientes entran por "localhost": CrewAI antepone el host al nombre de cada tool y la API de
# OpenAI/DeepSeek exige que un nombre de función empiece con letra ("127_0_0_1_..." se rechaza).
URL  = f"http://localhost:{PORT}/mcp"

SOLO_LECTURA = {"buscar_clausulas", "leer_seccion", "verificar_cita"}


def crear_servidor():
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    from src.tools.core import HERRAMIENTAS, a_json, descripcion, ejecutar

    servidor = FastMCP("clausefinder", host=HOST, port=PORT, stateless_http=True, log_level="WARNING")

    def registrar(nombre):
        modelo, _ = HERRAMIENTAS[nombre]
        # La firma se arma desde el esquema Pydantic de core.py: una sola fuente de JSON Schema.
        parametros = [
            inspect.Parameter(campo, inspect.Parameter.KEYWORD_ONLY,
                              annotation=Annotated[info.annotation, info],
                              default=inspect.Parameter.empty if info.is_required() else info.default)
            for campo, info in modelo.model_fields.items()
        ]

        def herramienta(**argumentos):
            return a_json(ejecutar(nombre, argumentos))

        herramienta.__name__ = nombre
        herramienta.__signature__ = inspect.Signature(parametros, return_annotation=str)

        lectura = nombre in SOLO_LECTURA
        servidor.add_tool(
            herramienta, name=nombre, description=descripcion(nombre),
            annotations=ToolAnnotations(readOnlyHint=lectura, destructiveHint=False,
                                        idempotentHint=lectura, openWorldHint=False),
        )

    for nombre in HERRAMIENTAS:
        registrar(nombre)
    return servidor


# --- Ciclo de vida -----------------------------------------------------------

def esta_activo():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


@contextlib.contextmanager
def servidor_activo(espera_max=300):
    '''Reutiliza el servidor si ya corre; si no, lo lanza y lo detiene al salir.

    El puerto se abre recién después de cargar modelos e índices, así que "puerto abierto"
    equivale a "listo para buscar".
    '''
    if esta_activo():
        yield URL
        return

    log = (RAIZ / "outputs" / "mcp_server.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as salida:
        proceso = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "src.mcp_server"],
                                   cwd=RAIZ, stdout=salida, stderr=subprocess.STDOUT)
        try:
            limite = time.time() + espera_max
            while not esta_activo():
                if proceso.poll() is not None:
                    raise RuntimeError(f"El servidor MCP terminó al arrancar; ver {log}")
                if time.time() > limite:
                    raise TimeoutError(f"El servidor MCP no abrió {URL} en {espera_max} s; ver {log}")
                time.sleep(1)
            yield URL
        finally:
            proceso.terminate()
            proceso.wait(timeout=10)


if __name__ == "__main__":
    from src.rag.retrieve import precargar

    print(f"Cargando corpus, modelos e índices...", file=sys.stderr, flush=True)
    precargar()
    servidor = crear_servidor()
    print(f"Servidor MCP listo en {URL}", file=sys.stderr, flush=True)
    servidor.run(transport="streamable-http")
