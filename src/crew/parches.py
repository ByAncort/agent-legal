"""Parche a CrewAI 1.15.26: llamadas a tools MCP que fallan sin motivo real.

Síntoma: toda llamada a una tool MCP desde un hilo sin event loop (el caso de las tareas del
crew) devuelve "Error during MCP client disconnect: no running event loop", aunque el servidor
respondió bien.

Causa: `MCPNativeTool._run` hace

    try:
        asyncio.get_running_loop()          # lanza RuntimeError si no hay loop
        ...
    except RuntimeError:
        return asyncio.run(...)             # ← se ejecuta DENTRO del except

Al estar dentro del `except`, Python encadena el RuntimeError original como `__context__` de
cualquier excepción posterior. Al cerrar la conexión, `HTTPTransport._raise_teardown_failure`
recorre esa cadena con `find_transport_failure`, encuentra el "no running event loop" y lo
relanza como si fuera un fallo del transporte.

Arreglo: decidir primero si hay loop y llamar a `asyncio.run` FUERA del bloque `except`.
Reproducción y verificación: `python -m tests.probar_mcp` (escenario "MCPNativeTool").

Segundo ajuste, de configuración: el cliente MCP de CrewAI corta cada llamada a los 30 s y la
fábrica de clientes no deja cambiarlo. Una búsqueda con re-ranking en CPU tarda ~6 s y el
servidor las atiende en serie; con tres auditores pidiendo varias a la vez, la cola supera los
30 s. Se sube el valor por defecto a TIMEOUT_TOOL.
"""
import asyncio
import concurrent.futures
import contextvars

from crewai.mcp import client as cliente_mcp
from crewai.tools.mcp_native_tool import MCPNativeTool

TIMEOUT_TOOL = 180      # segundos por llamada a una tool MCP

_original = MCPNativeTool._run
_init_original = cliente_mcp.MCPClient.__init__


def _init_con_timeout(self, *args, execution_timeout=TIMEOUT_TOOL, **kwargs):
    _init_original(self, *args, execution_timeout=execution_timeout, **kwargs)


def _hay_loop():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _run_corregido(self, **kwargs):
    try:
        if _hay_loop():
            ctx = contextvars.copy_context()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(ctx.run, asyncio.run, self._run_async(**kwargs)).result()
        return asyncio.run(self._run_async(**kwargs))       # fuera de cualquier except
    except Exception as e:
        raise RuntimeError(f"Error executing MCP tool {self.original_tool_name}: {e!s}") from e


def aplicar():
    if MCPNativeTool._run is _original:
        MCPNativeTool._run = _run_corregido
    if cliente_mcp.MCPClient.__init__ is _init_original:
        cliente_mcp.MCPClient.__init__ = _init_con_timeout


aplicar()
