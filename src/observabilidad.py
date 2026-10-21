"""Observabilidad con Arize Phoenix: trazas de agentes, LLM y tools, todo en local.

    python -m src.observabilidad            # levanta Phoenix en http://localhost:6006
    PHOENIX_TRAZAS=1 python -m src.crew.flow --doc Mantenimiento_XImage_Sagem_1999

Con PHOENIX_TRAZAS=1, cualquier comando del proyecto envía sus trazas a Phoenix: el Flow y el crew
(CrewAI), el chat y el baseline (LangChain) y los agentes desde cero (SDK openai). Cada corrida
aparece como un árbol Flow → crew → tarea → agente → llamada al LLM / llamada a tool, con
prompts, respuestas, tokens y latencias.

Privacidad: Phoenix corre en este equipo (escucha solo en 127.0.0.1) y guarda las trazas en
data/phoenix/. Nada sale hacia servicios externos. Las trazas incluyen texto de los contratos y
de los prompts: no se versionan.

El servidor vive en su propio venv (.venv-phoenix), porque es una aplicación aparte con
dependencias pesadas; en .venv-crew solo se instala la instrumentación (arize-phoenix-otel +
openinference-instrumentation-*).
"""
import contextlib
import json
import os
import subprocess
import sys

from src.rag.config import RAIZ

PUERTO   = int(os.environ.get("PHOENIX_PORT", "6006"))
URL_UI   = f"http://localhost:{PUERTO}"
ENDPOINT = f"{URL_UI}/v1/traces"
PROYECTO = os.environ.get("PHOENIX_PROJECT_NAME", "clausefinder-crew")
DIR_TRAZAS = RAIZ / "data" / "phoenix"

_activo = False


def activar(proyecto=PROYECTO):
    '''Instrumenta CrewAI, LangChain y el SDK openai, y envía las trazas a Phoenix.'''
    global _activo
    if _activo:
        return
    from openinference.instrumentation.crewai import CrewAIInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.otel import register

    proveedor = register(project_name=proyecto, endpoint=ENDPOINT, batch=True, verbose=False)
    # Sin LangChainInstrumentor a propósito: crea sus spans por callbacks sin dejarlos como contexto
    # activo, así que cada llamada del SDK openai hecha por LangChain quedaba como traza huérfana y
    # duplicada (con el costo contado dos veces). Los agentes LangChain se envuelven en un span
    # AGENT propio (src/chat) y sus llamadas al LLM y a las tools cuelgan de él.
    for instrumentador in (CrewAIInstrumentor(), OpenAIInstrumentor()):
        instrumentador.instrument(tracer_provider=proveedor)
    _activo = True
    print(f"[phoenix] trazas → {URL_UI} (proyecto {proyecto})", file=sys.stderr)


def activar_si_corresponde():
    if os.environ.get("PHOENIX_TRAZAS", "").strip().lower() in ("1", "true", "si", "sí"):
        activar()


def sesion(id_sesion):
    '''Marca con el mismo session.id todas las trazas creadas dentro del bloque: en Phoenix
    (pestaña Sessions) una conversación de varios turnos se ve como un solo hilo.'''
    try:
        from openinference.instrumentation import using_session
        return using_session(id_sesion)
    except ImportError:
        return contextlib.nullcontext()


def enviar_trazas():
    '''Envía ya las trazas pendientes (el exportador las agrupa cada ~5 s). Útil en Jupyter,
    donde el proceso no termina y uno quiere verlas en Phoenix enseguida.'''
    from opentelemetry import trace
    proveedor = trace.get_tracer_provider()
    if hasattr(proveedor, "force_flush"):
        proveedor.force_flush()


def tramo(nombre, tipo="CHAIN", entrada=None):
    '''Span propio con las convenciones de OpenInference (tipo AGENT / TOOL / CHAIN), para que una
    corrida completa se vea como un solo árbol en Phoenix. Sin PHOENIX_TRAZAS, OpenTelemetry
    entrega un tracer vacío y esto no cuesta nada.

        with tramo("agente react", "AGENT", entrada=doc_id) as span:
            ...
            salida(span, resultado)
    '''
    from opentelemetry import trace
    atributos = {"openinference.span.kind": tipo}
    if entrada is not None:
        atributos["input.value"] = entrada if isinstance(entrada, str) else json.dumps(entrada, ensure_ascii=False)
    return trace.get_tracer("clausefinder").start_as_current_span(nombre, attributes=atributos)


def salida(span, valor):
    '''Registra la salida y marca el span como OK (si hubo excepción, OpenTelemetry lo marca ERROR).'''
    from opentelemetry.trace import Status, StatusCode
    texto = valor if isinstance(valor, str) else json.dumps(valor, ensure_ascii=False, default=str)
    span.set_attribute("output.value", texto[:20000])
    span.set_status(Status(StatusCode.OK))


def _python_phoenix():
    for relativo in (".venv-phoenix/Scripts/python.exe", ".venv-phoenix/bin/python"):
        ruta = RAIZ / relativo
        if ruta.exists():
            return ruta
    raise SystemExit("No existe .venv-phoenix. Crear con:\n"
                     "  python -m venv .venv-phoenix\n"
                     "  .venv-phoenix\\Scripts\\python -m pip install arize-phoenix")


def servidor():
    DIR_TRAZAS.mkdir(parents=True, exist_ok=True)
    entorno = {**os.environ,
               "PHOENIX_WORKING_DIR": str(DIR_TRAZAS),   # base SQLite de las trazas, dentro del proyecto
               "PHOENIX_HOST": "127.0.0.1",              # solo este equipo, nunca 0.0.0.0
               "PHOENIX_PORT": str(PUERTO)}
    print(f"Phoenix en {URL_UI} · trazas en {DIR_TRAZAS.relative_to(RAIZ).as_posix()}/", flush=True)
    return subprocess.call([str(_python_phoenix()), "-m", "phoenix.server.main", "serve"], env=entorno)


if __name__ == "__main__":
    sys.exit(servidor())
