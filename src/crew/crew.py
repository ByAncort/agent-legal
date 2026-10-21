"""Agentes, tareas y crews (CrewAI) armados desde agents.yaml / tasks.yaml.

Permisos: cada agente se conecta al MISMO servidor MCP, pero `create_static_tool_filter` le deja
ver solo sus herramientas (PERMISOS). El Verificador no puede buscar, así que no puede inventar
hallazgos; solo el Redactor puede escribir, y solo con citas aprobadas.

Control de bucles: `max_iter` y `max_execution_time` por agente, `guardrail_max_retries` en las
tareas del Auditor y `max_rpm` cuando el proveedor tiene cuota (GitHub Models).
"""
import functools
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from crewai import Agent, Crew, Process, Task
from crewai.mcp import MCPServerHTTP
from crewai.mcp.filters import create_static_tool_filter

import src.crew.parches  # noqa: F401  corrige las llamadas MCP de CrewAI 1.15 (ver docstring)
from src.categorias import CATEGORIAS, GRUPOS, POR_CLAVE, checklist
from src.crew.schemas import Auditoria, Defensa, Verificaciones
from src.esquemas import extraer_json
from src.llm_factory import llm_crewai, proveedor
from src.tools.core import ejecutar

DIR = Path(__file__).parent

PERMISOS = {
    "auditor":     ["buscar_clausulas"],
    "defensor":    ["buscar_clausulas", "leer_seccion"],
    "verificador": ["verificar_cita", "leer_seccion"],
    "redactor":    ["guardar_informe"],
}
MAX_ITER = {"auditor": 12, "defensor": 15, "verificador": 20, "redactor": 6}
MAX_SEGUNDOS_AGENTE = 900
MAX_RECHAZOS_GUARDRAIL = 2


@dataclass(frozen=True)
class Config:
    '''Variantes del Experimento 2 (ablación).'''
    nombre: str
    defensor: bool = True
    reintentos: int = 0           # vueltas extra del Flow para los hallazgos rechazados
    jerarquico: bool = False      # Process.hierarchical: un manager reparte las tareas
    concurrente: bool = True      # tareas del Auditor con async_execution
    memoria: bool = False         # memoria a largo plazo de lecciones


CONFIGS = {
    "C1":     Config("C1", defensor=False),
    "C2":     Config("C2"),
    "C3":     Config("C3", reintentos=1),
    "C4":     Config("C4", jerarquico=True, concurrente=False),
    "C3-mem": Config("C3-mem", reintentos=1, memoria=True),
}


@functools.lru_cache(maxsize=None)
def _yaml(nombre):
    return yaml.safe_load((DIR / nombre).read_text(encoding="utf-8"))


def _completar(plantilla, **valores):
    return plantilla.format(**valores).strip()


def agente(nombre, doc_id, url, llm, especialidad=None):
    cfg = _yaml("agents.yaml")[nombre]
    rol = cfg["role"].strip() + (f" — especialista en {especialidad}" if especialidad else "")
    return Agent(
        role=rol,
        goal=_completar(cfg["goal"], doc_id=doc_id),
        backstory=cfg["backstory"].strip(),
        llm=llm,
        mcps=[MCPServerHTTP(url=url, cache_tools_list=True,
                            tool_filter=create_static_tool_filter(allowed_tool_names=PERMISOS[nombre]))],
        max_iter=MAX_ITER[nombre],
        max_execution_time=MAX_SEGUNDOS_AGENTE,
        max_rpm=proveedor().rpm,
        allow_delegation=False,
        verbose=True,
    )


# --- Guardrail: la primera línea de rechazo, determinística ------------------

def leer_salida(salida, modelo):
    '''El objeto Pydantic de una TaskOutput, o None si no se pudo interpretar.'''
    if salida is None:
        return None
    if isinstance(getattr(salida, "pydantic", None), modelo):
        return salida.pydantic
    try:
        return modelo.model_validate(extraer_json(salida.raw))
    except Exception:
        return None


class GuardrailCitas:
    '''Rechaza la auditoría si falta una categoría o si una cita no existe literalmente en su
    sección. Tras MAX_RECHAZOS la deja pasar: lo que siga mal lo rechaza el Verificador y lo
    devuelve el Flow, en vez de que CrewAI aborte toda la corrida.'''

    def __init__(self, doc_id, claves):
        self.doc_id, self.claves = doc_id, claves
        self.rechazos = 0
        self.mensajes = []

    def validar(self, salida):
        # método (no __call__): CrewAI hace inspect.getsource() del guardrail
        auditoria = leer_salida(salida, Auditoria)
        if auditoria is None:
            errores = ["la salida no tiene el formato pedido (una lista de hallazgos)"]
        else:
            faltan = sorted(set(self.claves) - {h.categoria for h in auditoria.hallazgos})
            errores = [f"faltan categorías: {faltan}; entrega una entrada por cada una"] if faltan else []
            for h in auditoria.hallazgos:
                if not h.presente:
                    continue
                if not h.cita or not h.seccion:
                    errores.append(f"{h.categoria}: está presente pero le falta cita o sección")
                    continue
                v = ejecutar("verificar_cita", {"cita": h.cita, "doc_id": self.doc_id, "seccion": str(h.seccion)})
                if "error" in v:
                    errores.append(f"{h.categoria}: {'; '.join(v.get('detalle', [v['error']]))}")
                elif not v["en_documento"]:
                    # si la cita existe en otra sección no se rechaza: decidir() corrige la sección
                    errores.append(f"{h.categoria}: la cita no existe literalmente en el contrato; "
                                   "cópiala carácter por carácter del fragmento devuelto por la búsqueda")
        if not errores or self.rechazos >= MAX_RECHAZOS_GUARDRAIL:
            return True, salida
        self.rechazos += 1
        self.mensajes.append(errores)
        return False, "Corrige y entrega de nuevo la auditoría completa:\n- " + "\n- ".join(errores)


# --- Crews ---------------------------------------------------------------

@dataclass
class Revision:
    crew: Crew
    auditorias: list[Task]
    matizar: Task | None
    verificar: Task
    guardrails: list[GuardrailCitas]


def crew_revision(doc_id, claves, config, url, lecciones="", feedback="", log_file=None):
    tareas_yaml = _yaml("tasks.yaml")
    llm = llm_crewai()
    verificador = agente("verificador", doc_id, url, llm)
    defensor = agente("defensor", doc_id, url, llm) if config.defensor else None

    # Descomposición: el checklist se reparte por grupos, uno por tarea. Cada grupo tiene su
    # propio Auditor especialista: CrewAI no permite que una misma instancia de Agent ejecute
    # dos tareas a la vez ("Executor is already running"), y así las tres corren en paralelo.
    grupos = [[c for c in cats if c.clave in claves] for cats in GRUPOS.values()]
    grupos = [g for g in grupos if g]
    asincronas = config.concurrente and not config.jerarquico and len(grupos) > 1
    auditores = [agente("auditor", doc_id, url, llm, especialidad=g[0].grupo) for g in grupos]
    auditorias, guardrails = [], []
    for grupo, auditor in zip(grupos, auditores):
        g = GuardrailCitas(doc_id, [c.clave for c in grupo])
        guardrails.append(g)
        auditorias.append(Task(
            name=f"auditar_{grupo[0].grupo}",
            description=_completar(tareas_yaml["auditar"]["description"], doc_id=doc_id,
                                   checklist=checklist(grupo), lecciones=lecciones, feedback=feedback),
            expected_output=tareas_yaml["auditar"]["expected_output"].strip(),
            agent=auditor, output_pydantic=Auditoria, async_execution=asincronas,
            guardrail=g.validar, guardrail_max_retries=MAX_RECHAZOS_GUARDRAIL + 1,
        ))

    matizar = None
    if defensor:
        matizar = Task(name="matizar",
                       description=_completar(tareas_yaml["matizar"]["description"], doc_id=doc_id),
                       expected_output=tareas_yaml["matizar"]["expected_output"].strip(),
                       agent=defensor, context=auditorias, output_pydantic=Defensa)
    verificar = Task(name="verificar",
                     description=_completar(tareas_yaml["verificar"]["description"], doc_id=doc_id),
                     expected_output=tareas_yaml["verificar"]["expected_output"].strip(),
                     agent=verificador, context=auditorias + ([matizar] if matizar else []),
                     output_pydantic=Verificaciones)

    tareas = auditorias + ([matizar] if matizar else []) + [verificar]
    agentes = auditores + [a for a in (defensor, verificador) if a]
    if config.jerarquico:
        # el manager decide a qué agente delegar cada tarea
        for t in tareas:
            t.agent = None
        # CrewAI solo resuelve `mcps=` para el agente dueño de la tarea (crew.py, _add_mcp_tools);
        # un coworker al que el manager delega trabajo se quedaba sin herramientas ("no tengo
        # acceso a buscar_clausulas"). Se resuelven por adelantado como tools normales.
        for a in agentes:
            a.tools = a.get_mcp_tools(a.mcps)
            a.mcps = None
    crew = Crew(
        agents=agentes, tasks=tareas,
        process=Process.hierarchical if config.jerarquico else Process.sequential,
        manager_llm=llm if config.jerarquico else None,
        memory=False, planning=False, verbose=True, max_rpm=proveedor().rpm,
        output_log_file=str(log_file) if log_file else None,
    )
    return Revision(crew, auditorias, matizar, verificar, guardrails)


def resultados(revision):
    '''(Auditoria unificada, Defensa o None, Verificaciones o None) tras el kickoff.'''
    hallazgos = []
    for t in revision.auditorias:
        a = leer_salida(t.output, Auditoria)
        if a:
            hallazgos.extend(a.hallazgos)
    vistos, unicos = set(), []
    for h in hallazgos:                       # una entrada por categoría (la primera)
        if h.categoria not in vistos:
            vistos.add(h.categoria)
            unicos.append(h)
    defensa = leer_salida(revision.matizar.output, Defensa) if revision.matizar else None
    verificaciones = leer_salida(revision.verificar.output, Verificaciones)
    return Auditoria(hallazgos=unicos), defensa, verificaciones


def _lista_rechazados(rechazados):
    lineas = [f"- {POR_CLAVE[f.categoria].nombre}: {'; '.join(f.razones)[:300]}" for f in rechazados]
    return "\n".join(lineas) or "ninguna"


def crew_informe(doc_id, aprobados, ausentes, rechazados, url, log_file=None):
    tarea_yaml = _yaml("tasks.yaml")["redactar"]
    redactor = agente("redactor", doc_id, url, llm_crewai())
    detalle = [{"categoria": POR_CLAVE[f.categoria].nombre, "severidad": f.severidad,
                "razones_severidad": f.razones, "seccion": f.hallazgo.seccion,
                "afirmacion": f.hallazgo.afirmacion, "cita": f.hallazgo.cita,
                "matiz_defensor": (f"{f.matiz.argumento} Contracita (§{f.matiz.seccion_contracita}): "
                                   f"{f.matiz.contracita}") if f.matiz else None}
               for f in aprobados]
    tarea = Task(name="redactar",
                 description=_completar(tarea_yaml["description"], doc_id=doc_id,
                                        aprobados=json.dumps(detalle, ensure_ascii=False, indent=1),
                                        ausentes=", ".join(POR_CLAVE[c].nombre for c in ausentes) or "ninguna",
                                        rechazados=_lista_rechazados(rechazados)),
                 expected_output=tarea_yaml["expected_output"].strip(), agent=redactor)
    return Crew(agents=[redactor], tasks=[tarea], process=Process.sequential, memory=False,
                planning=False, verbose=True, max_rpm=proveedor().rpm,
                output_log_file=str(log_file) if log_file else None)


def todas_las_claves():
    return [c.clave for c in CATEGORIAS]
