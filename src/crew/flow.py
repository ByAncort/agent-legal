"""Orquestación de la revisión con un Flow de CrewAI y estado estructurado.

    revisar ──► evaluar ──"reintento"──► reauditar ──► evaluar ──"listo"──► redactar
                  (router)                                (router)

- `revisar`: el crew de revisión (Auditor ×3 concurrente → Defensor → Verificador).
- `evaluar`: reglas en código. Si hay hallazgos rechazados y quedan intentos, los devuelve al
  Auditor con el motivo del rechazo; si no, sigue. El tope de intentos está en el estado: el
  ciclo no puede ser infinito (y CrewAI además rechaza un listener que se escuche a sí mismo).
- `reauditar`: vuelve a correr el crew SOLO sobre las categorías rechazadas, con feedback.
- `redactar`: registra las citas aprobadas (lo único que guardar_informe aceptará), corre el
  Redactor, guarda lecciones en la memoria a largo plazo y vuelca el estado a state.json.

    python -m src.crew.flow --doc MSA_MeritLife_2020 --config C3
"""
import datetime
import json
import time

from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel, Field

from src.categorias import CATEGORIAS, POR_CLAVE
from src.crew.crew import CONFIGS, crew_informe, crew_revision, resultados, todas_las_claves
from src.crew.schemas import HallazgoFinal
from src.crew.severity import decidir
from src.esquemas import Determinacion, RespuestaRevision
from src.llm_factory import proveedor
from src.observabilidad import salida, tramo
from src.rag.config import RAIZ
from src.tools.core import registrar_aprobados

DIR_CORRIDAS = RAIZ / "outputs" / "corridas"


class EstadoRevision(BaseModel):
    doc_id: str = ""
    config: str = "C3"
    url_mcp: str = ""
    redactar_informe: bool = True
    dir_corrida: str = ""
    intentos: int = 0
    finales: dict[str, HallazgoFinal] = Field(default_factory=dict)
    eventos: list[str] = Field(default_factory=list)
    rechazos_guardrail: int = 0
    mensajes_guardrail: list[list[str]] = Field(default_factory=list)
    lecciones: str = ""
    informe: str | None = None
    uso: dict[str, int] = Field(default_factory=lambda: {"llamadas_llm": 0, "tokens_entrada": 0,
                                                         "tokens_entrada_cache": 0, "tokens_salida": 0})
    segundos: float = 0.0


class FlujoRevision(Flow[EstadoRevision]):

    # --- ayudantes ---------------------------------------------------------
    def _evento(self, texto):
        marca = datetime.datetime.now().strftime("%H:%M:%S")
        self.state.eventos.append(f"{marca} {texto}")
        print(f"\n>>> [flow] {texto}\n", flush=True)

    def _sumar_uso(self, salida_crew):
        u = getattr(salida_crew, "token_usage", None)
        if u:
            self.state.uso["llamadas_llm"] += u.successful_requests or 0
            self.state.uso["tokens_entrada"] += u.prompt_tokens or 0
            self.state.uso["tokens_entrada_cache"] += u.cached_prompt_tokens or 0
            self.state.uso["tokens_salida"] += u.completion_tokens or 0

    def _ronda(self, claves, feedback="", etiqueta="revision"):
        config = CONFIGS[self.state.config]
        log = f"{self.state.dir_corrida}/crew_{etiqueta}_{self.state.intentos}.txt"
        revision = crew_revision(self.state.doc_id, claves, config, self.state.url_mcp,
                                 lecciones=self.state.lecciones, feedback=feedback, log_file=log)
        salida = revision.crew.kickoff()
        self._sumar_uso(salida)
        for g in revision.guardrails:
            self.state.rechazos_guardrail += g.rechazos
            self.state.mensajes_guardrail.extend(g.mensajes)
        auditoria, defensa, verificaciones = resultados(revision)
        faltantes = set(claves) - {h.categoria for h in auditoria.hallazgos}
        if faltantes:
            self._evento(f"el Auditor no entregó: {sorted(faltantes)} (se cuentan como no determinadas)")
        finales = decidir(self.state.doc_id, auditoria, defensa, verificaciones, intento=self.state.intentos)
        for f in finales:
            self.state.finales[f.categoria] = f
        conteo = {e: sum(f.estado == e for f in finales) for e in ("aprobado", "rechazado", "descartado", "ausente")}
        self._evento(f"ronda {self.state.intentos}: {conteo}")

    # --- pasos del Flow ----------------------------------------------------
    @start()
    def revisar(self):
        config = CONFIGS[self.state.config]
        self._evento(f"revisión de {self.state.doc_id} con {config}")
        if config.memoria:
            from src.crew.memoria import lecciones_para
            self.state.lecciones = lecciones_para(todas_las_claves(), self.state.doc_id)
            self._evento(f"memoria a largo plazo: {self.state.lecciones.count(chr(10) + '- ')} lecciones recuperadas")
        self._ronda(todas_las_claves())

    @listen("reintento")
    def reauditar(self):
        self.state.intentos += 1
        rechazados = [f for f in self.state.finales.values() if f.estado == "rechazado"]
        feedback = ("ATENCIÓN: en la revisión anterior el Verificador RECHAZÓ estos hallazgos. "
                    "Vuelve a buscar y corrige (cita literal correcta, sección correcta, o marca "
                    "la categoría como ausente si el contrato no la establece):\n" + "\n".join(
                        f"- {f.categoria}: afirmaste '{f.hallazgo.afirmacion}' citando la sección "
                        f"{f.hallazgo.seccion}. Motivo del rechazo: {'; '.join(f.razones)}" for f in rechazados))
        self._evento(f"reintento {self.state.intentos}: {[f.categoria for f in rechazados]}")
        self._ronda([f.categoria for f in rechazados], feedback=feedback, etiqueta="reintento")

    @router(or_(revisar, reauditar))
    def evaluar(self):
        rechazados = [f.categoria for f in self.state.finales.values() if f.estado == "rechazado"]
        if rechazados and self.state.intentos < CONFIGS[self.state.config].reintentos:
            return "reintento"
        if rechazados:
            self._evento(f"sin intentos disponibles: {rechazados} quedan rechazados")
        return "listo"

    @listen("listo")
    def redactar(self):
        aprobados = [f for f in self.state.finales.values() if f.estado == "aprobado"]
        aprobados.sort(key=lambda f: ("Alta", "Media", "Baja").index(f.severidad))
        registrar_aprobados(self.state.doc_id, [f.hallazgo.cita for f in aprobados])

        if self.state.redactar_informe and aprobados:
            rechazados = [f for f in self.state.finales.values() if f.estado == "rechazado"]
            ausentes = [c.clave for c in CATEGORIAS if self.state.finales.get(c.clave) is None
                        or self.state.finales[c.clave].estado in ("ausente", "descartado")]
            crew = crew_informe(self.state.doc_id, aprobados, ausentes, rechazados, self.state.url_mcp,
                                log_file=f"{self.state.dir_corrida}/crew_informe.txt")
            salida = crew.kickoff()
            self._sumar_uso(salida)
            self.state.informe = salida.raw
            self._evento(f"Redactor: {salida.raw[:200]}")

        if CONFIGS[self.state.config].memoria:
            from src.crew.memoria import guardar_lecciones
            lecciones, aprobados_mem = guardar_lecciones(self.state.doc_id, list(self.state.finales.values()),
                                                         self.state.mensajes_guardrail)
            self._evento(f"memoria a largo plazo: {lecciones} lecciones y {aprobados_mem} aprobados guardados")
        return self.state


def respuesta_revision(estado):
    '''Estado final → formato común de evaluación (una determinación por categoría).'''
    determinaciones = []
    for c in CATEGORIAS:
        f = estado.finales.get(c.clave)
        if f is None:
            continue
        aprobado = f.estado == "aprobado"
        determinaciones.append(Determinacion(
            categoria=c.clave, presente=aprobado,
            seccion=f.hallazgo.seccion if aprobado else None,
            cita=f.hallazgo.cita if aprobado else None,
            justificacion=f"{f.estado}: {'; '.join(f.razones)[:300]}"))
    return RespuestaRevision(doc_id=estado.doc_id, determinaciones=determinaciones)


def revisar_contrato(doc_id, config="C3", url=None, redactar_informe=True):
    '''Corre el Flow completo dentro del servidor MCP y deja todo en outputs/corridas/<id>/.'''
    from src.mcp_server import servidor_activo

    marca = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_corrida = DIR_CORRIDAS / f"{marca}_{config}_{doc_id}"
    dir_corrida.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    # span raíz: en Phoenix toda la revisión (Flow, crews, tareas, LLM, tools) es un solo árbol
    with servidor_activo() as url_activa, \
            tramo(f"revisión {doc_id} · {config}", "CHAIN", entrada={"doc_id": doc_id, "config": config}) as span:
        flujo = FlujoRevision()
        flujo.kickoff(inputs={"doc_id": doc_id, "config": config, "url_mcp": url or url_activa,
                              "redactar_informe": redactar_informe,
                              "dir_corrida": dir_corrida.as_posix()})
        salida(span, {c: f.estado for c, f in flujo.state.finales.items()})
    estado = flujo.state
    estado.segundos = round(time.time() - inicio, 1)
    (dir_corrida / "state.json").write_text(estado.model_dump_json(indent=2), encoding="utf-8")
    return estado, dir_corrida


if __name__ == "__main__":
    import argparse
    import sys

    from src.agents_scratch.comun import costo_usd

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="MSA_MeritLife_2020")
    parser.add_argument("--config", default="C3", choices=list(CONFIGS))
    parser.add_argument("--sin-informe", action="store_true", help="no correr el Redactor")
    args = parser.parse_args()

    estado, carpeta = revisar_contrato(args.doc, args.config, redactar_informe=not args.sin_informe)

    class _Uso:   # adaptador para reutilizar costo_usd
        tokens_entrada = estado.uso["tokens_entrada"]
        tokens_entrada_cache = estado.uso["tokens_entrada_cache"]
        tokens_salida = estado.uso["tokens_salida"]

    print("\n" + "═" * 78)
    print(f"{args.doc} · {args.config} · {proveedor().nombre} · {estado.segundos} s · "
          f"{estado.uso['llamadas_llm']} llamadas LLM · US$ {costo_usd(_Uso):.4f}")
    print(f"rechazos del guardrail: {estado.rechazos_guardrail} · reintentos del Flow: {estado.intentos}")
    for c in CATEGORIAS:
        f = estado.finales.get(c.clave)
        if f is None:
            print(f"  {c.clave:<28} (no determinada)")
            continue
        extra = f" §{f.hallazgo.seccion} {f.severidad}" if f.estado == "aprobado" else ""
        print(f"  {c.clave:<28} {f.estado}{extra}" + (f"  ← {f.razones[0][:70]}" if f.estado in ("rechazado", "descartado") else ""))
    print(f"\nEstado y trazas en {carpeta.relative_to(RAIZ).as_posix()}/")
