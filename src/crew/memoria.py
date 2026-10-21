"""Memoria a largo plazo del crew: lecciones de revisiones anteriores.

Qué se aprende: los errores del Auditor que el resto del equipo atrapó en un contrato, es decir
los hallazgos que el Defensor DESCARTÓ con evidencia (falsos positivos, p. ej. "una licencia no
es cesión de propiedad intelectual"), los que el Verificador RECHAZÓ y las citas que el
guardrail devolvió. Al revisar el siguiente contrato, el Auditor recupera por similitud
semántica las lecciones de sus categorías y las recibe como advertencia en su tarea. Los
hallazgos aprobados también se guardan, como ejemplos de lo que sí pasa la verificación.

Usa la `Memory` unificada de CrewAI (LanceDB en data/memory/, persiste entre procesos) con dos
cambios respecto de la configuración por defecto, que dependería de OpenAI:
- embedder: e5 multilingüe local, el mismo del retrieval de la Ev.1 (src/rag/retrieve.embeber);
- LLM de análisis: el mismo proveedor que el resto del sistema (DeepSeek por defecto).
"""
import functools

from src.categorias import POR_CLAVE
from src.llm_factory import llm_crewai
from src.rag.config import RAIZ

DIR_MEMORIA = RAIZ / "data" / "memory"
SCOPE_LECCIONES = "/revision/lecciones"
SCOPE_APROBADOS = "/revision/aprobados"


def _embedder(textos):
    from src.rag.retrieve import embeber
    # listas, no un ndarray: CrewAI evalúa `if not resultado`, que con un ndarray es ambiguo
    return embeber(textos).tolist()


@functools.lru_cache(maxsize=1)
def memoria():
    from crewai import Memory
    DIR_MEMORIA.mkdir(parents=True, exist_ok=True)
    return Memory(llm=llm_crewai(), embedder=_embedder, storage=str(DIR_MEMORIA))


def textos_lecciones(doc_id, finales, mensajes_guardrail=()):
    lecciones, aprobados = [], []
    for f in finales:
        nombre = POR_CLAVE[f.categoria].nombre
        if f.estado == "descartado":
            lecciones.append(
                f"[{f.categoria}] FALSO POSITIVO del Auditor en {doc_id}: afirmó '{f.hallazgo.afirmacion[:200]}' "
                f"(§{f.hallazgo.seccion}) y el Defensor lo descartó con evidencia: {'; '.join(f.razones)[:400]} "
                f"Lección al auditar '{nombre}': no confundir esa cláusula con esta categoría.")
        elif f.estado == "rechazado":
            lecciones.append(
                f"[{f.categoria}] Hallazgo RECHAZADO por el Verificador en {doc_id} (§{f.hallazgo.seccion}): "
                f"{'; '.join(f.razones)[:400]} Lección al auditar '{nombre}': no repetir este error.")
        elif f.estado == "aprobado":
            aprobados.append(f"[{f.categoria}] Hallazgo APROBADO en {doc_id} (§{f.hallazgo.seccion}): "
                             f"{f.hallazgo.afirmacion[:300]}")
    for errores in mensajes_guardrail:
        for e in errores:
            lecciones.append(f"Cita devuelta por el control automático en {doc_id}: {e[:300]} "
                             "Lección: copiar la cita carácter por carácter del fragmento devuelto.")
    return lecciones, aprobados


def guardar_lecciones(doc_id, finales, mensajes_guardrail=()):
    '''Guarda lecciones y aprobados de una revisión. Devuelve (lecciones, aprobados) guardados.'''
    lecciones, aprobados = textos_lecciones(doc_id, finales, mensajes_guardrail)
    m = memoria()
    if lecciones:
        m.remember_many(lecciones, scope=SCOPE_LECCIONES, importance=0.9, source=doc_id)
    if aprobados:
        m.remember_many(aprobados, scope=SCOPE_APROBADOS, importance=0.5, source=doc_id)
    # remember_many escribe en un pool de hilos: sin esperar, el proceso puede terminar antes de
    # persistir (en las primeras corridas C3-mem data/memory/ quedó vacío por esto)
    m.drain_writes()
    return len(lecciones), len(aprobados)


def lecciones_para(categorias, doc_id, limite=4):
    '''Texto para la tarea del Auditor con las lecciones más parecidas, de OTROS contratos.'''
    consulta = "errores del auditor descartados o rechazados al revisar: " + ", ".join(
        POR_CLAVE[c].nombre for c in categorias)
    encontrados = memoria().recall(consulta, scope=SCOPE_LECCIONES, limit=limite, depth="shallow")
    lecciones = [r.record.content for r in encontrados if doc_id != (r.record.source or "")]
    if not lecciones:
        return ""
    return ("Lecciones de revisiones anteriores (errores que el equipo atrapó en otros contratos; "
            "evita repetirlos):\n" + "\n".join(f"- {l}" for l in lecciones))
