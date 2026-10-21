"""Herramientas del sistema: funciones puras + esquemas de argumentos (JSON Schema).

Una sola definición sirve a los cuatro consumidores: el agente ReAct desde cero, el de function
calling nativo, LangChain y el servidor MCP que consume CrewAI. Cambiar una descripción o un
límite aquí lo cambia para todos.

Permisos: las tres primeras son de lectura; `guardar_informe` es la única que escribe, y solo
acepta citas que el orquestador registró como aprobadas (`registrar_aprobados`), que NO es una
tool: ningún agente puede aprobar sus propias citas.
"""
import datetime
import functools
import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.rag import corpus
from src.rag.config import RAIZ
from src.rag.verify import cita_existe

DIR_INFORMES  = RAIZ / "outputs" / "informes"
DIR_APROBADOS = RAIZ / "outputs" / "estado"

MAX_CONSULTA = 300
MIN_CITA     = 15          # mismo umbral que RE_CITA de la Ev.1
MAX_CITA     = 2000
MAX_INFORME  = 60_000

# comillas rectas o tipográficas: el redactor puede usar cualquiera de las dos
RE_CITA_INFORME = re.compile(r'["“]([^"”]{%d,})["”]' % MIN_CITA)


# --- Validación compartida --------------------------------------------------

def _enum_doc_ids(schema):
    '''JSON Schema dinámico: el modelo ve los contratos válidos como enum.'''
    schema["enum"] = corpus.doc_ids()


def _validar_doc_id(valor):
    if valor not in corpus.doc_ids():
        raise ValueError(f"doc_id desconocido {valor!r}. Valores válidos: {corpus.doc_ids()}")
    return valor


def _validar_seccion(doc_id, seccion):
    validas = corpus.secciones(doc_id)
    if seccion not in validas:
        raise ValueError(f"la sección {seccion!r} no existe en {doc_id}. Secciones válidas: {validas}")


class _ConDocId(BaseModel):
    doc_id: str = Field(..., description="Identificador del contrato.",
                        json_schema_extra=_enum_doc_ids)

    @field_validator("doc_id")
    @classmethod
    def validar_doc_id(cls, valor):
        return _validar_doc_id(valor)


class _ConSeccion(_ConDocId):
    seccion: str = Field(..., description="Número de artículo tal como aparece en el contrato, "
                                          "por ejemplo '10'.")

    @model_validator(mode="after")
    def validar_seccion(self):
        _validar_seccion(self.doc_id, self.seccion)
        return self


class BuscarClausulasArgs(_ConDocId):
    """Busca en UN contrato los fragmentos más relevantes para una consulta (retrieval híbrido
    denso + BM25 con re-ranking). Devuelve sección, epígrafe, página y texto de cada fragmento.
    Conviene escribir la consulta con los términos legales en inglés, porque los contratos lo están."""
    consulta: str = Field(..., min_length=3, max_length=MAX_CONSULTA,
                          description="Qué buscar, p. ej. 'limitation of liability cap'.")
    k: int = Field(5, ge=1, le=8, description="Cantidad de fragmentos a devolver (1 a 8).")


class LeerSeccionArgs(_ConSeccion):
    """Devuelve el texto COMPLETO de un artículo de un contrato. Sirve para seguir referencias
    cruzadas ('subject to Section 12') y para leer la excepción que acompaña a una regla."""


class VerificarCitaArgs(_ConDocId):
    """Comprueba de forma determinística si una cita existe LITERALMENTE en la sección indicada
    (ignora mayúsculas y espacios). Si no está ahí pero sí en otra sección, dice en cuáles."""
    # la sección NO se valida contra la segmentación: el agente puede usar la numeración real del
    # contrato, que no siempre coincide; la respuesta le dice dónde está la cita en el sistema
    seccion: str = Field(..., description="Número de artículo donde se afirma que está la cita.")
    cita: str = Field(..., min_length=MIN_CITA, max_length=MAX_CITA,
                      description="Texto exacto copiado del contrato.")


class GuardarInformeArgs(_ConDocId):
    """Guarda el informe final de riesgos en Markdown. Única herramienta de escritura: la ruta la
    fija el sistema y se rechaza el informe si cita algún texto que no pertenezca a un hallazgo
    aprobado por el Verificador."""
    markdown: str = Field(..., min_length=50, max_length=MAX_INFORME,
                          description="Informe completo en Markdown, con las citas entre comillas.")


# --- Implementaciones -------------------------------------------------------

def _texto_seccion(doc_id, seccion):
    return "\n".join(corpus.chunks()[i]["texto"] for i in corpus.indices_de_seccion(doc_id, seccion))


@functools.lru_cache(maxsize=2048)
def _recuperar(consulta, doc_id, k):
    '''El retrieval es determinístico: una consulta repetida (el Defensor suele repetir las del
    Auditor) se responde desde caché en vez de volver a re-rankear (~6 s en CPU).'''
    from src.rag.retrieve import pipeline_retrieval   # carga los modelos solo si se busca
    return tuple(pipeline_retrieval(consulta, doc_id=doc_id, k=k))


def buscar_clausulas(consulta, doc_id, k=5):
    fragmentos = []
    for n, idx in enumerate(_recuperar(consulta.strip(), doc_id, k), 1):
        c = corpus.chunks()[idx]
        fragmentos.append({"ref": f"F{n}", "seccion": c["seccion"], "epigrafe": c["epigrafe"],
                           "pagina": c["pagina"], "texto": c["texto"]})
    return {"doc_id": doc_id, "consulta": consulta, "fragmentos": fragmentos}


def leer_seccion(doc_id, seccion):
    indices = corpus.indices_de_seccion(doc_id, seccion)
    primero = corpus.chunks()[indices[0]]
    return {"doc_id": doc_id, "seccion": seccion, "epigrafe": primero["epigrafe"],
            "pagina": primero["pagina"], "texto": _texto_seccion(doc_id, seccion)}


def donde_aparece(cita, doc_id):
    return [s for s in corpus.secciones(doc_id) if cita_existe(cita, _texto_seccion(doc_id, s))]


def verificar_cita(cita, doc_id, seccion):
    donde = donde_aparece(cita, doc_id)
    resultado = {"cita_existe": seccion in donde, "en_documento": bool(donde),
                 "secciones_donde_aparece": donde}
    if seccion not in corpus.secciones(doc_id):
        resultado["nota"] = (f"La sección '{seccion}' no existe en la segmentación del sistema, que puede "
                             "diferir de la numeración impresa en el contrato. Usa la sección donde aparece la cita.")
    return resultado


def _ruta_aprobados(doc_id):
    return DIR_APROBADOS / f"aprobados_{doc_id}.json"


def registrar_aprobados(doc_id, citas):
    '''Lo llama el orquestador tras el Verificador. No es una tool: ningún agente lo invoca.'''
    _validar_doc_id(doc_id)
    DIR_APROBADOS.mkdir(parents=True, exist_ok=True)
    ruta = _ruta_aprobados(doc_id)
    ruta.write_text(json.dumps({"doc_id": doc_id, "citas": list(citas),
                                "registrado": datetime.datetime.now().isoformat(timespec="seconds")},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


def citas_aprobadas(doc_id):
    ruta = _ruta_aprobados(doc_id)
    return json.loads(ruta.read_text(encoding="utf-8"))["citas"] if ruta.exists() else []


def guardar_informe(doc_id, markdown):
    aprobadas = citas_aprobadas(doc_id)
    if not aprobadas:
        return {"error": f"No hay hallazgos aprobados registrados para {doc_id}; "
                         "el informe no se puede guardar."}

    citas = RE_CITA_INFORME.findall(markdown)
    if not citas:
        return {"error": "El informe no contiene ninguna cita literal entre comillas; "
                         "cada hallazgo debe citar el texto aprobado."}

    no_aprobadas = [c for c in citas if not any(cita_existe(c, a) for a in aprobadas)]
    if no_aprobadas:
        return {"error": "El informe cita texto que no pertenece a ningún hallazgo aprobado. "
                         "Usa solo las citas aprobadas, copiadas tal cual.",
                "citas_no_aprobadas": no_aprobadas[:5]}

    DIR_INFORMES.mkdir(parents=True, exist_ok=True)
    marca = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = DIR_INFORMES / f"{doc_id}_{marca}.md"
    ruta.write_text(markdown, encoding="utf-8")
    return {"guardado": True, "ruta": ruta.relative_to(RAIZ).as_posix(), "citas_verificadas": len(citas)}


# --- Registro y despacho ----------------------------------------------------

HERRAMIENTAS = {
    "buscar_clausulas": (BuscarClausulasArgs, buscar_clausulas),
    "leer_seccion":     (LeerSeccionArgs,     leer_seccion),
    "verificar_cita":   (VerificarCitaArgs,   verificar_cita),
    "guardar_informe":  (GuardarInformeArgs,  guardar_informe),
}


def descripcion(nombre):
    return " ".join(HERRAMIENTAS[nombre][0].__doc__.split())


def esquema_json(nombre):
    '''JSON Schema de los argumentos, sin los títulos que agrega Pydantic.'''
    schema = HERRAMIENTAS[nombre][0].model_json_schema()
    schema.pop("title", None)
    schema.pop("description", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


def esquema_openai(nombre):
    '''Formato `tools=[...]` de la API de chat compatible con OpenAI (function calling nativo).'''
    return {"type": "function",
            "function": {"name": nombre, "description": descripcion(nombre),
                         "parameters": esquema_json(nombre)}}


def ejecutar(nombre, argumentos):
    '''Valida los argumentos contra el esquema y ejecuta. Nunca lanza: un error vuelve como dato
    estructurado para que el agente lo lea y corrija la llamada.'''
    if nombre not in HERRAMIENTAS:
        return {"error": f"herramienta desconocida {nombre!r}", "disponibles": list(HERRAMIENTAS)}
    if not isinstance(argumentos, dict):
        return {"error": "los argumentos deben ser un objeto JSON", "recibido": str(argumentos)[:200]}

    esquema, funcion = HERRAMIENTAS[nombre]
    try:
        args = esquema.model_validate(argumentos)
    except ValidationError as e:
        return {"error": "argumentos inválidos",
                "detalle": [f"{'.'.join(map(str, x['loc'])) or 'argumentos'}: {x['msg']}"
                            for x in e.errors()]}
    try:
        return funcion(**args.model_dump())
    except Exception as e:                 # la tool falla, el agente no se cae
        return {"error": f"fallo interno en {nombre}: {type(e).__name__}: {e}"}


def a_json(resultado):
    return json.dumps(resultado, ensure_ascii=False)
