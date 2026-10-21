"""Lo que comparten los agentes desde cero: la tarea, la traza del ciclo y las métricas."""
import json
import time
from dataclasses import asdict, dataclass, field

from src.categorias import checklist
from src.esquemas import FORMATO_RESPUESTA
from src.observabilidad import salida, tramo
from src.tools.core import ejecutar

# Tools de solo lectura: el agente único no escribe informes.
TOOLS_AGENTE_UNICO = ["buscar_clausulas", "leer_seccion", "verificar_cita"]

MAX_OBSERVACION = 6000     # caracteres de cada observación que vuelven al contexto
MAX_REPETICIONES = 2       # misma llamada con los mismos argumentos: a la tercera se corta


def instrucciones_tarea(doc_id):
    return f"""Eres un analista de riesgo contractual. Revisas el contrato `{doc_id}` y, para CADA
una de las 10 categorías del checklist, determinas si el contrato contiene una cláusula de ese tipo.

Checklist:
{checklist()}

Reglas:
1. Usa las herramientas para leer el contrato; no respondas de memoria.
2. Una categoría está presente solo si encuentras texto del contrato que la respalde. Si tras
   buscar no lo encuentras, márcala como ausente.
3. Para cada categoría presente copia una cita LITERAL (entre 15 y 300 caracteres) con su número
   de sección, y compruébala con verificar_cita antes de entregarla.
4. Busca con términos en inglés: el contrato está en inglés. Usa k=3 para no saturar el contexto.
5. Responde las 10 categorías, presentes y ausentes.

La respuesta final es SOLO este JSON (sin texto adicional):
{FORMATO_RESPUESTA}"""


@dataclass
class Paso:
    '''Una vuelta del ciclo: razonamiento → decisión (acción) → observación (percepción).'''
    n: int
    razonamiento: str
    accion: str | None
    argumentos: dict | None
    observacion: str


@dataclass
class Corrida:
    patron: str
    doc_id: str
    modelo: str
    pasos: list[Paso] = field(default_factory=list)
    llamadas_llm: int = 0
    tokens_entrada: int = 0
    tokens_entrada_cache: int = 0
    tokens_salida: int = 0
    errores_formato: int = 0       # salida que no se pudo interpretar (regex o JSON de argumentos)
    errores_tool: int = 0          # la tool rechazó los argumentos (validación del esquema)
    repeticiones: int = 0          # llamadas idénticas a una anterior
    termino: str = ""              # "respuesta_final" | "max_pasos" | "repeticion" | "error"
    respuesta_valida: bool = False
    respuesta: dict | None = None
    texto_final: str = ""
    segundos: float = 0.0
    _inicio: float = field(default_factory=time.time, repr=False)

    def contar_uso(self, uso):
        if uso is None:
            return
        self.llamadas_llm += 1
        self.tokens_entrada += uso.prompt_tokens or 0
        self.tokens_salida += uso.completion_tokens or 0
        # DeepSeek informa el caché de prefijo como prompt_cache_hit_tokens
        self.tokens_entrada_cache += getattr(uso, "prompt_cache_hit_tokens", 0) or 0

    def cerrar(self, termino):
        self.termino = termino
        self.segundos = round(time.time() - self._inicio, 1)

    def resumen(self):
        d = asdict(self)
        d.pop("_inicio", None)
        return d


def recortar(observacion):
    texto = observacion if isinstance(observacion, str) else json.dumps(observacion, ensure_ascii=False)
    return texto if len(texto) <= MAX_OBSERVACION else texto[:MAX_OBSERVACION] + " …[recortado]"


def costo_usd(corrida, precio_in=0.28, precio_cache=0.028, precio_out=0.42):
    '''Tarifas de lista de DeepSeek-V3.2 (US$/1M tokens), las mismas que usa el notebook §9.'''
    sin_cache = corrida.tokens_entrada - corrida.tokens_entrada_cache
    return (sin_cache * precio_in + corrida.tokens_entrada_cache * precio_cache
            + corrida.tokens_salida * precio_out) / 1e6


def ejecutar_trazado(nombre, argumentos):
    '''ejecutar() de las tools, como span TOOL en Phoenix (si las trazas están activas).'''
    with tramo(nombre, "TOOL", entrada=argumentos) as span:
        resultado = ejecutar(nombre, argumentos)
        salida(span, resultado)
        return resultado


def trazar_agente(patron, ejecutar_ciclo, doc_id, **kwargs):
    '''Corre el ciclo de un agente dentro de un span AGENT: toda la corrida queda en un árbol.'''
    with tramo(f"agente {patron} · {doc_id}", "AGENT", entrada=f"Revisar el contrato {doc_id}") as span:
        corrida = ejecutar_ciclo(doc_id, **kwargs)
        span.set_attribute("clausefinder.termino", corrida.termino)
        span.set_attribute("clausefinder.llamadas_llm", corrida.llamadas_llm)
        salida(span, corrida.respuesta or corrida.texto_final)
        return corrida
