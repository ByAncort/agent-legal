"""Core engine del agente: un solo lugar decide qué LLM usa cada componente.

    LLM_PROVIDER=deepseek   (por defecto) DeepSeek-V3.2, DEEPSEEK_API_KEY
    LLM_PROVIDER=github     GitHub Models,  GITHUB_TOKEN (PAT con permiso models:read)

Las dos APIs son compatibles con OpenAI, así que el resto del código no cambia de proveedor:
los agentes desde cero usan el SDK `openai`, CrewAI su clase `LLM` y LangChain `ChatOpenAI`.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from src.rag.config import RAIZ

load_dotenv(RAIZ / ".env")
# CrewAI envía telemetría a sus servidores por defecto; aquí no sale nada del equipo.
# (No se usa OTEL_SDK_DISABLED: apagaría también OpenTelemetry local, que es lo que usa Phoenix.)
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")

# Trazas locales en Phoenix si PHOENIX_TRAZAS=1 (src/observabilidad.py). Este módulo lo importan
# todos los agentes (desde cero, LangChain y CrewAI), así que es el punto único de activación.
from src.observabilidad import activar_si_corresponde  # noqa: E402

activar_si_corresponde()


@dataclass(frozen=True)
class Proveedor:
    nombre: str
    base_url: str
    api_key_env: str
    modelo: str
    rpm: int | None          # techo de peticiones por minuto a respetar (None = sin límite propio)

    @property
    def api_key(self):
        clave = os.environ.get(self.api_key_env)
        if not clave:
            raise RuntimeError(f"Falta {self.api_key_env} en .env para usar el proveedor {self.nombre!r}.")
        return clave


PROVEEDORES = {
    "deepseek": Proveedor("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY",
                          os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"), rpm=None),
    # La cuota gratuita de GitHub Models es baja (del orden de 10-15 peticiones/minuto):
    # el crew respeta `rpm` con max_rpm para no recibir HTTP 429.
    "github":   Proveedor("github", "https://models.github.ai/inference", "GITHUB_TOKEN",
                          os.environ.get("GITHUB_MODEL", "openai/gpt-4.1-mini"), rpm=10),
}


def proveedor(nombre=None):
    nombre = (nombre or os.environ.get("LLM_PROVIDER", "deepseek")).lower()
    if nombre not in PROVEEDORES:
        raise ValueError(f"LLM_PROVIDER desconocido {nombre!r}; opciones: {list(PROVEEDORES)}")
    return PROVEEDORES[nombre]


def cliente_openai(nombre=None):
    '''(cliente, modelo) para los agentes escritos desde cero.'''
    from openai import OpenAI
    p = proveedor(nombre)
    return OpenAI(api_key=p.api_key, base_url=p.base_url), p.modelo


def llm_crewai(nombre=None, temperature=0.0, **extra):
    from crewai import LLM
    p = proveedor(nombre)
    if p.nombre == "deepseek":
        # proveedor nativo de CrewAI (OpenAI-compatible), lee DEEPSEEK_API_KEY
        return LLM(model=f"deepseek/{p.modelo}", api_key=p.api_key, temperature=temperature, **extra)
    # GitHub Models: cliente OpenAI apuntando a otro endpoint; el id lleva editor ("openai/...")
    return LLM(model=f"openai/{p.modelo}", base_url=p.base_url, api_key=p.api_key,
               temperature=temperature, **extra)


def chat_langchain(nombre=None, temperature=0.0):
    from langchain_openai import ChatOpenAI
    p = proveedor(nombre)
    return ChatOpenAI(model=p.modelo, base_url=p.base_url, api_key=p.api_key, temperature=temperature)
