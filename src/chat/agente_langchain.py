"""Agente individual con LangChain: `create_tool_calling_agent` + `AgentExecutor`.

AgentExecutor automatiza el ciclo que en src/agents_scratch se escribe a mano: llama al modelo,
ejecuta las tools que pida, devuelve la observación y repite hasta la respuesta final o hasta
`max_iterations` (el corte contra bucles infinitos). Las tools son las mismas de
src/tools/core.py, con el mismo esquema Pydantic como `args_schema`.
"""
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool

from src.llm_factory import chat_langchain
from src.agents_scratch.comun import ejecutar_trazado
from src.tools.core import HERRAMIENTAS, a_json, descripcion


def tools_langchain(nombres):
    def crear(nombre):
        modelo, _ = HERRAMIENTAS[nombre]
        return StructuredTool.from_function(
            func=lambda **argumentos: a_json(ejecutar_trazado(nombre, argumentos)),
            name=nombre, description=descripcion(nombre), args_schema=modelo,
            # si LangChain rechaza los argumentos al validar, el error vuelve al agente como dato
            handle_validation_error=lambda e: a_json({"error": "argumentos inválidos",
                                                      "detalle": str(e)[:500]}),
        )
    return [crear(n) for n in nombres]


class ContadorUso(BaseCallbackHandler):
    '''Cuenta llamadas al LLM y tokens (incluye los tokens servidos desde el caché de DeepSeek).'''

    def __init__(self):
        self.llamadas = self.entrada = self.salida = self.cache = 0

    def on_llm_end(self, response, **kwargs):
        self.llamadas += 1
        for generaciones in response.generations:
            for g in generaciones:
                uso = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.entrada += uso.get("input_tokens", 0)
                self.salida += uso.get("output_tokens", 0)
                self.cache += (uso.get("input_token_details") or {}).get("cache_read", 0) or 0


def crear_agente(sistema, nombres_tools, max_iteraciones=30, proveedor=None, verbose=False):
    # el prompt de sistema es literal: sus llaves (el JSON de respuesta) no son variables
    sistema = sistema.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", sistema),
        MessagesPlaceholder("chat_history", optional=True),   # memoria a corto plazo
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),               # el ciclo acción → observación
    ])
    tools = tools_langchain(nombres_tools)
    agente = create_tool_calling_agent(chat_langchain(proveedor), tools, prompt)
    return AgentExecutor(agent=agente, tools=tools, max_iterations=max_iteraciones,
                         handle_parsing_errors=True, return_intermediate_steps=True,
                         verbose=verbose)
