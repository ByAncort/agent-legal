"""Chat de consulta sobre un contrato (y su informe), con memoria a corto plazo de ventana.

La memoria guarda los últimos `k` turnos pregunta-respuesta y se los pasa al agente en
`chat_history`. Así una pregunta de seguimiento ("¿y esa cláusula tiene excepciones?") se
entiende sin repetir el contexto. Los turnos más antiguos se descartan: la ventana acota el
contexto y el costo, a cambio de olvidar lo lejano.

    python -m src.chat.consulta --doc MSA_MeritLife_2020 [--informe outputs/informes/....md] [--k 4]
"""
import uuid
from collections import deque

from langchain_core.messages import AIMessage, HumanMessage

from src.chat.agente_langchain import ContadorUso, crear_agente
from src.observabilidad import salida, sesion, tramo
from src.rag.config import RAIZ

TOOLS_CONSULTA = ["buscar_clausulas", "leer_seccion", "verificar_cita"]


def sistema_consulta(doc_id, informe=None):
    base = f"""Eres un asistente de consulta sobre el contrato `{doc_id}`. Respondes en español,
breve y directo, usando las herramientas para leer el contrato (está en inglés: busca con
términos en inglés y usa doc_id="{doc_id}").

Reglas:
1. Toda afirmación sobre el contrato va con una cita LITERAL entre comillas y su sección.
2. Si no lo encuentras en el contrato, dilo: "No encontrado en el contrato".
3. Las preguntas pueden referirse a turnos anteriores de la conversación ("esa cláusula",
   "lo anterior"): resuélvelas con el historial.
4. No eres asesor legal: describes lo que dice el contrato."""
    if informe:
        base += f"\n\nInforme de riesgos ya generado para este contrato (úsalo como punto de partida):\n{informe}"
    return base


class ConsultaContrato:
    def __init__(self, doc_id, informe=None, k=4, proveedor=None, verbose=False):
        self.doc_id = doc_id
        self.ventana = deque(maxlen=2 * k)          # k turnos = k preguntas + k respuestas
        self.uso = ContadorUso()
        # todas las preguntas de esta conversación comparten sesión en Phoenix (si hay trazas)
        self.id_sesion = f"consulta-{doc_id}-{uuid.uuid4().hex[:6]}"
        self.agente = crear_agente(sistema_consulta(doc_id, informe), TOOLS_CONSULTA,
                                   max_iteraciones=8, proveedor=proveedor, verbose=verbose)

    def preguntar(self, pregunta):
        with sesion(self.id_sesion), tramo(f"consulta · {self.doc_id}", "AGENT", entrada=pregunta) as span:
            resultado = self.agente.invoke({"input": pregunta, "chat_history": list(self.ventana)},
                                           config={"callbacks": [self.uso]})
            salida(span, resultado["output"])
        respuesta = resultado["output"]
        self.ventana.extend([HumanMessage(pregunta), AIMessage(respuesta)])
        return respuesta, resultado["intermediate_steps"]


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="MSA_MeritLife_2020")
    parser.add_argument("--informe", default=None, help="ruta a un informe .md generado por el crew")
    parser.add_argument("--k", type=int, default=4, help="turnos que recuerda la ventana")
    args = parser.parse_args()

    informe = (RAIZ / args.informe).read_text(encoding="utf-8") if args.informe else None
    chat = ConsultaContrato(args.doc, informe, k=args.k)
    print(f"Consulta sobre {args.doc} (memoria de {args.k} turnos). Línea vacía para salir.\n")
    while (pregunta := input("> ").strip()):
        respuesta, pasos = chat.preguntar(pregunta)
        print(f"\n{respuesta}\n   [{len(pasos)} llamadas a tools · {chat.uso.llamadas} llamadas LLM en la sesión]\n")
