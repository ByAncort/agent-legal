"""Baselines sin multiagente para el Experimento 2.

B0 — ClauseFinder de la Ev.1, sin agentes: por cada categoría, retrieval (filtrado al contrato)
     + UNA llamada al LLM con el SYSTEM_PROMPT del notebook. Flujo lineal entrada → salida.
B1 — un solo agente LangChain (AgentExecutor) con las mismas tools que el crew: src/chat/revision.py
"""
import time

from src.agents_scratch.comun import Corrida
from src.categorias import CATEGORIAS
from src.esquemas import Determinacion, RespuestaRevision
from src.llm_factory import cliente_openai
from src.rag import corpus
from src.rag.retrieve import armar_contexto, pipeline_retrieval
from src.rag.verify import RE_CITA, cita_existe

# SYSTEM_PROMPT de clausefinder.ipynb §7, sin cambios
SYSTEM_PROMPT = '''Eres un asistente de análisis contractual. Respondes preguntas sobre cláusulas
usando EXCLUSIVAMENTE los fragmentos de contrato que se te entregan.

Reglas obligatorias:
1. No uses conocimiento externo. Si la respuesta no está en los fragmentos, responde exactamente:
   "No encontrado en los documentos proporcionados."
2. Toda afirmación debe ir acompañada de una cita LITERAL entre comillas, copiada carácter por
   carácter del fragmento, seguida de su origen en el formato (documento, sección).
3. Si dos contratos regulan el punto de forma distinta, indícalo explícitamente en lugar de
   promediar o generalizar.
4. Responde en español, de forma breve y directa. No agregues advertencias legales genéricas.
5. No eres asesor legal: describes lo que dice el contrato, no recomiendas cursos de acción.'''

NO_ENCONTRADO = "no encontrado en los documentos proporcionados"


def ejecutar_b0(doc_id, proveedor=None):
    cliente, modelo = cliente_openai(proveedor)
    corrida = Corrida(patron="B0_rag_ev1", doc_id=doc_id, modelo=modelo)
    determinaciones = []
    for cat in CATEGORIAS:
        indices = pipeline_retrieval(f"{cat.nombre}: {cat.terminos}", doc_id=doc_id)
        pregunta = (f"¿El contrato contiene una cláusula de {cat.nombre.lower()}? "
                    f"Definición: {cat.definicion} Si no la contiene, responde exactamente "
                    f"\"No encontrado en los documentos proporcionados.\"")
        resp = cliente.chat.completions.create(
            model=modelo, temperature=0.0, max_tokens=1500,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"FRAGMENTOS RECUPERADOS:\n\n{armar_contexto(indices)}"
                                                  f"\n\nPREGUNTA: {pregunta}"}])
        corrida.contar_uso(resp.usage)
        texto = resp.choices[0].message.content or ""

        presente = NO_ENCONTRADO not in texto.lower()
        cita = seccion = None
        if presente:
            citas = RE_CITA.findall(texto)
            cita = citas[0] if citas else None
            # la sección es la del fragmento recuperado que contiene la cita
            for i in indices:
                if cita and cita_existe(cita, corpus.chunks()[i]["texto"]):
                    seccion = corpus.chunks()[i]["seccion"]
                    break
        determinaciones.append(Determinacion(categoria=cat.clave, presente=presente, seccion=seccion,
                                             cita=cita, justificacion=texto[:300]))
    corrida.respuesta = RespuestaRevision(doc_id=doc_id, determinaciones=determinaciones).model_dump()
    corrida.respuesta_valida = True
    corrida.cerrar("respuesta_final")
    return corrida
