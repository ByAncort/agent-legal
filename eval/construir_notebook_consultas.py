"""Genera consultas_phoenix.ipynb: consultas en vivo desde Jupyter, con trazas en Phoenix.

    python -m eval.construir_notebook_consultas

A diferencia de multiagente.ipynb (que solo lee resultados guardados), este notebook llama al LLM:
requiere DEEPSEEK_API_KEY, el kernel "Python (ClauseFinder Crew)" y Phoenix corriendo.
"""
import nbformat as nbf

from src.rag.config import RAIZ

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELDAS = [
    md("""# Consultas en vivo con trazas en Phoenix

Cada consulta que se haga aquí aparece en **Phoenix** (http://localhost:6006 → proyecto `clausefinder-crew`)
como un árbol de trazas: agente → llamadas al LLM → llamadas a tools, con prompts, tokens, costo y latencia.

**Antes de empezar**
1. Kernel: **Python (ClauseFinder Crew)** (menú *Kernel → Change kernel*). Es el Python de `.venv-crew`,
   donde están CrewAI, LangChain y la instrumentación.
2. Phoenix corriendo: `python -m src.observabilidad` en una terminal (o el panel de vista previa).
3. `DEEPSEEK_API_KEY` en `.env`. Cada pregunta del chat cuesta del orden de US$ 0,002; una revisión
   completa del crew, unos US$ 0,10."""),
    code("""import pathlib, sys

RAIZ = pathlib.Path.cwd()
sys.path.insert(0, str(RAIZ))
if ".venv-crew" not in sys.executable:
    raise RuntimeError("Este notebook necesita el kernel 'Python (ClauseFinder Crew)' (Kernel → Change kernel).")

# Activar ANTES de crear agentes: instrumenta CrewAI, LangChain y el SDK openai.
from src.observabilidad import URL_UI, activar, enviar_trazas
activar()
print(f"Trazas → {URL_UI}  (proyecto clausefinder-crew)")"""),

    md("""## 1. Chat de consulta (LangChain) — una conversación = una sesión en Phoenix

Todas las preguntas de `chat` comparten `session.id`: en Phoenix, pestaña **Sessions**, se ve la
conversación completa turno a turno. La memoria de ventana (k=4) permite preguntas de seguimiento."""),
    code("""from src.chat.consulta import ConsultaContrato

chat = ConsultaContrato("MSA_MeritLife_2020", k=4)
print("Sesión en Phoenix:", chat.id_sesion)"""),
    code("""respuesta, pasos = chat.preguntar("¿El contrato limita la responsabilidad del contratista? ¿En qué sección?")
print(respuesta)
print(f"\\n[{len(pasos)} llamadas a tools]")
enviar_trazas()   # sin esto aparecen igual, a los ~5 s"""),
    code("""respuesta, pasos = chat.preguntar("¿Esa cláusula tiene alguna excepción?")
print(respuesta)
enviar_trazas()"""),
    code("""# Haz tus propias preguntas: cambia el texto y vuelve a ejecutar esta celda.
respuesta, pasos = chat.preguntar("¿Se puede ceder el contrato a un tercero?")
print(respuesta)
enviar_trazas()"""),

    md("""## 2. Agente único desde cero (function calling)

Revisa un contrato contra el checklist de 10 categorías. En Phoenix: un span `agente function calling`
con sus llamadas al LLM y cada tool como hijo (~1-2 min, ~US$ 0,007)."""),
    code("""from src.agents_scratch.function_calling import ejecutar_fc

c = ejecutar_fc("Mantenimiento_XImage_Sagem_1999")
for d in c.respuesta["determinaciones"]:
    print(f"  {d['categoria']:<28} {'Sí §' + str(d['seccion']) if d['presente'] else 'No'}")
enviar_trazas()"""),

    md("""## 3. El crew completo (Auditor ×3 → Defensor → Verificador → Redactor)

En Phoenix: span raíz `revisión <contrato> · C3` → Flow → crews → cada agente → LLM y tools MCP.
Levanta el servidor MCP si no está corriendo (~50 s) y tarda unos minutos (~US$ 0,10).
Con el MSA suele verse el ciclo rechazo → reintento → aprobación."""),
    code("""from src.crew.flow import revisar_contrato

estado, carpeta = revisar_contrato("MSA_MeritLife_2020", "C3")
for categoria, f in estado.finales.items():
    print(f"  {categoria:<28} {f.estado}")
print("\\nEventos del Flow:")
for ev in estado.eventos:
    print("  ", ev[:140])
enviar_trazas()"""),
]


if __name__ == "__main__":
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELDAS
    nb["metadata"]["kernelspec"] = {"name": "clausefinder-crew", "display_name": "Python (ClauseFinder Crew)",
                                    "language": "python"}
    ruta = RAIZ / "consultas_phoenix.ipynb"
    nbf.write(nb, ruta)
    print(f"→ {ruta.name} ({len(CELDAS)} celdas)")
