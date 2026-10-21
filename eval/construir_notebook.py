"""Genera multiagente.ipynb: la demostración de la Ev.2 sobre resultados ya guardados.

    python -m eval.construir_notebook

El notebook corre de arriba abajo SIN API key: lee eval/results/ y outputs/corridas/. Las
celdas que llaman al LLM están al final, comentadas.
"""
import nbformat as nbf

from src.rag.config import RAIZ

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELDAS = [
    md("""# ClauseFinder Crew — revisión adversarial multiagente de contratos

**Evaluación 2 · ISY0101 Ingeniería de Soluciones con IA (DUOC UC)**

ClauseFinder (Ev.1) responde preguntas sobre cláusulas con RAG: un flujo **lineal**, pregunta → respuesta.
Este notebook muestra su evolución a un **sistema de agentes**: cuatro agentes con herramientas y permisos
distintos revisan un contrato completo contra un checklist de 10 categorías de riesgo, se controlan entre sí
y entregan un informe con citas verificadas.

Todo lo que se muestra aquí se lee de resultados guardados (`eval/results/`, `outputs/corridas/`): el notebook
corre sin API key. Cómo reproducir cada número está en el README, sección 11."""),
    code("""import json, pathlib, sys
from IPython.display import Markdown, display

RAIZ = pathlib.Path.cwd()
sys.path.insert(0, str(RAIZ))
RES = RAIZ / "eval" / "results"
leer = lambda p: json.loads(pathlib.Path(p).read_text(encoding="utf-8"))"""),

    md("""## 1. Herramientas: una sola definición, JSON Schema y validación

Las 4 tools viven en `src/tools/core.py` como funciones puras con un esquema Pydantic. De ese esquema salen el
JSON Schema del function calling, la firma del servidor MCP y el `args_schema` de LangChain. Un argumento
inválido no rompe al agente: vuelve como error estructurado con los valores válidos, para que el agente corrija."""),
    code("""from src.tools.core import ejecutar, esquema_openai
print(json.dumps(esquema_openai("verificar_cita"), ensure_ascii=False, indent=1))"""),
    code("""print(ejecutar("leer_seccion", {"doc_id": "MSA_MeritLife_2020", "seccion": "6"}))   # el MSA salta del 5 al 7
cita = "Neither party shall assign, transfer, or subcontract this Agreement"
print(ejecutar("verificar_cita", {"cita": cita, "doc_id": "MSA_MeritLife_2020", "seccion": "10"}))
# "either" es subcadena de "Neither": la verificación de la Ev.1 la aceptaba; con límites de palabra, no
print(ejecutar("verificar_cita", {"cita": cita.replace("Neither", "either"), "doc_id": "MSA_MeritLife_2020", "seccion": "10"}))"""),

    md("""## 2. Ground truth: etiquetas expertas de CUAD

Los 4 contratos vienen de CUAD, que publica etiquetas de abogados por categoría. 10 categorías × 4 contratos =
40 etiquetas (21 presentes, 19 ausentes). *Liquidated Damages* está ausente en los cuatro: es el control negativo."""),
    code("""verdad = leer(RAIZ / "eval" / "ground_truth.json")
filas = ["| categoría | " + " | ".join(d[:22] for d in verdad) + " |", "|---" * (len(verdad) + 1) + "|"]
for cat in next(iter(verdad.values())):
    celdas = [("Sí §" + ",".join(verdad[d][cat]["secciones"])) if verdad[d][cat]["presente"] else "—" for d in verdad]
    filas.append(f"| {cat} | " + " | ".join(celdas) + " |")
display(Markdown("\\n".join(filas)))"""),

    md("""## 3. Experimento 1 — patrones de ejecución de un agente único

Misma tarea, mismas tools y mismo LLM, con tres formas de implementar el ciclo:
**ReAct desde cero** (texto libre interpretado con regex), **function calling nativo** (JSON Schema, llamadas
estructuradas) y **LangChain AgentExecutor** (el ciclo automatizado por el framework)."""),
    code("""p = RES / "resumen_exp1.md"
display(Markdown(p.read_text(encoding="utf-8") if p.exists() else "_Correr `python -m eval.exp1_patrones`._"))"""),
    md("""### El ciclo percepción → razonamiento → decisión → acción, en una traza real (ReAct)"""),
    code("""trazas = sorted((RES / "exp1").glob("react__*.json"))
if trazas:
    c = leer(trazas[0])
    for p in c["pasos"][:6]:
        print(f"── paso {p['n']}")
        print(f"   razonamiento : {p['razonamiento'][:160]}")
        print(f"   decisión     : {p['accion']}({json.dumps(p['argumentos'], ensure_ascii=False)[:100] if p['argumentos'] else ''})")
        print(f"   percepción   : {p['observacion'][:160]}")"""),

    md("""## 4. El crew: Auditor → Defensor → Verificador → Redactor

| Agente | Tools (filtro MCP) | Restricción | Razón de existir |
|---|---|---|---|
| Auditor | `buscar_clausulas` | solo lectura | propone hallazgos por categoría (3 tareas concurrentes) |
| Defensor | `buscar_clausulas`, `leer_seccion` | solo lectura | busca excepciones, topes y reciprocidad que anulan o atenúan |
| Verificador | `verificar_cita`, `leer_seccion` | **sin búsqueda** | aprueba o rechaza con motivo; no puede inventar hallazgos |
| Redactor | `guardar_informe` | única escritura | solo puede citar texto aprobado |

Las decisiones finales y la severidad salen de **reglas en código** (`src/crew/severity.py`), no de la opinión del LLM.
Un `Flow` con estado Pydantic devuelve al Auditor los hallazgos rechazados (una sola vez)."""),
    code("""DEMO = RES / "demo_crew"          # copia de una corrida completa (python -m eval.guardar_demo)
estados = sorted(DEMO.glob("state.json"))
if estados:
    e = leer(estados[0])
    print(f"{e['doc_id']} · config {e['config']} · {e['segundos']} s · {e['uso']['llamadas_llm']} llamadas LLM")
    print(f"rechazos del guardrail: {e['rechazos_guardrail']} · reintentos del Flow: {e['intentos']}\\n")
    print("Eventos del Flow:")
    for ev in e["eventos"]:
        print("  ", ev[:150])
    print("\\nResultado por categoría:")
    for cat, f in e["finales"].items():
        extra = f"§{f['hallazgo']['seccion']} {f['severidad']}" if f["estado"] == "aprobado" else "; ".join(f["razones"])[:90]
        print(f"   {cat:<28} {f['estado']:<10} {extra}")"""),
    code("""for mensajes in (leer(estados[0])["mensajes_guardrail"] if estados else [])[:2]:
    print("Guardrail rechazó la auditoría:"); [print("  -", m[:200]) for m in mensajes]"""),
    md("""### El ciclo rechazo → devolución (C3 sobre el MSA, Experimento 2)

El Verificador rechaza un hallazgo, el router del Flow lo devuelve al Auditor con el motivo, y en la ronda
siguiente se aprueba. El tope de un solo reintento está en el estado del Flow."""),
    code("""r = DEMO / "reintento_MSA_state.json"
if r.exists():
    e2 = leer(r)
    for ev in e2["eventos"]:
        print("  ", ev[:170])
    f = e2["finales"]["uncapped_liability"]
    print(f"\\nuncapped_liability → {f['estado']} en el intento {f['intento']} (§{f['hallazgo']['seccion']}, {f['severidad']})")"""),
    md("""### El informe que produjo el Redactor"""),
    code("""informes = sorted(DEMO.glob("*.md"))
display(Markdown(informes[-1].read_text(encoding="utf-8")) if informes else Markdown("_Sin informes aún._"))"""),

    md("""## 5. Experimento 2 — ¿aporta el multiagente?

Contra las 40 etiquetas de CUAD. B0 = RAG de la Ev.1 sin agentes; B1 = un agente LangChain con las mismas tools;
C1–C4 = variantes del crew (ver README)."""),
    code("""p = RES / "resumen_exp2.md"
display(Markdown(p.read_text(encoding="utf-8") if p.exists() else "_Correr `python -m eval.exp2_ablacion`._"))"""),

    md("""## 6. Memoria a corto plazo — ventana de turnos

La segunda pregunta ("¿esa cláusula…?") solo se entiende con la primera."""),
    code("""p = RES / "demo_memoria_corta.json"
if p.exists():
    d = leer(p)
    for clave in ("con_memoria_k4", "sin_memoria_k0"):
        print(f"══ {clave}")
        for t in d[clave]:
            print(f"> {t['pregunta']}\\n{t['respuesta'][:500]}\\n")"""),

    md("""## 7. Correr en vivo (requiere `DEEPSEEK_API_KEY` en `.env`)

```python
# Crew completo sobre un contrato (levanta el servidor MCP, ~5-10 min):
# from src.crew.flow import revisar_contrato
# estado, carpeta = revisar_contrato("Mantenimiento_XImage_Sagem_1999", "C3")

# Chat de consulta con memoria de ventana:
# from src.chat.consulta import ConsultaContrato
# chat = ConsultaContrato("MSA_MeritLife_2020", k=4)
# print(chat.preguntar("¿Hay tope de responsabilidad?")[0])
```"""),
]


if __name__ == "__main__":
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELDAS
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ruta = RAIZ / "multiagente.ipynb"
    nbf.write(nb, ruta)
    print(f"→ {ruta.name} ({len(CELDAS)} celdas)")
