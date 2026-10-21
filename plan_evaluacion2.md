# Plan Evaluación 2: ClauseFinder Crew, revisión adversarial multiagente de contratos

## Contexto

La Evaluación 1 entregó **ClauseFinder**, un RAG híbrido (e5 + BM25 + RRF + re-ranker) sobre 4 contratos
reales de CUAD, con generación en DeepSeek y verificación literal de citas (`clausefinder.ipynb`,
`README.md`).

La Evaluación 2 exige un **sistema multiagente** que aplique lo visto en la unidad de agentes autónomos
(ISY0101): ciclo de agente, ReAct, function calling, memoria, MCP, control de bucles, LangChain y CrewAI.
Las condiciones son estas:
- Menos de 2 semanas y trabajo individual.
- Stack **CrewAI + DeepSeek**, con GitHub Models como proveedor alternativo.
- No hay rúbrica, así que se asumen estos entregables: código, README técnico, evaluación con números y demo.

**Caso:** un analista de compliance recibe un contrato del proveedor y necesita un informe de riesgos.
El informe lista las cláusulas desfavorables con cita literal, matiz y severidad calculada por reglas.
Después puede hacer preguntas de seguimiento sobre el informe.

**Cómo se defiende que no es un "pipeline disfrazado":**
- Cada agente tiene herramientas y permisos distintos.
- Hay un verificador que **rechaza y devuelve** trabajo.
- Una ablación contra las etiquetas expertas de CUAD mide cuánto aporta cada agente frente a un solo agente.

Entorno verificado: **Python 3.11.0**, compatible con CrewAI (≥3.10, <3.14). `pip` no está en el PATH,
así que se usa `python -m pip`.

---

## Matriz de cobertura: contenido de la unidad → implementación

Esta tabla va también en el README, porque es la respuesta directa a "¿se aplicó lo enseñado?".

| Módulo del curso | Qué se implementa | Dónde | Evidencia en la entrega |
|---|---|---|---|
| **1. LLM vs agente; ciclo percepción → razonamiento → decisión → acción** | Diagrama del ciclo anotado con una traza real: la observación de la tool es la percepción, el Thought es el razonamiento, la elección de tool o veredicto es la decisión, y la llamada o escritura es la acción. Comparación explícita con ClauseFinder Ev.1, que es lineal (entrada → salida) | README §Ev.2 | Traza `verbose` guardada |
| **1. Cuatro componentes** | **Core engine:** DeepSeek vía `llm_factory.py`. **Memoria:** §3. **Tools:** servidor MCP. **Planificación:** Flow + tareas del Crew + manager jerárquico | Tabla en README | — |
| **2. ReAct desde cero** | Agente único con prompt Thought/Action/Action Input/Observation, parsing con regex y `max_steps`. Cuenta los errores de parsing | `src/agents_scratch/react.py` | Experimento 1 |
| **2. Function calling nativo** | Mismo agente y mismas tools, pero con `tools=[{type:function, …}]` en JSON Schema vía SDK `openai` contra DeepSeek. Se reutiliza el `client` de la Ev.1. Sin regex | `src/agents_scratch/function_calling.py` | Experimento 1: tasa de errores ReAct vs FC |
| **3. Memoria a corto plazo** | Ventana de los últimos *k* turnos en el agente conversacional de consulta. Dentro del Crew, el `context=[…]` entre tareas | `src/chat/consulta.py` | Demo: pregunta de seguimiento con pronombre ("¿y esa cláusula tiene tope?") |
| **3. Memoria a largo plazo** | `Memory` de CrewAI (LanceDB en `data/memory/`) con embedder propio, que es una función que reutiliza e5 de la Ev.1, y DeepSeek como LLM de análisis, **sin OpenAI**. Guarda hallazgos aprobados y **rechazos con su motivo**, que el Auditor consulta al analizar el siguiente contrato | `src/crew/memoria.py` | Tasa de rechazo con memoria ON vs OFF en los contratos 2–4 |
| **4. Descomposición y selección de tools de lectura y escritura** | El checklist de 10 categorías se divide en tareas. Las tools de lectura (buscar, leer sección, verificar) están separadas de la de escritura (guardar informe) | Crew + MCP | — |
| **4. JSON Schema + validación de argumentos** | Cada tool tiene un `args_schema` Pydantic: `doc_id` debe existir, `seccion` debe existir en ese doc y `consulta` tiene un máximo de 300 caracteres. Un argumento inválido devuelve un error estructurado con los valores válidos, y el agente lo corrige (evaluación adaptativa) | `src/tools/core.py` | Test de validación |
| **4. MCP** | **Servidor MCP propio** (FastMCP, stdio) que expone las 4 tools. CrewAI lo consume con `mcps=[MCPServerStdio(...)]` y aplica `create_static_tool_filter` **por agente**: el protocolo mismo impone la separación de permisos | `src/mcp_server.py` | Cada agente lista solo sus tools |
| **4. Seguridad en escritura** | `guardar_informe` solo lo tiene el Redactor. La ruta la fija el código (`outputs/informes/{doc_id}_{timestamp}.md`). Rechaza la escritura si alguna cita del markdown no pertenece a un hallazgo con `estado=aprobado`. Tiene límite de tamaño | `src/tools/core.py` | Test: un informe con cita no aprobada se rechaza |
| **4. GitHub Models API** | `llm_factory.py` con `LLM_PROVIDER=deepseek\|github`. GitHub Models se usa vía endpoint compatible con OpenAI y `GITHUB_TOKEN`, con `max_rpm` en el Crew por su límite de cuota | `src/llm_factory.py` | 1 corrida completa de un contrato con GitHub Models |
| **4. Bucles infinitos y estado** | Los límites son `max_iter`, `max_execution_time` y `max_rpm` por agente, `guardrail_max_retries=2`, un **solo reintento** en el router del Flow y `max_steps` en ReAct y FC. El estado es un `Flow[EstadoRevision]` Pydantic que se vuelca a `state.json` por corrida | `src/crew/flow.py` | Traza con rechazo → reintento → corte |
| **5. LangChain** | Agente individual con `create_tool_calling_agent` + `AgentExecutor`: (a) chat de consulta con memoria de ventana, (b) baseline de agente único en el Experimento 2. Usa los mismos esquemas de tools | `src/chat/consulta.py`, `eval/baselines.py` | Comparación en README |
| **5. CrewAI: Agent, Task, Crew, Process** | Agentes con `role/goal/backstory` en `agents.yaml`, y tareas con `description/expected_output/context` en `tasks.yaml`. **Secuencial** (principal), **concurrente** (tareas del Auditor con `async_execution=True`) y **jerárquico** (`manager_llm`, como variante medida) | `src/crew/` | Ablación: secuencial vs jerárquico, latencia secuencial vs async |

---

## Diseño del crew

| Agente | Tools MCP (filtro) | Restricción | Razón de existir |
|---|---|---|---|
| **Auditor** (abogado de la contraparte) | `buscar_clausulas` | Solo lectura | Propone hallazgos por categoría con cita y atributos. Corre 3 tareas **async** (grupos de categorías) |
| **Defensor** (matizador) | `buscar_clausulas`, `leer_seccion` | Solo lectura. Sigue referencias cruzadas | Busca topes, excepciones, mutualidad y plazos de cura que anulan o atenúan cada hallazgo, para bajar los falsos positivos |
| **Verificador** | `verificar_cita`, `leer_seccion` | **Sin búsqueda**: no puede crear hallazgos | Revisa que la cita exista literalmente (código) y que respalde la afirmación (LLM). Aprueba o rechaza con motivo |
| **Redactor** | `guardar_informe` | Única escritura. Solo ve los hallazgos aprobados | Redacta el informe y la matriz de riesgo. No decide severidades |

**Lo determinístico va en código:**
- **Guardrail** en las tareas del Auditor (`guardrail=fn → (bool, Any)`, confirmado en la doc de CrewAI). Si una cita no existe literalmente, la tarea se reintenta.
- **Severidad** (`severity.py`): el LLM extrae atributos booleanos con cita (`mutua`, `tiene_tope`, `plazo_cura_dias`, `unilateral`) y una tabla de reglas por categoría calcula Alta/Media/Baja.

**Salidas Pydantic** (`output_pydantic`, en `schemas.py`):
- `Hallazgo`: categoria, afirmacion, cita, doc_id, seccion, pagina, atributos.
- `Matiz`: veredicto `se_sostiene|se_matiza|se_descarta`, contracita.
- `Verificacion`: cita_existe, cita_respalda, estado `aprobado|rechazado`, motivo.

**Categorías** (nombres CUAD, para tener ground truth): Uncapped Liability, Cap on Liability, Termination for
Convenience, Renewal Term, Exclusivity, Non-Compete, Anti-Assignment, Liquidated Damages, IP Ownership
Assignment, Governing Law.

**Orquestación con un Flow** (`Flow[EstadoRevision]`):
```
@start cargar_contrato
  → crew_revision.kickoff()   (Auditor ×3 async → Defensor → Verificador; Process.sequential)
  → @router evaluar:  rechazados y intentos < 1 → "reintentar" | si no → "informe"
@listen("reintentar") → re-lanza al Auditor solo en las categorías rechazadas, con el motivo como feedback
@listen("informe")    → severity.py → Redactor → guardar_informe → memoria.guardar(aprobados, rechazos)
```
El patrón de bucle con router no está documentado de forma explícita en la doc de Flows, así que se
valida el día 7. **Respaldo:** el mismo bucle en Python plano dentro de `run.py`, con el mismo estado
Pydantic.

**Configuración de CrewAI (para evitar fallas comunes):**
- `llm=` explícito en **cada** agente y en `manager_llm`. Por defecto CrewAI usa gpt-4o-mini.
- `Memory(llm=deepseek, embedder=fn_e5)`: el embedder por defecto es de OpenAI.
- `planning=False`. Telemetría desactivada con `CREWAI_DISABLE_TELEMETRY=true` y `OTEL_SDK_DISABLED=true`.
- Agentes con tools usan `deepseek-chat`, no `deepseek-reasoner`.
- Se instala `crewai[litellm]` y `mcp`.

---

## Estructura de archivos

```
src/
  rag/ingest.py, chunking.py, retrieve.py, verify.py
      ← funciones del notebook copiadas sin cambiar lógica (§1–§7.1), más filtro por doc_id;
        modelos y chunks como singleton
  llm_factory.py               ← DeepSeek | GitHub Models
  tools/core.py                ← funciones puras + args_schema Pydantic + validación
  mcp_server.py                ← FastMCP que expone tools/core.py
  agents_scratch/react.py      ← ReAct desde cero (regex)
  agents_scratch/function_calling.py ← FC nativo (JSON Schema)
  chat/consulta.py             ← LangChain AgentExecutor + memoria de ventana
  crew/agents.yaml, tasks.yaml ← role/goal/backstory, description/expected_output
  crew/schemas.py, severity.py, memoria.py, crew.py, flow.py
eval/
  ground_truth.json            ← CUAD: 4 contratos × 10 categorías (Sí/No + span)
  baselines.py, exp1_patrones.py, exp2_ablacion.py
  results/                     ← JSON, trazas, state.json (la demo no depende de la API)
outputs/informes/
multiagente.ipynb              ← demo + resultados (corre sin API key leyendo eval/results/)
```

`clausefinder.ipynb` queda intacto como entregable de la Ev.1. De paso quedan resueltos dos ítems de
"Próximos pasos" del README:
- El filtro por metadatos (`doc_id`).
- La expansión de referencias cruzadas, con `leer_seccion`.

---

## Evaluación

**Experimento 1: patrones de ejecución (agente único, 2 contratos, 10 categorías)**

Compara ReAct desde cero, function calling nativo y LangChain AgentExecutor. Mide errores de parsing y
acciones inválidas, número de pasos, tasa de tarea completada, tokens y latencia. Muestra con números la
fragilidad de ReAct que se vio en clase.

**Experimento 2: multiagente vs agente único (4 contratos, ground truth CUAD)**

| ID | Configuración |
|---|---|
| B0 | ClauseFinder Ev.1 (`preguntar()` por categoría, sin agentes) |
| B1 | Agente único LangChain con las mismas tools |
| C1 | Auditor + Verificador |
| C2 | Crew completo, secuencial |
| C3 | C2 + reintento en el Flow |
| C4 | C2 con `Process.hierarchical` |
| C3-mem | C3 con memoria a largo plazo, procesando los contratos en orden |

**Métricas:** recall y precisión por categoría contra CUAD, fidelidad de cita, % de afirmaciones
respaldadas (muestra revisada a mano), falsos positivos en categorías ausentes, llamadas LLM, tokens,
costo y latencia.

**Hipótesis explícitas** (se reportan aunque fallen, igual que la lectura crítica de RRF en la Ev.1):
- H1: el Defensor sube la precisión.
- H2: el Verificador lleva la fidelidad a ≈100 %.
- H3: el jerárquico cuesta más sin mejorar la calidad.
- H4: la memoria reduce los rechazos.

---

## Cronograma (≈10 días hábiles)

| Día | Trabajo | Listo cuando |
|---|---|---|
| 1 | `src/rag/*` desde el notebook + filtro `doc_id` | El top-5 es igual al del notebook en 3 preguntas |
| 2 | `tools/core.py` con schemas + `mcp_server.py` + `llm_factory.py` | Un cliente MCP lista las 4 tools y las invoca. Un argumento inválido da error estructurado |
| 3 | ReAct desde cero + FC nativo + Experimento 1 | Tabla del Exp. 1 en 2 contratos |
| 4 | Ground truth CUAD + LangChain AgentExecutor (chat + baseline) | 40 etiquetas. El chat responde un seguimiento usando la ventana |
| 5–6 | Crew: YAML, MCP con filtros, tareas async, guardrail, Pydantic, Memory local | Salida válida en 1 contrato. Cada agente ve solo sus tools |
| 7 | Flow con estado + router/reintento + severidad + Redactor + escritura segura + variante jerárquica | Informe de los 4 contratos y una traza con rechazo → reintento |
| 8 | Experimento 2 en background + 1 corrida con GitHub Models | Tabla de ablación completa |
| 9 | README §Ev.2 (matriz de cobertura, diagramas, resultados) + `multiagente.ipynb` | Notebook corre de punta a punta sin API key |
| 10 | Holgura y ensayo de demo | — |

**Recortes si hay atraso (en orden, sin perder cobertura de ningún módulo):**
1. C3-mem pasa a ser solo una demo cualitativa.
2. C4 jerárquico se queda en una corrida en un solo contrato.
3. Las corridas por configuración bajan de 3 a 1.
4. Las categorías bajan de 10 a 6.
5. El Experimento 1 se hace en 1 contrato.

---

## Riesgos

- **APIs de CrewAI que cambian entre versiones** (`Memory` unificada, `mcps=`, bucle en Flow): se
  verifica cada una contra la versión instalada el día en que se usa, y se fijan las versiones en
  `requirements.txt`. Respaldos: bucle en Python plano para el Flow, `MCPServerAdapter` para `mcps=`, y un
  índice NumPy propio con e5 para `Memory`.
- **`AgentExecutor`** está en `langchain-classic` en LangChain 1.x: se fija la versión vista en clase.
- **Conflictos de dependencias en Windows:** venv nuevo `.venv-crew`.
- **CUAD mide la presencia de la cláusula, no el riesgo para la parte:** se declara como limitación, y la
  severidad se evalúa cualitativamente en una muestra.
- **Demo:** se usan resultados cacheados en `eval/results/` y una sola corrida en vivo en el contrato más corto.

---

## Verificación end-to-end

1. `python -m src.rag.retrieve "¿cuándo puede terminarse el contrato?"`: el top-5 coincide con el notebook.
2. `python -m src.mcp_server` + un cliente de prueba: lista las tools. `verificar_cita` da `True` con una
   cita real y `False` con una alterada. Un `doc_id` inválido da error con los valores válidos.
3. `python -m eval.exp1_patrones`: tabla ReAct vs FC vs LangChain.
4. `python -m src.crew.flow --doc MSA_MeritLife_2020`: genera el informe en `outputs/informes/`, `state.json`
   y la traza. Todas las citas pasan `verificar_citas`. Un intento de escribir con una cita no aprobada se rechaza.
5. `LLM_PROVIDER=github python -m src.crew.flow --doc <contrato corto>`: misma salida con GitHub Models.
6. `python -m eval.exp2_ablacion --runs 3`: tabla B0–C4 y C3-mem.
7. `multiagente.ipynb` corre de arriba abajo sin `DEEPSEEK_API_KEY`.
