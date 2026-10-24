# Cómo se midió: ground truth, métricas y experimentos

Todo el puntaje lo calcula código (`eval/metricas.py`) contra etiquetas de abogados. Ningún LLM se
evalúa a sí mismo.

## 1. Ground truth

- **Fuente:** CUAD (The Atticus Project, CC BY 4.0): contratos reales con cláusulas anotadas por
  abogados (`data/cuad/master_clauses.csv`).
- `python -m eval.construir_ground_truth` cruza los 4 contratos con las 10 categorías del checklist:
  **40 determinaciones** (21 presentes, 19 ausentes) en `eval/ground_truth.json`.
- Por cada caso guarda si la cláusula está presente y **en qué sección de nuestro chunking** cae el
  span anotado por CUAD, para medir además si el sistema cita la sección correcta.
- **Control negativo:** *Liquidated Damages* no aparece en ningún contrato. Marcarla es inventar.

## 2. Puntaje por caso

El sistema entrega, por categoría: presente sí/no, cita literal y sección.

| | CUAD: presente | CUAD: ausente |
|---|---|---|
| **Sistema: presente** | VP | FP |
| **Sistema: ausente** | FN | VN |

Una categoría sin respuesta cuenta como "ausente" y se informa aparte (`sin_responder`).

## 3. Métricas

| Métrica | Fórmula / regla | Qué responde |
|---|---|---|
| Precisión | VP / (VP + FP) | De lo que marcó, ¿cuánto existía? |
| Recall | VP / (VP + FN) | De lo que existía, ¿cuánto encontró? |
| F1 | 2·P·R / (P + R) | Resumen de la detección |
| Fidelidad de cita | citas verificadas / determinaciones "presente" | ¿La cita existe literalmente en la sección indicada? |
| Sección correcta | solo en VP: sección citada ∈ secciones de CUAD | ¿Ubicó bien la cláusula? |
| FP control negativo | marcas en *Liquidated Damages* | ¿Inventa cláusulas? |
| Costo, tiempo, llamadas | promedio por contrato | Eficiencia (tarifas DeepSeek por token) |

**Fidelidad de cita en detalle.** Se verifica con `verificar_cita`, la misma tool del Verificador,
pero llamada por el script de evaluación (no depende de lo que diga el agente). La comparación:

- exige límites de palabra (`"either party shall assign"` no vale dentro de `"Neither party shall assign"`);
- normaliza comillas tipográficas (`contractor’s` = `contractor's`);
- aplica Unicode NFKC (el `;` de los PDF es U+037E, no el ASCII).

Una paráfrasis o una traducción **falla**. Por eso el RAG de la Ev.1 (B0) obtuvo 0,750: traducía las
citas al español.

## 4. Experimentos

- **Experimento 1** (`python -m eval.exp1_patrones`): 2 contratos × 10 categorías. ReAct desde cero,
  function calling nativo y LangChain AgentExecutor con la misma tarea, tools y modelo.
- **Experimento 2** (`python -m eval.exp2_ablacion`): 4 contratos × 10 categorías = 40 por
  configuración, 7 configuraciones:

  | Config | Qué agrega |
  |---|---|
  | B0 | RAG de la Ev.1, sin agentes |
  | B1 | Agente único LangChain con las mismas tools |
  | C1 | Auditor + Verificador |
  | C2 | + Defensor |
  | C3 | + reintento del Flow |
  | C4 | C2 en proceso jerárquico |
  | C3-mem | C3 + memoria a largo plazo |

- 1 corrida por configuración. Resultados en `eval/results/`, errores caso a caso en
  `eval/results/errores_exp2.md` (`python -m eval.analisis_errores`).

## 5. Resultados clave

| | B0 RAG | B1 agente único | C3 crew |
|---|---|---|---|
| F1 | 0,878 | 0,952 | 0,977 |
| Fidelidad de cita | 0,750 | 0,952 | 1,000 |
| Sección correcta | 0,722 | 0,900 | 0,952 |
| US$/contrato | 0,007 | 0,009 | 0,100 |

Control negativo: 0 falsos positivos en las 7 configuraciones.

## 6. Cómo leerlo (y defenderlo)

- La diferencia de F1 entre C3 y B1 es **1 determinación de 40**: con 4 contratos y una corrida, es ruido.
- Lo sólido es la **fidelidad de cita**: 1,0 en todas las variantes del crew, y además garantizada por
  construcción (el Redactor no puede guardar una cita que el código no aprobó).
- Los dos errores que quedan en C3 son discutibles frente a CUAD (*Uncapped Liability* en Sparkling
  Spring y *Non-Compete* en CoBranding): medir contra etiquetas expertas no elimina la ambigüedad del dominio.

## 7. Cómo se logró la fidelidad de cita 1,0

No es que el modelo "cite mejor": hay **cuatro filtros en serie**, y tres de ellos son código.

1. **Guardrail del Auditor** (`GuardrailCitas` en `src/crew/crew.py`). Antes de que la auditoría
   salga de la tarea, el código llama a `verificar_cita` por cada hallazgo. Si una cita no existe
   literalmente, la devuelve al Auditor con el error ("cópiala carácter por carácter del fragmento
   devuelto por la búsqueda"). Máximo 2 rechazos; después la deja pasar para no abortar la corrida.
2. **Verificador** (agente). Sin tool de búsqueda, así que no puede inventar hallazgos. Con
   `verificar_cita` y `leer_seccion` juzga si la cita existe **y si respalda lo afirmado**.
   Aprueba o rechaza con motivo.
3. **`decidir()`** (`src/crew/severity.py`). Vuelve a comprobar la cita en código, sin confiar en lo
   que transcribió el Verificador. Si la cita existe pero en otra sección, la corrige (§6 → §5) y deja
   constancia. El estado final sale de reglas: aprobado / rechazado / descartado / ausente.
4. **`guardar_informe`** (`src/tools/core.py`). La única tool de escritura **rechaza cualquier cita
   que no esté en la lista de aprobadas**. Aunque el Redactor quisiera, no puede publicar una cita
   no verificada.

Y si el Verificador rechaza, el **Flow** (`src/crew/flow.py`, `evaluar` → `reauditar`) devuelve solo
esas categorías al Auditor con el motivo, una vez. Así se recuperó la responsabilidad ilimitada del MSA.

La comparación en sí (`cita_existe` en `src/rag/verify.py`): minúsculas y espacios normalizados,
Unicode NFKC, comillas y guiones tipográficos a ASCII, y **límites de palabra** en los extremos.

**Frase:** "La fidelidad no depende de que el modelo se porte bien: el informe no puede contener una
cita que el código no haya encontrado literalmente en el contrato."

## 8. El proceso jerárquico (C4)

**Cómo se implementó** (`crew_revision` en `src/crew/crew.py`, `Config("C4", jerarquico=True)`):

- Mismos agentes y tareas que C2, pero `process=Process.hierarchical` y `manager_llm=llm` (DeepSeek).
- Se quita el dueño de cada tarea (`t.agent = None`): **el manager decide a quién delegar**.
- Las tareas del Auditor dejan de ser concurrentes (`concurrente=False`).
- Arreglo necesario: CrewAI solo carga las tools MCP del agente dueño de la tarea, y en jerárquico no
  hay dueño. La primera corrida tuvo **0 llamadas MCP y recall 0** (los auditores respondían "no tengo
  acceso a buscar_clausulas"; está en `eval/results/exp2_descartados/`). Se corrigió resolviendo las
  tools de cada agente por adelantado: `a.tools = a.get_mcp_tools(a.mcps)`.

**Resultado:** F1 0,927 (peor que C2 y C3), 513 llamadas al LLM, 427 s y US$ 0,50 por contrato:
58 veces el costo del agente único y 5 veces el de C3. Además el manager intentaba delegar en
paralelo al mismo agente y CrewAI rechazaba esas delegaciones.

**Por qué no rinde aquí:** las dependencias entre tareas son fijas (auditar → matizar → verificar).
El manager no tiene nada útil que decidir; solo agrega vueltas de coordinación. Por eso la versión
elegida es secuencial con auditores concurrentes.

**Frase:** "Lo implementé y lo medí. En un flujo con dependencias fijas, el manager solo agrega costo:
58 veces el del agente único y peor recall."

## 9. Costo y tokens: por qué sale barato

Ojo con el término: el crew **no** usa pocos tokens. Una revisión C3 de un contrato consume del orden
de **600.000 tokens de entrada y 45.000 de salida** (`eval/results/demo_crew/state.json`). Lo que es
bajo es el **costo** (US$ 0,10 por contrato), por estas razones:

1. **Caché de prefijo de DeepSeek.** En la corrida de demo, **424.000 de los 601.000 tokens de entrada
   (70 %)** salieron de caché, que se cobra 10 veces más barato (US$ 0,028 frente a 0,28 por millón).
   El crew repite mucho prefijo (instrucciones, backstory, historial de cada agente), así que el caché
   acierta seguido. Cuenta aproximada de esa corrida:

   | | Tokens | Tarifa / 1M | US$ |
   |---|---|---|---|
   | Entrada sin caché | 177.380 | 0,28 | 0,050 |
   | Entrada en caché | 423.926 | 0,028 | 0,012 |
   | Salida | 45.688 | 0,42 | 0,019 |
   | **Total** | | | **≈ 0,081** |

   Sin caché la misma corrida costaría ≈ US$ 0,19: **el caché ahorra más de la mitad**.
2. **Modelo barato.** `deepseek-chat` (DeepSeek-V3.2) a US$ 0,28 / 0,42 por millón.
3. **Retrieval local, sin costo de API.** Embeddings, BM25 y re-ranking corren en la máquina; el LLM
   recibe solo los 5 fragmentos mejor rankeados (`k` limitado a 1–8 por el esquema de la tool).
4. **Contexto acotado por diseño.** Cada agente ve solo sus tools (filtro MCP) y el checklist se
   reparte en 3 auditores especialistas, en vez de un prompt gigante con las 10 categorías.
5. **El reintento solo revisa lo rechazado.** `reauditar` vuelve a correr únicamente las categorías
   que el Verificador rechazó, no la revisión completa: C3 cuesta solo ~10 % más que C2.
6. **Caché de búsquedas en el servidor MCP** (`lru_cache` en `_recuperar`). El retrieval es
   determinístico: una consulta repetida (el Defensor suele repetir las del Auditor) no se recalcula.
   Esto ahorra tiempo (~6 s por búsqueda), no tokens.
7. **Temperatura 0** y salidas con esquema Pydantic (`output_pydantic`): menos respuestas mal formadas
   que haya que repetir.

**Comparación honesta:** el agente único cuesta US$ 0,009 por contrato; el crew C3, US$ 0,10 (unas
12 veces más). Frente a una hora de un abogado sigue siendo irrelevante, pero es un costo real.

**Frase:** "No consume pocos tokens; consume tokens baratos. El 70 % de la entrada sale del caché de
DeepSeek, el retrieval es local y el reintento solo repite lo rechazado. Así una revisión completa
cuesta diez centavos de dólar."

---

> **Frase para la defensa:** "Comparé contra etiquetas de abogados de CUAD, 40 casos por configuración.
> La detección la mido con precisión, recall y F1; la fidelidad la verifico en código: la cita tiene que
> existir literalmente en la sección indicada, así que una paráfrasis o una traducción cuenta como error.
> Incluí un control negativo y ninguna configuración lo inventó."
