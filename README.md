# ClauseFinder — RAG para revisión de contratos legales

Sistema de *Retrieval-Augmented Generation* que responde preguntas sobre cláusulas contractuales citando el texto exacto del contrato de origen.

> **Asignatura:** Ingeniería de Soluciones con Inteligencia Artificial (012V) — DUOC UC
> **Evaluación 1:** RAG (secciones 1 a 10, `clausefinder.ipynb`).
> **Evaluación 2:** **ClauseFinder Crew**, sistema multiagente de revisión adversarial construido sobre el RAG
> (sección 11, `multiagente.ipynb`).

---

## 1. Problema

Los equipos legales y de compliance dedican una cantidad desproporcionada de horas a revisar contratos extensos (NDAs, M&A, licencias) buscando manualmente cláusulas específicas —terminación, indemnización, exclusividad, jurisdicción— dispersas en cientos de páginas y sin estructura uniforme entre documentos.

Los modelos de lenguaje generales, usados **sin contexto documental**, alucinan cláusulas o citan condiciones inexistentes cuando se les pregunta sobre un contrato concreto, porque no tienen acceso al texto real del documento. Esto genera riesgo legal si un abogado junior o un analista de compliance confía en una respuesta no verificada.

### Por qué RAG y no fine-tuning

El problema no es de **conocimiento del dominio legal** (que un modelo grande ya tiene en buena medida), sino de ***grounding* en documentos específicos**: el sistema debe responder sobre *este* contrato, no sobre "los contratos en general". Esa distinción es la que justifica la arquitectura. Un fine-tuning cambiaría el estilo y el vocabulario del modelo, pero no le daría acceso al PDF que el usuario subió hace treinta segundos.

### Alcance elegido (delimitación del usuario final)

Un abogado de M&A y una startup revisando NDAs de proveedores tienen necesidades distintas. Para que el sistema sea evaluable, este proyecto se acota a:

- **Usuario final:** analista de compliance / abogado junior en una empresa mediana.
- **Tipo de documento:** NDAs y contratos comerciales de servicios (no M&A).
- **Tarea:** localizar y explicar cláusulas de un conjunto acotado de categorías, **siempre con cita literal y ubicación en el documento**.

### Criterios de éxito (medibles)

| Métrica | Definición | Meta inicial |
|---|---|---|
| Recall@10 | ¿El chunk que contiene la cláusula correcta aparece en el top-10 recuperado? | ≥ 0,85 |
| Precision@5 | Proporción de chunks relevantes en el top-5 | ≥ 0,60 |
| Fidelidad de cita | % de respuestas cuya cita literal existe textualmente en el documento | ≥ 0,98 |
| Abstención correcta | % de preguntas sin respuesta en el corpus donde el sistema responde "no encontrado" | ≥ 0,80 |

---

## 2. Datasets

Todos públicos y con licencia clara. Los dos primeros son el corpus principal del proyecto; el resto queda documentado como alternativa o extensión.

| Dataset | Contenido | Tamaño | Utilidad para RAG |
|---|---|---|---|
| **CUAD** (Contract Understanding Atticus Dataset) | Contratos comerciales reales, 41 categorías de cláusulas anotadas por expertos | +510 contratos de distintas industrias, +13.000 etiquetas expertas en 41 categorías | El estándar de facto para *contract review* con LLMs |
| **ContractNLI** | NDAs anotados para inferencia lógica (entailment / contradiction / not mentioned) | ~607 contratos, mayoritariamente acuerdos de confidencialidad | Bueno para probar razonamiento sobre obligaciones y excepciones |
| **MAUD** | Acuerdos de fusión y adquisición (M&A) | 47.457 anotaciones generadas por expertos | Nicho especializado; útil solo si el caso de uso es M&A |
| **LegalBench-RAG** | Corpus combinado (CUAD + ContractNLI + MAUD + PrivacyQA) ya preparado con pares pregunta–respuesta | Documentos `.txt` de cuatro dominios legales | El más práctico para *benchmarking* de retrieval, no solo texto crudo |
| **ACORD** | Retrieval de cláusulas específicas para redacción contractual | 114 queries con cláusulas relevantes anotadas | Diseñado para evaluar sistemas de retrieval en tareas legales |
| **LEDGAR** | Cláusulas contractuales etiquetadas por tipo | +1,8 M de cláusulas, +180.000 etiquetas | Útil para clasificación; menos para RAG conversacional |

**Selección del proyecto:** CUAD (corpus + *ground truth* de spans) y ContractNLI (NDAs, coherente con el usuario final elegido). **LegalBench-RAG** se usa como set de evaluación de retrieval porque ya trae pares pregunta–respuesta con los spans esperados, lo que evita construir anotaciones a mano.

### Corpus incluido en el repositorio

`data/raw/` trae **cuatro contratos comerciales reales en PDF**, extraídos de CUAD —que a su vez los toma de filings públicos de la SEC de Estados Unidos—. Son documentos efectivamente firmados, con toda la irregularidad de formato que eso implica, que es justamente lo que hace útil probar contra ellos:

| Archivo | Tipo | Partes | Año | Secciones detectadas |
|---|---|---|---|---|
| `MSA_MeritLife_2020.pdf` | Master Services Agreement | Merit Life Insurance | 2020 | 18 |
| `CoBranding_Healthcentral_1999.pdf` | Co-Branding Agreement | Healthcentral.com / MLX | 1999 | 30 |
| `Licencia_Mantenimiento_SparklingSpring_2002.pdf` | Licencia de software y mantenimiento | Sparkling Spring / Garman | 2002 | 20 |
| `Mantenimiento_XImage_Sagem_1999.pdf` | Contrato de mantenimiento | XIMAGE / SAGEM Morpho | 1999 | 19 |

El pipeline no está atado a estos archivos: cualquier PDF que se deje en `data/raw/` entra por el mismo flujo.

Repositorio curado con más opciones: <https://github.com/maastrichtlawtech/awesome-legal-nlp>

---

## 3. Arquitectura

```
              ┌──────────────── INGESTA (offline) ────────────────┐
PDF / TXT ──▶  parseo ──▶ segmentación  ──▶ embeddings ──▶ índice vectorial
              (pdfplumber)  por cláusula      (batch)        (NumPy / FAISS)
                                 │                                  │
                                 └──▶ índice léxico BM25 ───────────┘

              ┌──────────────── CONSULTA (online) ────────────────┐
pregunta ──▶ retrieval híbrido ──▶ fusión RRF ──▶ re-ranking ──▶ prompt
              (denso + BM25)         top-30         top-5       con citas
                                                                    │
                                                                    ▼
                                                     DeepSeek-V3.2 ──▶ respuesta
                                                                       + cita literal
                                                                       + doc / sección
```

### 3.1 De PDF a texto utilizable

Extraer el texto de un PDF es la parte fácil; lo que rompe el pipeline es lo que viene pegado a él:

- **Índice con puntos guía** (`3. Software Maintenance Services......4`): cada línea parece un encabezado de sección. Sin filtrarlas, el contrato de XIMAGE generaba 16 fragmentos basura de 82 caracteres y perdía su estructura real.
- **Encabezados y pies repetidos** en cada página (`Exhibit 10.22`, `Page 4 of 12`), que parten cláusulas por la mitad y ensucian los embeddings.

Los puntos guía se detectan por su patrón; los encabezados repetidos, porque son las líneas que aparecen en la mitad o más de las páginas. El separador de página se conserva, y es lo que después permite citar el número de página.

### 3.2 Chunking por cláusula (la decisión clave)

El *chunking* de tamaño fijo (por ejemplo 512 tokens con solape) es el default de la mayoría de los tutoriales y **es la peor opción para contratos**: parte cláusulas por la mitad y mezcla obligaciones de secciones distintas en un mismo chunk, lo que produce respuestas que suenan bien y son falsas.

Estrategia usada aquí:

1. **Segmentación estructural:** detectar encabezados de artículo (`3.`, `ARTICLE V`, `SECTION 8 —`) con expresiones regulares y cortar en esos límites.
2. **Granularidad: artículo, no subcláusula.** Cortar en cada `3.1`, `3.2` da fragmentos más precisos pero rompe el sentido: `5.2 "The indemnification obligations in Section 5.1 shall not apply..."` no significa nada sin `5.1`. Manteniendo el artículo completo, la regla y su excepción viajan juntas — que es justamente el error más común en RAG legal.
3. **Techo de ~1.000 tokens:** si un artículo lo excede, se divide por subincisos (`(a)`, `(b)`, `(i)`) y nunca por conteo de caracteres.
4. **Metadatos por chunk** (indispensables para citar y filtrar): `doc_id`, título del contrato, número y epígrafe del artículo, página, offsets de caracteres.
5. **Contexto de encabezado:** cada chunk se indexa con su ruta jerárquica prefijada (`Contrato X > 8. FORCE MAJEURE`), porque una cláusula aislada suele ser ininteligible sin su encabezado.

#### El problema real: varias numeraciones compitiendo

En un contrato de verdad, `1.`, `2.`, `3.` aparece en varios lugares a la vez: los recitales (*"Whereas: 1... 2..."*), las listas dentro de una cláusula y los artículos. Un detector ingenuo los mezcla todos y la segmentación queda inservible.

La solución es simple y robusta: la numeración verdadera es **la que cubre el documento entero**. Se prueba cada arranque posible en `1` y gana la secuencia ascendente que más secciones encadena — los recitales dan una corrida de dos o tres, los artículos dan una de veinte. Se tolera además un salto de un número, para absorber los encabezados que la extracción de PDF se come.

### 3.3 Retrieval híbrido

Los contratos están llenos de términos exactos que la búsqueda semántica pierde (`Force Majeure`, `indemnitee`, plazos como `thirty (30) days`). Por eso se combinan dos índices:

- **Denso:** similitud coseno sobre embeddings → captura paráfrasis ("¿puedo salirme del contrato?" → cláusula de terminación).
- **Léxico (BM25):** captura terminología exacta y cifras. Es monolingüe por definición: con preguntas en español sobre contratos en inglés aporta poco —solo los términos que el usuario escribe en inglés, que en jerga legal son muchos—, y aporta plenamente sobre un corpus en español.
- **Fusión:** Reciprocal Rank Fusion (RRF) sobre ambas listas.
- **Re-ranking:** cross-encoder sobre el top-30 → top-5 que entra al prompt.

### 3.4 Generación con citas obligatorias

El prompt de sistema exige tres cosas: (1) responder **solo** con lo que está en los fragmentos recuperados, (2) incluir la **cita literal** entre comillas junto con documento y sección, y (3) responder `"No encontrado en los documentos proporcionados"` cuando el contexto no alcanza.

Un validador post-generación comprueba que cada cita exista textualmente en el chunk de origen; si no coincide, la respuesta se marca como no verificada. Esta verificación es barata y es la principal defensa contra la alucinación de cláusulas.

---

## 4. Stack técnico

| Componente | Elección | Nota |
|---|---|---|
| Lenguaje | Python 3.11+ | |
| Parseo de PDF | `pdfplumber` | Conserva número de página y coordenadas, necesarios para citar |
| Índice vectorial | Matriz NumPy normalizada (búsqueda exacta) | Sin servidor ni dependencias; se cambia por FAISS o ChromaDB para decenas de miles de contratos |
| Índice léxico | `rank_bm25` | |
| Embeddings | `intfloat/multilingual-e5-small` | **Multilingüe es obligatorio**: las preguntas van en español y el corpus en inglés. Un modelo solo-inglés (`bge-small-en-v1.5`) acierta 2/4 preguntas de prueba; este acierta 4/4. Para más calidad: `BAAI/bge-m3` o `voyage-law-2` |
| Re-ranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | También multilingüe, por la misma razón |
| Generación | **DeepSeek-V3.2** (`deepseek-chat`) vía SDK `openai` apuntando a `https://api.deepseek.com` | API compatible con OpenAI; `deepseek-reasoner` para razonamiento explícito |
| Evaluación | Script propio sobre LegalBench-RAG | |

> **Nota sobre embeddings:** la API de DeepSeek no expone un endpoint de embeddings, así que el *retrieval* se resuelve con un proveedor externo o con un modelo local. Para una entrega de clase, `bge-m3` en local evita costos y rinde razonablemente; `voyage-law-2` está entrenado en dominio legal y funciona mejor sobre cláusulas.

### Modelo de generación y costo

| Modelo | ID | Contexto | Input $/1M (sin caché / caché) | Output $/1M |
|---|---|---|---|---|
| DeepSeek-V3.2 | `deepseek-chat` | 128K | $0,28 / $0,028 | $0,42 |
| DeepSeek-V3.2 (razonamiento) | `deepseek-reasoner` | 128K | $0,28 / $0,028 | $0,42 |

Precios de lista según <https://api-docs.deepseek.com/quick_start/pricing>; conviene verificarlos antes de presupuestar.

Se usa **`deepseek-chat`** para la respuesta final por costo y latencia: una consulta típica (~2.000 tokens de entrada, ~400 de salida) cuesta menos de una milésima de dólar. Para preguntas con obligaciones, excepciones y condiciones anidadas basta cambiar `MODEL_LLM` a `deepseek-reasoner`, que razona explícitamente antes de responder; el resto del código no cambia. La ventana de 128K permite además, como línea base de comparación, pasar contratos completos sin retrieval y medir si el RAG realmente aporta.

Dos observaciones sobre el costo:

- **El caché de prefijo es automático** en DeepSeek: el bloque de instrucciones de sistema, fijo entre consultas, se cobra a la tarifa de caché (10× más barata) cuando acierta, sin configurar nada.
- **`deepseek-reasoner`** genera tokens de razonamiento que también se cobran como salida; una consulta puede multiplicar por 5–10 los tokens de salida respecto a `deepseek-chat`.

---

## 5. Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .env                                  # DEEPSEEK_API_KEY / GITHUB_TOKEN (no se versiona)
├── clausefinder.ipynb                    # Ev.1: implementación completa del RAG
├── multiagente.ipynb                     # Ev.2: demostración sobre resultados guardados
├── data/
│   ├── raw/                              # corpus: 4 contratos reales en PDF
│   └── cuad/master_clauses.csv           # etiquetas expertas de CUAD (ground truth Ev.2)
├── src/
│   ├── rag/                              # Ev.1 extraída del notebook sin cambiar su lógica
│   │   ├── ingest.py  chunking.py  corpus.py  retrieve.py  verify.py  config.py
│   ├── categorias.py                     # checklist de 10 categorías de riesgo (nombres CUAD)
│   ├── esquemas.py                       # formato de salida común a todas las configuraciones
│   ├── llm_factory.py                    # core engine: DeepSeek | GitHub Models
│   ├── observabilidad.py                 # trazas en Arize Phoenix (PHOENIX_TRAZAS=1)
│   ├── tools/core.py                     # las 4 tools: funciones + JSON Schema + validación
│   ├── mcp_server.py                     # servidor MCP propio (streamable HTTP, 127.0.0.1)
│   ├── agents_scratch/                   # agente único desde cero: ReAct (regex) y function calling
│   ├── chat/                             # LangChain AgentExecutor: chat con memoria + baseline
│   └── crew/                             # CrewAI: agents.yaml, tasks.yaml, crew, Flow, memoria,
│                                         #   severidad por reglas, parche MCP
├── eval/                                 # regresión Ev.1, ground truth, experimentos 1 y 2
│   └── results/                          # resultados guardados (la demo no depende de la API)
└── tests/                                # pytest: validación de tools, escritura segura, reglas
```

El notebook de la Ev.1 se mantiene intacto. Su lógica se extrajo a `src/rag/` sin cambios, y
`python -m eval.regresion_ev1` comprueba que los módulos reproducen exactamente la tabla de la
sección 7 (88 chunks, Recall@10 0,808, MRR 0,491).

---

## 6. Instalación y uso

La implementación completa está en **`clausefinder.ipynb`**, que corre de punta a punta con tres
contratos de demostración incluidos: no necesita descargar el corpus ni credenciales para los
bloques 1 a 6. Solo la generación (bloque 7) requiere `DEEPSEEK_API_KEY`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configurar la credencial en un archivo `.env` en la raíz del proyecto (el notebook lo carga con
`python-dotenv`; el archivo está en `.gitignore` para que la clave no termine en el repositorio):

```
DEEPSEEK_API_KEY=sk-...
```

La clave se obtiene en <https://platform.deepseek.com/api_keys>.

Abrir el notebook y ejecutar de arriba hacia abajo:

```bash
jupyter notebook clausefinder.ipynb
```

Para usar otros contratos basta con dejar los PDF en `data/raw/` y volver a ejecutar desde el bloque 1:
el resto del pipeline no cambia.

### Evaluación 2 (sistema multiagente)

> **Para ejecutarlo en vivo desde CMD** (consultas, crew, agente único y trazas en Phoenix), seguir
> [`GUIA_EJECUCION.md`](GUIA_EJECUCION.md): comandos probados en `cmd.exe`, con tiempos y costos medidos.

Requiere Python 3.10 a 3.13 (CrewAI no soporta 3.14). En Windows, si `pip` no está en el PATH, se usa
`python -m pip`.

```bash
python -m venv .venv-crew
.venv-crew\Scripts\python -m pip install -r requirements.txt
```

Comandos principales (todos desde la raíz del proyecto):

```bash
python -m pytest tests -q                                   # tools, escritura segura y reglas (sin LLM)
python -m src.mcp_server                                    # servidor MCP en http://localhost:8765/mcp
python -m tests.probar_mcp                                  # cliente MCP real contra el servidor
python -m src.crew.flow --doc MSA_MeritLife_2020 --config C3   # crew completo + informe
python -m src.chat.consulta --doc MSA_MeritLife_2020        # chat de consulta con memoria
python -m eval.exp1_patrones                                # Experimento 1
python -m eval.exp2_ablacion                                # Experimento 2
jupyter notebook multiagente.ipynb                          # demo sobre resultados guardados
python -m src.observabilidad                                # Phoenix (trazas) en http://localhost:6006
```

Para ver el trabajo de los agentes en Phoenix (sección 11.7) se instala el servidor en su propio venv y
se corre cualquier comando con `PHOENIX_TRAZAS=1`:

```bash
python -m venv .venv-phoenix
.venv-phoenix\Scripts\python -m pip install -r requirements-phoenix.txt
```

```powershell
$env:PHOENIX_TRAZAS = "1"; .venv-crew\Scripts\python -m src.crew.flow --doc MSA_MeritLife_2020 --config C3
```

`src.crew.flow` levanta el servidor MCP si no está corriendo y lo detiene al terminar. Si se van a
hacer varias corridas, conviene dejarlo levantado en otra terminal, porque carga los modelos una sola
vez (~50 s). Para usar GitHub Models en vez de DeepSeek se agrega `GITHUB_TOKEN` (un PAT con permiso
`models:read`) al `.env` y se define `LLM_PROVIDER=github`.

---

## 7. Evaluación

Se mide **retrieval y generación por separado**, porque un fallo en cada etapa se corrige de forma distinta. Sin líneas base no hay forma de defender que la arquitectura aporta algo: el punto es mostrar *cuánto* suma cada componente, no que el sistema "funciona".

### Resultados medidos

26 preguntas en español, anotadas con el contrato y el artículo correctos, sobre los cuatro contratos reales del corpus:

| Estrategia | Recall@10 | MRR |
|---|---|---|
| Solo BM25 (léxico) | 0,385 | 0,107 |
| Solo denso (embeddings) | 0,692 | 0,458 |
| Híbrido (RRF) | 0,577 | 0,241 |
| **Híbrido + re-ranking** | **0,808** | **0,491** |

| k | Recall@k |
|---|---|
| 1 | 0,385 |
| 3 | 0,500 |
| 5 | 0,654 |
| 10 | 0,808 |

### Lectura crítica de estos números

**La fusión RRF por sí sola empeora el resultado** (0,577 contra 0,692 del índice denso solo). No es un error de implementación: las preguntas van en español y los contratos están en inglés, así que BM25 acierta poco y la fusión le da la mitad del peso igual. El re-ranking rescata la combinación —0,808, el mejor resultado— porque reordena leyendo cada par pregunta-fragmento y descarta lo que BM25 metió por ruido.

La conclusión no es "el híbrido es mejor", que es lo que repiten los tutoriales, sino que **un componente solo aporta si su supuesto se cumple**. Sobre un corpus en español, o con preguntas en inglés, BM25 aportaría y la fusión ganaría. Aquí no, y solo se sabe midiendo.

El Recall@1 de 0,385 frente al Recall@10 de 0,808 indica además que el sistema **casi siempre encuentra el artículo, pero no siempre lo pone primero**. Para este caso de uso es aceptable: los cinco fragmentos van igual al prompt y el modelo elige. Sería un problema si se mostrara un único resultado al usuario.

**Contra las metas declaradas en la sección 1:** el Recall@10 de 0,808 queda por debajo de la meta de 0,85. El notebook incluye un bloque de análisis de errores que lista las preguntas fallidas para atacar esa brecha.

### Lo que falta medir

- Generación: fidelidad de cita sobre una muestra mayor y revisión manual (el notebook trae el validador y la corrida de muestra; requiere `DEEPSEEK_API_KEY`).
- Línea base de *long-context*: pasar el contrato completo al modelo sin retrieval y comparar.
- Corrida sobre LegalBench-RAG completo, para cifras defendibles fuera de un set de 26 preguntas escrito por quien construyó el sistema.

---

## 8. Limitaciones y riesgos conocidos

Estos puntos se declaran a propósito, porque son las objeciones previsibles al proyecto:

1. **No es un problema nuevo.** CUAD, ACORD y LegalBench-RAG existen precisamente para atacarlo. El ángulo diferencial de este trabajo es el *chunking* por estructura de cláusula más la verificación automática de citas, no el retrieval en sí.
2. **RAG reduce la alucinación, no la elimina.** Los sistemas RAG legales publicados siguen teniendo tasas de error no triviales en cláusulas complejas (referencias cruzadas, excepciones anidadas, definiciones que viven en otra sección). Este proyecto no promete cero errores.
3. **Las referencias cruzadas rompen el supuesto del chunk autocontenido** ("sujeto a lo dispuesto en la Sección 12.3"). Mitigación parcial: expandir el contexto al chunk referenciado cuando se detecta la referencia.
4. **El corpus está en inglés.** CUAD y ContractNLI son contratos bajo derecho anglosajón; las conclusiones no se transfieren directamente a contratos chilenos. Extender a español es la continuación natural del trabajo y probablemente el ángulo más defendible.
5. **No es asesoría legal.** El sistema localiza y explica cláusulas; toda salida requiere revisión humana.

---

## 9. Próximos pasos

- [ ] Corpus en español (contratos tipo bajo legislación chilena) y evaluación específica.
- [x] Expansión de referencias cruzadas entre cláusulas: la hace el Defensor de la Ev.2 con `leer_seccion`
  (no es automática; el agente decide cuándo seguir la referencia).
- [ ] Comparación de costo y calidad frente a *long-context* sin retrieval.
- [x] Filtro por metadatos previo al retrieval: `doc_id` en `src/rag/retrieve.py` (Ev.2).

---

## 10. Referencias

- CUAD — <https://www.atticusprojectai.org/cuad>
- ContractNLI — <https://stanfordnlp.github.io/contract-nli/>
- LegalBench-RAG — <https://github.com/zeroentropy-ai/legalbenchrag>
- ACORD — dataset de retrieval de cláusulas para redacción contractual
- Awesome Legal NLP — <https://github.com/maastrichtlawtech/awesome-legal-nlp>
- Documentación de la API de DeepSeek — <https://api-docs.deepseek.com>
- CrewAI — <https://docs.crewai.com> · Model Context Protocol — <https://modelcontextprotocol.io>

---

## 11. Evaluación 2 — ClauseFinder Crew (sistema multiagente)

### 11.1 Qué cambia respecto de la Ev.1

ClauseFinder responde **una pregunta** con un flujo lineal: recupera fragmentos, genera una respuesta y
termina. La Ev.2 resuelve una tarea más grande, que un flujo lineal no cubre bien:

> **Caso de uso.** Un analista de compliance recibe un contrato del proveedor y necesita un **informe de
> riesgos**: qué cláusulas desfavorables tiene, con cita literal, matiz y severidad. Después hace preguntas
> de seguimiento sobre ese informe.

Revisar un contrato completo contra un checklist de 10 categorías exige decidir qué buscar, leer, contrastar
y corregir. Para eso se usa un **ciclo de agente**: percepción (la observación que devuelve una tool) →
razonamiento (el modelo evalúa qué sabe y qué le falta) → decisión (qué tool llamar o qué veredicto dar) →
acción (la llamada o la escritura). El ciclo se repite hasta cumplir la meta o hasta un tope. En
`multiagente.ipynb` §3 se ve el ciclo en una traza real.

### 11.2 Por qué varios agentes y no un pipeline disfrazado

Un "crew" donde cada agente es solo otro prompt en una cadena lineal no justifica ser multiagente. Aquí
cada agente tiene **herramientas y permisos distintos**, y hay un rol que **rechaza y devuelve** trabajo:

| Agente | Tools (filtro MCP) | Restricción | Razón de existir |
|---|---|---|---|
| **Auditor** ×3 (abogado de la contraparte) | `buscar_clausulas` | solo lectura | propone hallazgos por categoría; tres especialistas (responsabilidad, vigencia, competencia) trabajan en paralelo |
| **Defensor** (abogado de nuestro cliente) | `buscar_clausulas`, `leer_seccion` | solo lectura | busca topes, excepciones, reciprocidad o referencias cruzadas que anulen o atenúen cada hallazgo |
| **Verificador** | `verificar_cita`, `leer_seccion` | **sin búsqueda**: no puede inventar hallazgos | comprueba que cada cita exista y respalde lo afirmado; aprueba o rechaza con motivo |
| **Redactor** | `guardar_informe` | única escritura | redacta el informe; la tool rechaza cualquier cita que no esté aprobada |

**Lo determinístico va en código, no en el LLM** (`src/crew/severity.py`):
- La existencia literal de cada cita se vuelve a comprobar en código; no se confía en lo que transcribe el Verificador.
- El estado final (aprobado / rechazado / descartado / ausente) sale de reglas explícitas.
- La severidad sale de una tabla de política de riesgo: base por categoría, −1 si es mutua y −1 si el
  Defensor la matizó con una contracita verificada.
- Los agentes aportan hechos y juicios acotados (hay cita, la cita respalda, el matiz se sostiene), pero no deciden el resultado.

```
                         ┌───────────── Servidor MCP (127.0.0.1:8765) ─────────────┐
                         │ buscar_clausulas · leer_seccion · verificar_cita ·      │
                         │ guardar_informe   (src/tools/core.py, RAG de la Ev.1)   │
                         └───▲──────────▲─────────────▲───────────────▲────────────┘
                 filtro:     │ buscar   │ buscar+leer │ verificar+leer│ guardar
                         ┌───┴────┐ ┌───┴────┐   ┌────┴─────┐   ┌─────┴────┐
 Flow[EstadoRevision] ─► │Auditor │ │Defensor│ ─►│Verificador│─► │ Redactor │
  @start revisar         │  ×3 ∥  │─►        │   └────┬─────┘   └──────────┘
                         └────────┘ └────────┘        │ reglas en código: decidir() + severidad()
  @router evaluar ◄────────────────────────────────────┘
     ├─ rechazados y quedan intentos ─► @listen reauditar (solo esas categorías, con el motivo)
     └─ listo ─► registrar citas aprobadas ─► Redactor ─► memoria a largo plazo ─► state.json
```

### 11.3 Contenido de la unidad → dónde está implementado

| Módulo del curso | Implementación | Archivo | Evidencia |
|---|---|---|---|
| LLM vs agente; ciclo percepción → razonamiento → decisión → acción | Agentes con ciclo de tools frente al RAG lineal de la Ev.1 | `src/agents_scratch/comun.py` (`Paso`) | traza en `multiagente.ipynb` §3 |
| Componentes: core engine, memoria, tools, planificación | DeepSeek vía fábrica; memoria corta y larga; servidor MCP; Flow + tareas | `src/llm_factory.py`, `src/crew/` | esta sección |
| **ReAct desde cero** | Thought / Action / Action Input / Observation con regex, `stop=["Observation:"]`, corte por pasos y por repetición | `src/agents_scratch/react.py` | Experimento 1 |
| **Function calling nativo** | `tools=[…]` con JSON Schema generado desde Pydantic; llamadas en paralelo; sin regex | `src/agents_scratch/function_calling.py` | Experimento 1 |
| **Memoria a corto plazo** | Ventana de los últimos *k* turnos en el chat de consulta; `context=[…]` entre tareas del crew | `src/chat/consulta.py` | `eval/results/demo_memoria_corta.json` |
| **Memoria a largo plazo** | `Memory` de CrewAI (LanceDB) con embedder e5 local y LLM DeepSeek, **sin OpenAI**; guarda los rechazos del Verificador y el Auditor los recupera por similitud en el contrato siguiente | `src/crew/memoria.py` | configuración C3-mem |
| Descomposición y selección de tools de lectura/escritura | Checklist repartido en 3 tareas; tools de lectura separadas de la única de escritura | `src/crew/crew.py` | — |
| **JSON Schema + validación de argumentos** | Un esquema Pydantic por tool: `doc_id` como `enum`, sección existente, largos máximos; un error vuelve como dato con los valores válidos | `src/tools/core.py` | `tests/test_tools.py` |
| **MCP** | Servidor propio (FastMCP, streamable HTTP) con `ToolAnnotations` de solo lectura / escritura; CrewAI lo consume con `MCPServerHTTP` y `create_static_tool_filter` **por agente** | `src/mcp_server.py` | `tests/probar_mcp.py` |
| **Seguridad en la escritura** | Solo el Redactor tiene `guardar_informe`; la ruta la fija el código; rechaza citas no aprobadas, informes sin citas y textos demasiado largos; aprobar citas no es una tool | `src/tools/core.py` | `tests/test_tools.py` |
| **GitHub Models API** | `LLM_PROVIDER=github`: endpoint compatible con OpenAI, `GITHUB_TOKEN`, `max_rpm` por la cuota | `src/llm_factory.py` | ver 11.6 |
| **Bucles infinitos y estado** | `max_iter`, `max_execution_time`, `max_rpm`, guardrail con tope de rechazos, un solo reintento en el router, `max_pasos` y corte por repetición en ReAct/FC; estado Pydantic volcado a `state.json` | `src/crew/flow.py` | `eval/results/demo_crew/state.json` |
| **Observabilidad del ciclo y control** | Trazas OpenTelemetry/OpenInference en Arize Phoenix (local): árbol Flow → crew → tarea → agente → LLM / tool, con prompts, tokens, costo y latencia | `src/observabilidad.py` | sección 11.7 |
| **LangChain** | `create_tool_calling_agent` + `AgentExecutor`: chat de consulta y baseline de agente único | `src/chat/` | Experimentos 1 y 2 |
| **CrewAI: Agent, Task, Crew, Process** | `role/goal/backstory` en `agents.yaml`; `description/expected_output` en `tasks.yaml`; proceso **secuencial**, **concurrente** (`async_execution`) y **jerárquico** (`manager_llm`, variante C4) | `src/crew/` | Experimento 2 |

**LangChain vs CrewAI, cada uno donde rinde.** El chat de consulta es un agente individual conversacional:
`AgentExecutor` le da el ciclo de tools, el manejo de errores de parsing y el historial (`chat_history`). La
revisión es trabajo de un equipo con roles, dependencias entre tareas y paralelismo, que es justo lo que
modela CrewAI (Agent / Task / Crew / Process). Además, ambos frameworks usan exactamente las mismas tools y
los mismos esquemas de `src/tools/core.py`.

### 11.4 Problemas encontrados al implementar (y cómo se resolvieron)

Se documentan porque son el tipo de "desafío operacional" que la unidad anticipa:

1. **MCP por stdio no sirve con CrewAI 1.15.** CrewAI abre un cliente y un transporte nuevos en cada llamada a
   una tool MCP. Con stdio eso relanza el proceso en cada búsqueda, y con él la carga de modelos (~50 s),
   contra un timeout de 30 s. Se resolvió con un **servidor MCP de larga vida por HTTP local**, que carga los
   índices una sola vez.
2. **Nombres de función inválidos.** CrewAI antepone el host al nombre de la tool (`127_0_0_1_8765_mcp_…`), y la
   API de OpenAI/DeepSeek exige que el nombre empiece con letra. Los clientes entran por `localhost`.
3. **Bug de CrewAI en `MCPNativeTool._run`.** Toda llamada MCP desde un hilo sin event loop fallaba con
   "no running event loop", aunque el servidor había respondido bien. El `asyncio.run` se ejecuta dentro de un
   `except RuntimeError`, y la excepción encadenada se reinterpreta como fallo del transporte. Se corrigió con un
   parche mínimo y documentado (`src/crew/parches.py`) y quedó verificado en `tests/probar_mcp.py`.
4. **Una instancia de `Agent` no ejecuta dos tareas a la vez** ("Executor is already running"). Por eso hay tres
   auditores especialistas en vez de uno con tres tareas concurrentes.
5. **El guardrail debe ser una función o un método**, porque CrewAI hace `inspect.getsource()` sobre él. Si agota
   sus reintentos, CrewAI aborta la corrida entera. Por eso el guardrail rechaza como máximo 2 veces y después deja
   pasar el resultado: el Verificador atrapa lo que siga mal y el Flow lo devuelve.
6. **La verificación de citas de la Ev.1 aceptaba subcadenas.** `"either party shall assign"` "existía" dentro de
   `"Neither party shall assign"`, con el sentido invertido. Ahora se exigen límites de palabra y se normalizan
   comillas tipográficas (`contractor’s` = `contractor's`).
7. **El Flow rechaza bucles triviales.** CrewAI no acepta un listener que escuche su propia etiqueta. El ciclo
   `evaluar → reauditar → evaluar` usa etiquetas distintas y un tope de intentos en el estado.
8. **El Verificador destapó un error del chunking de la Ev.1.** En el MSA, el Auditor citó correctamente la
   cláusula de terminación como "§6", la numeración impresa del contrato, pero esa sección no existía en el
   sistema. El PDF pega el final de una página con el inicio de la siguiente en una sola línea
   (`'Page 6 of 10\f6. TERMINATION.'`), así que el encabezado no queda al comienzo de línea y el artículo 6
   quedó dentro del chunk §5. El Verificador lo diagnosticó en su motivo ("el artículo 6 está embebido al final
   de la sección 5").
   - **Lo que se cambió:** `verificar_cita` ya no rechaza una sección desconocida, sino que dice dónde está la
     cita. `decidir()` corrige la sección de forma determinística y deja constancia ("§6 → §5"). El feedback del
     reintento incluye el diagnóstico del Verificador.
   - **Lo que no se cambió:** el chunking de la Ev.1, porque cambiaría la regresión y el ground truth a mitad de
     la evaluación. Queda como próximo paso: tratar `\f` como salto de línea antes de buscar encabezados.
9. **Búsquedas encoladas.** Una búsqueda con re-ranking en CPU tarda ~6 s y el servidor las atiende en serie. Con
   tres auditores pidiendo varias a la vez, la cola superaba el timeout de 30 s del cliente MCP de CrewAI. Se
   resolvió con una caché de búsquedas en el servidor (el retrieval es determinístico) y un timeout por llamada
   de 180 s.
10. **En el proceso jerárquico, los agentes delegados se quedaban sin herramientas.** CrewAI resuelve `mcps=` solo
    para el agente dueño de la tarea, y en modo jerárquico las tareas no tienen dueño porque las reparte el
    manager. La primera corrida de C4 hizo 0 llamadas MCP, los auditores respondieron "No tengo acceso a
    `buscar_clausulas`" y el recall fue 0 (`eval/results/exp2_descartados/`). Se corrigió resolviendo las tools
    MCP de cada agente por adelantado (`agent.get_mcp_tools()`).
11. **La memoria a largo plazo no funcionaba y no avisaba.** El embedder devolvía un `ndarray` y CrewAI evalúa
    `if not resultado`, así que `recall` fallaba. Además, `remember_many` escribe en un pool de hilos y, sin
    `drain_writes()`, el proceso terminaba antes de persistir. La primera corrida de C3-mem equivalía a C3 sin
    memoria y se descartó. Se agregó `tests/test_memoria.py` y `eval/demo_memoria_larga.py`, que guarda en un
    proceso y recupera en otro.
12. **El punto y coma de los PDF no es un punto y coma.** Los 4 contratos traen U+037E (GREEK QUESTION MARK),
    idéntico a la vista, en lugar de `;`. Un LLM siempre escribe el ASCII, así que **ninguna cita con punto y
    coma podía verificarse**. Esto afectaba al verificador de la Ev.1 y al de la Ev.2. El síntoma fue un falso
    negativo persistente: la exclusividad de CoBranding se rechazaba aunque la cita era correcta, y el reintento
    no podía arreglarlo. Se corrigió aplicando la normalización Unicode NFKC en `normalizar_cita`. Al volver a
    puntuar las 34 corridas guardadas (`eval/repuntuar.py`) ninguna métrica cambió, porque las citas entregadas
    ya esquivaban el punto y coma. El efecto estaba en las decisiones del crew en tiempo de ejecución, así que
    se repitieron C3 y C3-mem.

### 11.5 Evaluación

**Ground truth.** Los 4 contratos vienen de CUAD, que publica etiquetas de abogados por categoría
(`master_clauses.csv`, CC BY 4.0). `python -m eval.construir_ground_truth` arma 40 etiquetas (4 contratos × 10
categorías: 21 presentes y 19 ausentes) y ubica cada span de CUAD en nuestras secciones. *Liquidated Damages* está
ausente en los cuatro contratos y sirve de **control negativo**.

**Experimento 1: patrones de ejecución de un agente único.** Misma tarea, mismas tools y mismo LLM:

2 contratos (MSA y CoBranding) × 10 categorías, 1 corrida por patrón (`python -m eval.exp1_patrones`):

| Patrón | Completa | Llamadas LLM | Tools | Err. formato | Err. tool | Seg. | US$/contrato | F1 | Fidelidad cita |
|---|---|---|---|---|---|---|---|---|---|
| ReAct desde cero (regex) | 2/2 | 13,0 | 12,0 | 0 | 0 | 105 | 0,0081 | 1,0 | 1,0 |
| Function calling nativo | 2/2 | 5,5 | 20,5 | 0 | 0 | 77 | 0,0073 | 1,0 | 1,0 |
| LangChain AgentExecutor | 2/2 | 5,0 | 20,5 | 0 | 0 | 53 | 0,0092 | 1,0 | 1,0 |

**Lectura.**
- **No hubo errores de formato, ni siquiera en ReAct.** Con DeepSeek-V3.2 los tres patrones resuelven la tarea sin
  errores. La fragilidad de ReAct vista en clase (salirse del formato, JSON inválido en `Action Input`) no
  apareció: el patrón la tiene, pero un modelo grande la enmascara. El agente la cuenta (`errores_formato`) y
  tiene la recuperación implementada, por si se usa un modelo más chico.
- **La diferencia medible es de eficiencia.** ReAct da una vuelta del ciclo por cada herramienta (13 llamadas al
  LLM). Function calling pide varias herramientas en paralelo en una misma respuesta (5,5 llamadas) y termina
  un 27 % antes.
- **El costo es casi igual.** El caché de prefijo de DeepSeek abarata el historial que ReAct repite en cada vuelta.

**Experimento 2: ¿aporta el multiagente?** B0 = RAG de la Ev.1 sin agentes; B1 = un agente LangChain con las
mismas tools; C1 = Auditor + Verificador; C2 = + Defensor; C3 = C2 + reintento; C4 = C2 jerárquico;
C3-mem = C3 con memoria a largo plazo.

4 contratos × 10 categorías = 40 determinaciones por configuración, 1 corrida (`python -m eval.exp2_ablacion`).
Costo y tiempo son promedios por contrato; el multiplicador es respecto del agente único B1.

| Config | Precisión | Recall | F1 | Fidelidad cita | Sección correcta | Llamadas LLM | Seg. | US$/contrato | × costo B1 |
|---|---|---|---|---|---|---|---|---|---|
| B0 RAG Ev.1 (sin agentes) | 0,900 | 0,857 | 0,878 | 0,750 | 0,722 | 10 | 95 | 0,0072 | 0,8 |
| B1 agente único (LangChain) | 0,952 | 0,952 | 0,952 | 0,952 | 0,900 | 6 | 67 | 0,0085 | 1 |
| C1 Auditor + Verificador | 0,913 | **1,000** | 0,955 | **1,000** | 0,952 | 49 | 88 | 0,051 | 6,0 |
| C2 + Defensor | 0,952 | 0,952 | 0,952 | **1,000** | 0,950 | 92 | 85 | 0,091 | 10,7 |
| **C3 + reintento (Flow)** | **0,955** | **1,000** | **0,977** | **1,000** | 0,952 | 100 | 154 | 0,100 | 11,8 |
| C4 jerárquico | 0,950 | 0,905 | 0,927 | **1,000** | 0,947 | 513 | 427 | 0,496 | 58,4 |
| C3-mem + memoria larga | 0,955 | 1,000 | 0,977 | 1,000 | 0,952 | 83 | 99 | 0,083 | 9,8 |

Ninguna configuración marcó el control negativo (*Liquidated Damages*): 0 falsos positivos en las 7. El detalle
de cada error está en `eval/results/errores_exp2.md` (`python -m eval.analisis_errores`).

**Hipótesis.**

- **H1, el Defensor sube la precisión: confirmada a medias.** De C1 a C2 la precisión pasa de 0,913 a 0,952. El
  Defensor eliminó un falso positivo real: en Sparkling Spring el Auditor tomó una *licencia* por *cesión de
  propiedad intelectual*. Pero también descartó la responsabilidad ilimitada del MSA, con un argumento legal
  plausible (la excepción al tope remite al párrafo 9, que no crea obligaciones de daños) que CUAD no comparte.
  Por eso el recall baja de 1,0 a 0,952 y el F1 no cambia.
- **H2, el Verificador lleva la fidelidad a ≈100 %: confirmada.** Todas las variantes del crew tienen fidelidad 1,0,
  frente a 0,952 del agente único y 0,750 del RAG de la Ev.1. Este último **traduce las citas al español**
  ("Este Acuerdo se regirá…"), que es justo lo que su prompt prohíbe.
- **H3, el jerárquico cuesta más sin mejorar: confirmada, y por amplio margen.** Cuesta 58 veces lo del agente
  único y 5 veces lo de C3, es 6 veces más lento y obtiene peor recall. Además mostró dos fallas operativas:
  - los agentes delegados se quedaban sin herramientas (11.4, punto 10);
  - el manager delegaba en paralelo al mismo agente, y CrewAI rechazaba esas delegaciones.
- **H4, la memoria reduce los rechazos: no evaluable con estos datos.** Con el pipeline corregido, el equipo no
  dejó rechazos ni descartes finales de los que aprender, así que C3-mem se comporta igual que C3 (las
  diferencias de costo son ruido). El mecanismo sí está demostrado de punta a punta en
  `eval/results/demo_memoria_larga.json`: una lección guardada en un proceso la recupera el Auditor del contrato
  siguiente en otro proceso. Esa demo dejó ver además un riesgo: la lección recuperada era el descarte del MSA
  que CUAD contradice. **Una memoria de lecciones también puede propagar un criterio discutible.**
- **El reintento (C3 frente a C2) es lo que más aporta.** En el MSA el Verificador rechazó la responsabilidad
  ilimitada en la ronda 0. El Flow la devolvió al Auditor con el motivo, y en la ronda 1 se aprobó con la cita
  correcta. Eso recuperó el verdadero positivo que C2 perdía (recall de 0,952 a 1,0), con un costo apenas 10 %
  mayor que C2.

**Memoria a corto plazo.** En la misma conversación de tres turnos (`eval/demo_memoria_corta.json`):
- **Con ventana k=4:** resuelve "¿esa cláusula tiene alguna excepción?" (§7(a), la excepción del párrafo 9).
  En el tercer turno ("¿qué dice la sección que mencionaste?") responde sin llamar a ninguna tool, solo con
  el historial.
- **Sin memoria (k=0):** pide aclaración en los dos turnos de seguimiento.

**Lectura crítica.**

- **El mejor resultado (C3, F1 0,977) le gana al agente único (0,952) por una sola determinación de 40.** Con
  4 contratos y una corrida, esa diferencia está dentro del ruido. Las diferencias sólidas son otras:
  - **fidelidad de cita:** 1,0 frente a 0,952;
  - **sección correcta:** 0,952 frente a 0,900;
  - **una garantía estructural:** el informe solo puede citar texto verificado en código.
- **Los dos errores que quedan en C3 son discutibles frente a CUAD.**
  - *Uncapped Liability* en Sparkling Spring: el contrato dice que el tope "shall not apply to breaches of
    Sections 13 and 14". **Las 7 configuraciones** lo marcan como presente y CUAD no.
  - *Non-Compete* en CoBranding: se cita §9, la cláusula que impide a HCI trabajar con otros portales
    canadienses, mientras CUAD anota §4 y §7.

  Medir contra etiquetas expertas no elimina la ambigüedad del dominio.
- **El RAG de la Ev.1 (B0) es el más débil.** F1 0,878, citas traducidas y sección correcta en solo el 72 % de los
  casos. Además, su forma de decidir "presente" (no respondió "No encontrado") cuenta un "No, el contrato dice lo
  contrario" como positivo. Pasar de un flujo lineal a un agente que busca, lee y verifica es el salto más grande
  de la tabla, y ocurre **antes** de agregar más agentes.
- **Qué aporta el multiagente y qué no.** Aporta verificación garantizada, corrección determinística de secciones,
  separación de permisos y un ciclo de rechazo y devolución que recupera hallazgos. No aporta mejor detección
  que un agente único capaz en contratos de esta dificultad. A US$ 0,10 por contrato (C3) el costo es irrelevante
  frente a una hora de un abogado. Pero si el uso tolera citas imperfectas, el agente único es la opción eficiente.
- **La verificación rigurosa encontró errores que el resto del sistema escondía.** Tres de los problemas de 11.4
  vienen de la Ev.1 y se descubrieron porque el Verificador se negó a aprobar algo:
  - la sección §6 que el chunking perdió;
  - el punto y coma U+037E;
  - la comparación por subcadena.

### 11.6 Limitaciones

- **CUAD mide presencia de la cláusula, no "riesgo para nuestra parte".** La detección se evalúa contra CUAD; la
  severidad se evalúa solo de forma cualitativa.
- **4 contratos son pocos y hubo una sola corrida por configuración** (el plan preveía 3; se recortó por tiempo,
  según el orden de recortes acordado). Las diferencias de 1 o 2 categorías entre configuraciones están dentro
  del ruido. Los scripts aceptan `--runs 3` y reanudan desde donde quedaron.
- **La memoria aprende solo del estado final.** Un rechazo que el reintento corrige en la misma revisión (como la
  responsabilidad ilimitada del MSA) no queda guardado como lección. Guardar también el historial de rondas es
  el siguiente paso natural.
- **GitHub Models** está implementado y enrutado (`llm_factory.py`), pero no se corrió en este equipo porque
  requiere un `GITHUB_TOKEN` propio. La cuota gratuita es baja, del orden de 10 peticiones por minuto, y el crew
  la respeta con `max_rpm`.
- **El parche a CrewAI** (`src/crew/parches.py`) depende de la versión fijada en `requirements.txt`. Al actualizar
  CrewAI hay que correr `python -m tests.probar_mcp` para saber si sigue siendo necesario.
- Como en la Ev.1, **no es asesoría legal**: el informe localiza y explica cláusulas, y requiere revisión humana.

### 11.7 Observabilidad: ver trabajar a los agentes con Arize Phoenix

Los logs `verbose` de CrewAI muestran todo, pero entremezclado. Phoenix organiza cada corrida como un
**árbol de trazas** (OpenTelemetry + OpenInference) que se puede recorrer en el navegador:

```
revisión MSA_MeritLife_2020 · C3                      chain   ← span raíz (src/crew/flow.py)
└─ Flow → Crew
   ├─ Auditor — especialista en responsabilidad       agent
   │  ├─ ChatCompletion (deepseek)                    llm    prompt, respuesta, tokens, costo
   │  └─ localhost_8765_mcp_buscar_clausulas          tool   argumentos y fragmentos devueltos
   ├─ Auditor — vigencia / competencia                agent  (en paralelo)
   ├─ Defensor → leer_seccion, buscar_clausulas
   ├─ Verificador → verificar_cita
   └─ Crew (informe) → Redactor → guardar_informe
```

- **Qué se instrumenta.**
  - CrewAI: Flow, crews, tareas, agentes y tools.
  - El SDK `openai`: todas las llamadas al LLM.
  - Spans propios: los agentes ReAct y function calling, el chat y el baseline LangChain abren cada uno un
    span AGENT, y cada llamada a una tool es un span TOOL. Así cada corrida o cada pregunta queda en un solo
    árbol.

  La instrumentación específica de LangChain se probó y se descartó. Registra sus spans por callbacks sin
  dejarlos como contexto activo, así que cada llamada al LLM quedaba duplicada en una traza huérfana, con el
  costo contado dos veces.

  Implementación: `src/observabilidad.py`, con un único punto de activación en `src/llm_factory.py`.
- **Desde Jupyter.** `consultas_phoenix.ipynb` hace consultas en vivo:
  - un chat de varias preguntas, que aparece como **una sesión** en la pestaña *Sessions*;
  - el agente único;
  - el crew completo.

  Requiere el kernel **"Python (ClauseFinder Crew)"**, que es el Python de `.venv-crew` y se registra con
  `.venv-crew\Scripts\python -m ipykernel install --user --name clausefinder-crew --display-name "Python (ClauseFinder Crew)"`.
  Las trazas se activan con `activar()` antes de crear los agentes, y `enviar_trazas()` las manda de inmediato.
- **Se enciende con una variable de entorno.** Con `PHOENIX_TRAZAS=1` se trazan las corridas; sin ella, los spans
  propios usan el tracer vacío de OpenTelemetry y no cuestan nada (los tests corren así).
- **Privacidad.**
  - Phoenix escucha solo en `127.0.0.1` y guarda en `data/phoenix/` (fuera del control de versiones), porque
    las trazas contienen texto de los contratos y los prompts.
  - Para que funcione se quitó `OTEL_SDK_DISABLED`, que apagaba todo OpenTelemetry. La telemetría propia de
    CrewAI sigue cortada con `CREWAI_DISABLE_TELEMETRY` y `CREWAI_DISABLE_TRACKING`, y esta versión de CrewAI
    no instala un proveedor global de trazas.
- **Costo en Phoenix.** Phoenix no trae las tarifas de la API de DeepSeek. Se agregan una vez en *Settings →
  Models*:
  - Nombre `deepseek-chat`, con patrón `^deepseek-(chat|flash)$`.
  - Tarifas por millón de tokens: entrada US$ 0,28, `cache_read` 0,028 y salida 0,42.

  El patrón incluye `flash` porque las trazas revelaron que **la API responde con `deepseek-flash` aunque se pida
  `deepseek-chat`**: DeepSeek reasignó el alias. Conviene confirmar las tarifas vigentes, porque son las de lista
  usadas en la Ev.1.
- **Para la defensa.** Una corrida trazada de C3 sobre el MSA muestra en vivo el ciclo completo: la búsqueda del
  Auditor, el rechazo del Verificador, la devolución del Flow y la aprobación en la segunda ronda, con el costo
  de cada paso.

### 11.8 Preguntas previsibles

**¿No es esto un pipeline con varios prompts?** No. Cada agente tiene herramientas distintas, impuestas por
el filtro MCP y no solo por el prompt. El Verificador no puede buscar, y el único que escribe es el
Redactor. Además hay un rol que rechaza y un Flow que devuelve el trabajo rechazado. La prueba de que la
estructura importa está en la ablación (11.5): sacar o agregar un agente cambia los resultados de forma medible.

**Si un agente único detecta igual, ¿para qué el crew?** Es la pregunta correcta y la respuesta está en los
números. En detección, la mejora del crew sobre el agente único (F1 0,977 frente a 0,952) es de una sola
determinación de 40, dentro del ruido. Lo que sí entrega es:
- **Garantía de evidencia:** fidelidad de cita 1,0 en todas las variantes, verificada en código.
- **Corrección de errores que el agente único deja pasar:** por ejemplo, la sección §6 del MSA, ver 11.4 punto 8.
- **Un reintento que recupera hallazgos.**
- **Separación de permisos y un informe que solo puede citar texto aprobado.**

El costo es de 6 a 12 veces mayor (US$ 0,05 a 0,10 por contrato). Si el uso tolera errores de cita, el agente
único es la opción más razonable; si un abogado junior va a confiar en el informe, la verificación justifica el costo.

**¿Por qué secuencial y no jerárquico?** Se midió (C4): 58 veces el costo del agente único, 6 veces más lento y
peor recall. En este caso las dependencias entre tareas son fijas, así que el manager no tiene nada útil que
decidir: solo agrega llamadas y fallas de delegación.

**¿Por qué el puntaje de severidad no lo da el LLM?** Porque tiene que ser reproducible y discutible. Una
tabla de reglas se puede auditar y ajustar a la política de riesgo del cliente; la "opinión" del modelo cambia
entre corridas.

**¿Qué pasa si el LLM entra en un bucle?** Hay seis topes independientes: `max_iter` y `max_execution_time` por
agente, tope de rechazos del guardrail, un único reintento en el Flow, `max_pasos` en los agentes desde cero y
corte por llamada repetida. CrewAI además rechaza un listener que se escuche a sí mismo.
