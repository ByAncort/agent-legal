# Guía de ejecución en vivo — ClauseFinder Crew (CMD)

Pasos para mostrar el sistema funcionando en este computador, desde la consola **CMD** de Windows. Hay dos
formas de ver trabajar a los agentes:

- en la consola, donde se ve lo que hace cada agente y el resultado;
- en **Phoenix** (`http://localhost:6006`), donde cada ejecución queda como un árbol de trazas: agente →
  llamada al LLM → llamada a herramienta.

Todo está instalado: `.venv-crew` (el sistema), `.venv-phoenix` (Phoenix), los modelos de embeddings en
caché y la clave en `.env`. Lo único que requiere internet es la API de DeepSeek.

---

## 1. Antes de empezar (5 minutos antes)

Abrir **tres ventanas de CMD** (Inicio → escribir `cmd` → Enter). En **cada una** escribir, línea por línea:

```bat
cd /d "C:\Users\user\Desktop\DUOC UC\INGENIERIA DE SOLUCIONES CON INTELIGENCIA ARTIFICIAL_012V_OLS"
.venv-crew\Scripts\activate.bat
chcp 65001
set PYTHONUTF8=1
```

El prompt queda con el prefijo `(.venv-crew)`. `chcp 65001` y `PYTHONUTF8` hacen que las tildes y los
símbolos se vean bien.

### Ventana 1: Phoenix (trazas)

```bat
python -m src.observabilidad
```

Esperar a que diga `Phoenix en http://localhost:6006` y abrir esa dirección en el navegador. Las trazas de
ejecuciones anteriores siguen ahí, porque se guardan en `data/phoenix/`. Dejar la ventana abierta.

### Ventana 2: servidor MCP (herramientas de los agentes)

```bat
python -m src.mcp_server
```

Esperar a que diga `Servidor MCP listo en http://localhost:8765/mcp` (unos 50 segundos: carga los modelos
y los índices una sola vez). Dejar la ventana abierta.

> Si no se levanta, el crew lo hace solo, pero agrega esa espera a la demo.

### Ventana 3: consultas

Además de los cuatro comandos iniciales:

```bat
set PHOENIX_TRAZAS=1
```

Con esto, todo lo que se ejecute en esta ventana aparece en Phoenix. Sin esa línea funciona igual, pero
no deja trazas.

---

## 2. Comprobación rápida, sin usar el LLM (1 minuto, US$ 0)

```bat
python -m pytest tests -q
```

Resultado esperado: `28 passed`. Esto prueba:
- la validación de las herramientas;
- la escritura segura del informe;
- las reglas de decisión y severidad;
- la memoria.

```bat
python -m tests.probar_mcp
```

Un cliente MCP real se conecta al servidor y:
- lista las 4 herramientas, con su anotación de solo lectura o escritura;
- verifica una cita real (`cita_existe: True`) y una con el sentido invertido (`False`);
- muestra que un `doc_id` inválido devuelve los valores válidos.

---

## 3. Consulta 1: chat sobre un contrato, con memoria (≈1 min la primera pregunta, ≈20 s las demás)

```bat
python -m src.chat.consulta --doc MSA_MeritLife_2020
```

Escribir una pregunta después de `>` y presionar Enter. La **primera** respuesta tarda alrededor de un
minuto, porque el chat carga los modelos de búsqueda; las siguientes, unos 20 s. El aviso
`You are sending unauthenticated requests to the HF Hub` se puede ignorar: los modelos ya están en caché.

Sugerencia de secuencia, para mostrar la memoria a corto plazo:

```
¿El contrato limita la responsabilidad del contratista? ¿En qué sección?
¿Esa cláusula tiene alguna excepción?
¿Se puede ceder el contrato a un tercero?
```

La segunda pregunta ("esa cláusula") solo se entiende con la primera: el agente la resuelve con el
historial. Para salir, **Enter en una línea vacía**.

**En Phoenix:** *Projects → clausefinder-crew → Sessions*. La conversación aparece como **una sesión**, con
un turno por pregunta. Al abrir un turno se ve el agente, sus llamadas al LLM y las herramientas que usó
(`buscar_clausulas`, `leer_seccion`, `verificar_cita`).

Costo: unos US$ 0,002 por pregunta.

---

## 4. Consulta 2: el crew completo revisa un contrato (≈2–4 min con el servidor MCP ya levantado)

```bat
python -m src.crew.flow --doc Mantenimiento_XImage_Sagem_1999 --config C3
```

Es el contrato más corto. En la consola se ve trabajar a cada agente:
- los **3 auditores en paralelo**;
- el **Defensor**;
- el **Verificador**;
- el **Redactor**, que escribe el informe;
- los eventos del Flow, marcados con `>>> [flow]`.

Al final aparece un resumen: estado y severidad de cada una de las 10 categorías, llamadas al LLM, costo y
la carpeta de la corrida.

Para mostrar el **rechazo → reintento → aprobación** se usa el MSA, que suele activarlo:

```bat
python -m src.crew.flow --doc MSA_MeritLife_2020 --config C3
```

Ver el informe que escribió el Redactor:

```bat
dir /b /o-d outputs\informes
type outputs\informes\<nombre_del_archivo>.md
```

(`dir /o-d` lista primero el más reciente.)

**En Phoenix:** pestaña *Traces*, traza `revisión <contrato> · C3`. El árbol muestra el Flow, la crew de
revisión, cada agente con sus llamadas al LLM y a las herramientas MCP, y la crew del informe con
`guardar_informe`. En cada nodo se ven el prompt, la respuesta, los tokens, la latencia y el costo.

Costo: entre US$ 0,07 y 0,15 por contrato.

Otras configuraciones (`--config`): `C1` (Auditor + Verificador), `C2` (+ Defensor), `C3` (+ reintento),
`C4` (proceso jerárquico, lento y caro), `C3-mem` (+ memoria a largo plazo). Con `--sin-informe` se omite
el Redactor.

---

## 5. Consulta 3: agente único, para comparar (≈1–2 min)

Function calling nativo (JSON Schema):

```bat
python -m src.agents_scratch.function_calling --doc Mantenimiento_XImage_Sagem_1999
```

ReAct escrito desde cero (texto interpretado con expresiones regulares):

```bat
python -m src.agents_scratch.react --doc Mantenimiento_XImage_Sagem_1999
```

Cada uno imprime sus llamadas a herramientas paso a paso y al final las 10 determinaciones, el costo y los
errores de formato. En Phoenix aparecen como `agente function calling · …` o `agente ReAct · …`.

Costo: unos US$ 0,007 cada uno.

---

## 6. Resultados de los experimentos, sin gastar (US$ 0)

```bat
python -m eval.exp2_ablacion --solo-tabla
python -m eval.exp1_patrones --solo-tabla
python -m eval.analisis_errores
```

- `exp2_ablacion`: tabla de las 7 configuraciones contra las 40 etiquetas de CUAD.
- `exp1_patrones`: ReAct, function calling y LangChain.
- `analisis_errores`: cada error con su explicación.

---

## 7. Alternativa: Jupyter

```bat
jupyter notebook consultas_phoenix.ipynb
```

Elegir el kernel **"Python (ClauseFinder Crew)"** (menú *Kernel → Change kernel*) y ejecutar las celdas en
orden. Hacen lo mismo que las secciones 3 a 5, con trazas en Phoenix. `multiagente.ipynb` muestra los
resultados guardados y no necesita API key.

---

## 8. Referencia rápida

| `--doc` | Contrato |
| --- | --- |
| `MSA_MeritLife_2020` | Master Services Agreement, Merit Life (2020) |
| `CoBranding_Healthcentral_1999` | Co-branding, Healthcentral (1999), el más largo |
| `Licencia_Mantenimiento_SparklingSpring_2002` | Licencia de software y mantenimiento (2002) |
| `Mantenimiento_XImage_Sagem_1999` | Mantenimiento, XImage / Sagem (1999), el más corto |

Tiempos y costos medidos al probar esta guía en CMD (9 de octubre de 2026):

| Comando | Tiempo | Costo aprox. |
| --- | --- | --- |
| `pytest` (28 tests) | ≈ 40 s | US$ 0 |
| `probar_mcp` / `--solo-tabla` | < 15 s | US$ 0 |
| Chat: primera pregunta / siguientes | ≈ 1 min / ≈ 20 s | US$ 0,002 por pregunta |
| Agente único (XImage) | ≈ 2 min | US$ 0,003–0,007 |
| Crew C3 (XImage, con MCP levantado) | ≈ 2 min 15 s | US$ 0,06 |
| Crew C3 (MSA, con reintento) | 3–4 min | US$ 0,10–0,15 |

---

## 9. Si algo falla

- **"Address already in use" o el servidor no arranca.** Algo ocupa el puerto. Buscar el proceso y cerrarlo:
  ```bat
  netstat -ano | findstr :8765
  taskkill /PID <número_de_la_última_columna> /F
  ```
  Para Phoenix, lo mismo con `:6006`.
- **`Falta DEEPSEEK_API_KEY en .env`.** El archivo `.env` debe tener la línea `DEEPSEEK_API_KEY=sk-...`.
- **Caracteres raros (Ã³, â†’) en la consola.** Faltó `chcp 65001` o `set PYTHONUTF8=1` en esa ventana.
- **`No module named ...`.** No está activado el venv: falta `.venv-crew\Scripts\activate.bat`, o se
  activó otro Python.
- **Detener cualquier comando:** `Ctrl+C`.
- **Sin internet (plan B):** todo lo que no llama a DeepSeek sigue funcionando:
  - las secciones 2 y 6;
  - las trazas ya guardadas en Phoenix;
  - el informe real de CoBranding en `eval/results/demo_crew/`;
  - `multiagente.ipynb`.
