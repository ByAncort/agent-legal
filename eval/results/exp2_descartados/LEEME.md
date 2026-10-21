# Corridas C4 descartadas (proceso jerárquico sin herramientas)

Primera corrida de C4 (Process.hierarchical). El manager delegó el trabajo, pero los agentes
delegados no recibieron sus herramientas MCP: CrewAI 1.15 solo resuelve `mcps=` para el agente
dueño de la tarea, y en modo jerárquico las tareas no tienen dueño. Resultado: 0 llamadas MCP,
los auditores respondieron "No tengo acceso a la herramienta buscar_clausulas" y las 40
categorías quedaron como ausentes (recall 0), a un costo de unos US$ 0,10 por contrato.

Corrección en src/crew/crew.py: en modo jerárquico, las tools MCP de cada agente se resuelven
por adelantado con `agent.get_mcp_tools()` y se asignan como `tools`. Los resultados válidos de
C4 están en eval/results/exp2/.

# Corridas C3-mem descartadas (memoria a largo plazo que no persistía)

Primera corrida de C3-mem. Hubo dos fallos en src/crew/memoria.py:
1. El embedder devolvía un ndarray de NumPy y CrewAI evalúa `if not resultado`, así que `recall`
   fallaba siempre. El error se atrapaba en silencio, por eso las 4 corridas dicen "0 lecciones
   recuperadas".
2. `remember_many` escribe en un pool de hilos; sin `drain_writes()`, data/memory/ quedó vacío.

Estas corridas equivalen a C3 sin memoria. Los resultados válidos de C3-mem, con ambos fallos
corregidos, están en eval/results/exp2/.

# Corridas C3 y C3-mem con el verificador anterior (punto y coma U+037E)

Los 4 PDF usan U+037E (GREEK QUESTION MARK, idéntico a la vista) en lugar de ";". Un LLM siempre
escribe ";" ASCII, así que cualquier cita con punto y coma fallaba la verificación literal. En
estas corridas eso provocó el rechazo de la exclusividad de CoBranding (un falso negativo), que
el reintento no pudo corregir. La verificación ahora aplica Unicode NFKC (src/rag/verify.py).
Al volver a puntuar todas las corridas (eval/repuntuar.py) ninguna métrica cambió: el error solo
afectaba las decisiones del crew en tiempo de ejecución, por eso se repitieron C3 y C3-mem.
