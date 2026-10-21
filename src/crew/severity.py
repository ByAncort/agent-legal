"""Decisión final y severidad: reglas en código, no opinión del LLM.

Los agentes aportan hechos y juicios acotados (hay cita, la cita respalda, el matiz se sostiene,
la obligación es mutua). Lo que se hace con ellos lo deciden estas reglas, que son las mismas en
cada corrida y se pueden leer y discutir. La existencia literal de cada cita se vuelve a
comprobar aquí con la tool determinística: no se confía en que el Verificador la haya
transcrito bien.
"""
from src.crew.schemas import HallazgoFinal
from src.tools.core import ejecutar

# Política de riesgo del cliente: severidad base de cada categoría cuando está presente.
SEVERIDAD_BASE = {
    "uncapped_liability":          3,
    "liquidated_damages":          3,
    "non_compete":                 3,
    "exclusivity":                 3,
    "ip_ownership_assignment":     2,
    "termination_for_convenience": 2,
    "renewal_term":                2,
    "anti_assignment":             1,
    "cap_on_liability":            1,
    "governing_law":               1,
}
NIVEL = {3: "Alta", 2: "Media", 1: "Baja"}


def ubicar(cita, doc_id, seccion):
    '''Sección donde está realmente la cita, o None si no existe en el contrato.

    Si el agente dio otra sección (p. ej. la numeración impresa "6" cuando la segmentación dejó
    ese artículo dentro del chunk "5"), se corrige aquí, de forma determinística.'''
    if not cita or not seccion:
        return None
    r = ejecutar("verificar_cita", {"cita": cita, "doc_id": doc_id, "seccion": str(seccion)})
    if r.get("cita_existe"):
        return str(seccion)
    donde = r.get("secciones_donde_aparece") or []
    return donde[0] if donde else None


def _existe(cita, doc_id, seccion):
    return ubicar(cita, doc_id, seccion) is not None


def severidad(hallazgo, matiz_valido):
    puntos = SEVERIDAD_BASE[hallazgo.categoria]
    razones = [f"severidad base de la categoría: {NIVEL[puntos]}"]
    if hallazgo.mutua:
        puntos -= 1
        razones.append("la obligación es mutua")
    if matiz_valido and matiz_valido.veredicto == "se_matiza":
        puntos -= 1
        razones.append(f"matizada por el Defensor (§{matiz_valido.seccion_contracita})")
    return NIVEL[max(1, puntos)], razones


def decidir(doc_id, auditoria, defensa, verificaciones, intento=0):
    '''Combina los tres agentes en un HallazgoFinal por categoría.

    presente=false                                   → ausente
    cita inexistente, o el Verificador dice que no
    respalda, o no hay verificación                  → rechazado (vuelve al Auditor si quedan intentos)
    el Defensor descarta con contracita existente
    y respaldada                                     → descartado
    en otro caso                                     → aprobado, con severidad por reglas
    '''
    matices = {m.categoria: m for m in (defensa.matices if defensa else [])}
    verifs = {v.categoria: v for v in (verificaciones.verificaciones if verificaciones else [])}
    finales = []
    for h in auditoria.hallazgos:
        m, v = matices.get(h.categoria), verifs.get(h.categoria)
        if not h.presente:
            finales.append(HallazgoFinal(categoria=h.categoria, estado="ausente", hallazgo=h, intento=intento))
            continue

        seccion_real = ubicar(h.cita, doc_id, h.seccion)
        correccion = []
        if seccion_real and seccion_real != str(h.seccion):
            correccion = [f"sección corregida por el sistema: §{h.seccion} → §{seccion_real}"]
            h = h.model_copy(update={"seccion": seccion_real})
        if seccion_real is None or v is None or not v.cita_respalda:
            if seccion_real is None:
                motivo = ["la cita no existe literalmente en el contrato"] + ([v.motivo] if v else [])
            else:
                motivo = ["sin verificación"] if v is None else [v.motivo]
            finales.append(HallazgoFinal(categoria=h.categoria, estado="rechazado", hallazgo=h, matiz=m,
                                         verificacion=v, razones=motivo, intento=intento))
            continue

        matiz_valido = None
        if m and m.veredicto != "se_sostiene":
            if v.matiz_respaldado and _existe(m.contracita, doc_id, m.seccion_contracita):
                matiz_valido = m
        if matiz_valido and matiz_valido.veredicto == "se_descarta":
            finales.append(HallazgoFinal(categoria=h.categoria, estado="descartado", hallazgo=h, matiz=m,
                                         verificacion=v, razones=[m.argumento], intento=intento))
            continue

        nivel, razones = severidad(h, matiz_valido)
        finales.append(HallazgoFinal(categoria=h.categoria, estado="aprobado", hallazgo=h,
                                     matiz=matiz_valido, verificacion=v, severidad=nivel,
                                     razones=razones + correccion, intento=intento))
    return finales
