"""Reglas en código que combinan a los agentes (src/crew/severity.py), sin LLM."""
from src.crew.schemas import Auditoria, Defensa, Hallazgo, Matiz, Verificacion, Verificaciones
from src.crew.severity import decidir

DOC = "MSA_MeritLife_2020"
CITA_10 = "Neither party shall assign, transfer, or subcontract this Agreement"          # §10
CITA_11 = "This Agreement shall be governed by"                                           # §11


def _hallazgo(**kw):
    base = dict(categoria="anti_assignment", presente=True, afirmacion="Ceder requiere consentimiento.",
                seccion="10", cita=CITA_10, mutua=False)
    return Hallazgo(**{**base, **kw})


def _verif(**kw):
    base = dict(categoria="anti_assignment", cita_existe=True, cita_respalda=True, motivo="ok")
    return Verificaciones(verificaciones=[Verificacion(**{**base, **kw})])


def _uno(hallazgo, defensa=None, verificaciones=None):
    return decidir(DOC, Auditoria(hallazgos=[hallazgo]), defensa, verificaciones)[0]


def test_ausente_no_requiere_verificacion():
    assert _uno(_hallazgo(presente=False, cita=None, seccion=None)).estado == "ausente"


def test_aprobado_con_severidad_base():
    f = _uno(_hallazgo(), verificaciones=_verif())
    assert f.estado == "aprobado" and f.severidad == "Baja"


def test_cita_inexistente_se_rechaza_aunque_el_verificador_diga_que_existe():
    # el código vuelve a comprobar la cita: no confía en lo que transcribió el Verificador
    f = _uno(_hallazgo(cita="Either party may assign this Agreement freely to anyone"),
             verificaciones=_verif(cita_existe=True))
    assert f.estado == "rechazado"


def test_sin_verificacion_se_rechaza():
    assert _uno(_hallazgo(), verificaciones=None).estado == "rechazado"


def test_verificador_dice_que_no_respalda():
    f = _uno(_hallazgo(), verificaciones=_verif(cita_respalda=False, motivo="habla de otra cosa"))
    assert f.estado == "rechazado" and f.razones == ["habla de otra cosa"]


def test_descarte_del_defensor_requiere_contracita_existente_y_respaldada():
    matiz = Matiz(categoria="anti_assignment", veredicto="se_descarta", argumento="No aplica.",
                  seccion_contracita="11", contracita=CITA_11)
    defensa = Defensa(matices=[matiz])
    assert _uno(_hallazgo(), defensa, _verif(matiz_respaldado=True)).estado == "descartado"
    # mismo descarte, pero el Verificador no lo respalda: el hallazgo se mantiene
    assert _uno(_hallazgo(), defensa, _verif(matiz_respaldado=False)).estado == "aprobado"
    # contracita inventada: tampoco descarta
    falsa = Defensa(matices=[matiz.model_copy(update={"contracita": "This Agreement may be assigned freely"})])
    assert _uno(_hallazgo(), falsa, _verif(matiz_respaldado=True)).estado == "aprobado"


def test_matiz_y_mutualidad_bajan_la_severidad():
    h = _hallazgo(categoria="exclusivity", mutua=True)
    matiz = Matiz(categoria="exclusivity", veredicto="se_matiza", argumento="Limitada.",
                  seccion_contracita="11", contracita=CITA_11)
    v = _verif(categoria="exclusivity", matiz_respaldado=True)
    assert _uno(h.model_copy(update={"mutua": False}), None, v).severidad == "Alta"
    assert _uno(h, None, v).severidad == "Media"                       # mutua: −1
    assert _uno(h, Defensa(matices=[matiz]), v).severidad == "Baja"    # mutua y matizada: −2


def test_seccion_mal_numerada_se_corrige_si_la_cita_existe_en_otra():
    # el agente usa la numeración impresa; la segmentación tiene la cita en otra sección
    f = _uno(_hallazgo(seccion="6"), verificaciones=_verif())
    assert f.estado == "aprobado" and f.hallazgo.seccion == "10"
    assert any("corregida" in r for r in f.razones)


def test_rechazo_incluye_el_diagnostico_del_verificador():
    f = _uno(_hallazgo(cita="Either party may assign this Agreement freely to anyone"),
             verificaciones=_verif(motivo="la cita está en otra cláusula"))
    assert f.razones == ["la cita no existe literalmente en el contrato", "la cita está en otra cláusula"]
