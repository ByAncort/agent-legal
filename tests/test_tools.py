"""Validación de argumentos y seguridad de escritura de las tools (sin LLM ni modelos).

    python -m pytest tests -q
"""
import pytest

from src.tools import core
from src.tools.core import ejecutar

DOC = "MSA_MeritLife_2020"
CITA_REAL = "Neither party shall assign, transfer, or subcontract this Agreement"   # §10


@pytest.fixture
def carpetas_temporales(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "DIR_INFORMES", tmp_path / "informes")
    monkeypatch.setattr(core, "DIR_APROBADOS", tmp_path / "estado")
    monkeypatch.setattr(core, "RAIZ", tmp_path)
    return tmp_path


# --- Validación de argumentos -------------------------------------------------

def test_doc_id_invalido_devuelve_error_con_valores_validos():
    r = ejecutar("leer_seccion", {"doc_id": "NO_EXISTE", "seccion": "1"})
    assert r["error"] == "argumentos inválidos"
    assert DOC in r["detalle"][0]


def test_seccion_inexistente_lista_las_validas():
    r = ejecutar("leer_seccion", {"doc_id": DOC, "seccion": "6"})       # el MSA salta del 5 al 7
    assert "Secciones válidas" in r["detalle"][0]


def test_tool_desconocida_y_argumentos_no_objeto():
    assert "disponibles" in ejecutar("borrar_todo", {})
    assert "error" in ejecutar("leer_seccion", "doc_id=MSA")


def test_limites_de_longitud():
    r = ejecutar("verificar_cita", {"cita": "corta", "doc_id": DOC, "seccion": "10"})
    assert r["error"] == "argumentos inválidos"
    r = ejecutar("buscar_clausulas", {"consulta": "x" * 400, "doc_id": DOC})
    assert r["error"] == "argumentos inválidos"


def test_esquema_openai_expone_enum_de_contratos():
    params = core.esquema_openai("verificar_cita")["function"]["parameters"]
    assert DOC in params["properties"]["doc_id"]["enum"]
    assert set(params["required"]) == {"cita", "doc_id", "seccion"}


# --- Verificación determinística de citas -------------------------------------

def test_cita_real_existe():
    r = ejecutar("verificar_cita", {"cita": CITA_REAL, "doc_id": DOC, "seccion": "10"})
    assert r["cita_existe"] is True


def test_cita_alterada_no_existe():
    alterada = CITA_REAL.replace("assign, transfer", "assign, sell")
    r = ejecutar("verificar_cita", {"cita": alterada, "doc_id": DOC, "seccion": "10"})
    assert r["cita_existe"] is False and r["en_documento"] is False


def test_subcadena_que_invierte_el_sentido_no_cuenta():
    # "Either party..." es subcadena de "Neither party...": la Ev.1 la habría dado por válida
    alterada = CITA_REAL.replace("Neither party", "either party")
    r = ejecutar("verificar_cita", {"cita": alterada, "doc_id": DOC, "seccion": "10"})
    assert r["cita_existe"] is False


def test_cita_en_otra_seccion_indica_donde_esta():
    r = ejecutar("verificar_cita", {"cita": CITA_REAL, "doc_id": DOC, "seccion": "2"})
    assert r["cita_existe"] is False
    assert r["secciones_donde_aparece"] == ["10"]


def test_espacios_y_mayusculas_no_importan():
    r = ejecutar("verificar_cita", {"cita": "  NEITHER party shall\n assign,  transfer", "doc_id": DOC,
                                    "seccion": "10"})
    assert r["cita_existe"] is True


# --- Seguridad en la escritura ------------------------------------------------

INFORME_OK = f'# Informe\n\n- Cesión restringida: "{CITA_REAL}" ({DOC}, §10)\n'


def test_sin_aprobados_no_se_escribe(carpetas_temporales):
    r = ejecutar("guardar_informe", {"doc_id": DOC, "markdown": INFORME_OK})
    assert "No hay hallazgos aprobados" in r["error"]
    assert not (carpetas_temporales / "informes").exists()


def test_cita_no_aprobada_se_rechaza(carpetas_temporales):
    core.registrar_aprobados(DOC, [CITA_REAL])
    informe = INFORME_OK + '\n- Inventado: "The Contractor shall have unlimited liability for all damages"\n'
    r = ejecutar("guardar_informe", {"doc_id": DOC, "markdown": informe})
    assert "citas_no_aprobadas" in r
    assert not (carpetas_temporales / "informes").exists()


def test_informe_sin_citas_se_rechaza(carpetas_temporales):
    core.registrar_aprobados(DOC, [CITA_REAL])
    sin_citas = "# Informe\n\nEl contrato no presenta riesgos relevantes en ninguna de las categorías."
    r = ejecutar("guardar_informe", {"doc_id": DOC, "markdown": sin_citas})
    assert "ninguna cita" in r["error"]


def test_informe_valido_se_guarda_en_ruta_fija(carpetas_temporales):
    core.registrar_aprobados(DOC, [CITA_REAL + ", which will not be unreasonably withheld."])
    informe = INFORME_OK.replace(f'"{CITA_REAL}"', f'“{CITA_REAL}”')     # comillas tipográficas
    r = ejecutar("guardar_informe", {"doc_id": DOC, "markdown": informe})
    assert r["guardado"] is True
    assert r["ruta"].startswith(f"informes/{DOC}_")
    assert len(list((carpetas_temporales / "informes").glob("*.md"))) == 1


def test_informe_con_subcadena_invertida_se_rechaza(carpetas_temporales):
    core.registrar_aprobados(DOC, [CITA_REAL])
    invertida = CITA_REAL.replace("Neither party", "either party")
    informe = f'# Informe\n\n- Cesión: "{invertida}" ({DOC}, §10)\n'
    r = ejecutar("guardar_informe", {"doc_id": DOC, "markdown": informe})
    assert "citas_no_aprobadas" in r


def test_verificar_cita_con_seccion_inexistente_informa_donde_esta():
    r = ejecutar("verificar_cita", {"cita": CITA_REAL, "doc_id": DOC, "seccion": "6"})
    assert r["cita_existe"] is False and r["secciones_donde_aparece"] == ["10"] and "nota" in r


def test_punto_y_coma_griego_del_pdf_equivale_al_ascii():
    # los PDF del corpus traen U+037E en vez de ";": un LLM siempre escribe el ASCII
    from src.rag.verify import cita_existe
    assert cita_existe("a health based web directory; or ii) other", "a health based web directory; or ii) other")
