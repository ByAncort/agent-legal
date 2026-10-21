"""Verificación de citas (notebook §7.1): barata, determinística y la principal defensa contra
la alucinación de cláusulas."""
import re
import unicodedata

from .corpus import chunks

RE_CITA = re.compile(r'"([^"]{15,})"')


def normalizar(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def verificar_citas(respuesta, indices):
    fuente = normalizar(" ".join(chunks()[i]["texto"] for i in indices))
    resultados = [(cita, normalizar(cita) in fuente) for cita in RE_CITA.findall(respuesta)]
    return resultados, all(ok for _, ok in resultados)


TIPOGRAFICOS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalizar_cita(t):
    '''normalizar() + Unicode NFKC + comillas y guiones tipográficos a ASCII.

    - El PDF trae "contractor’s" y un LLM (o CUAD) escribe "contractor's": es la misma cita.
    - Los 4 PDF del corpus usan U+037E (GREEK QUESTION MARK, idéntico a la vista) en vez del
      punto y coma; un LLM siempre escribe ";" ASCII. Sin NFKC, ninguna cita con punto y coma
      podía verificarse.'''
    return normalizar(unicodedata.normalize("NFKC", t).translate(TIPOGRAFICOS))


def cita_existe(cita, texto):
    '''Como verificar_citas, pero exige límites de palabra en los extremos de la cita.

    La comparación por subcadena de la Ev.1 acepta "either party shall assign" dentro de
    "neither party shall assign": la cita "existe" y dice lo contrario del contrato.
    '''
    patron = r"(?<!\w)" + re.escape(normalizar_cita(cita)) + r"(?!\w)"
    return re.search(patron, normalizar_cita(texto)) is not None
