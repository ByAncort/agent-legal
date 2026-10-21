"""De PDF a texto utilizable (notebook §1)."""
import collections
import re

from .config import DATA_RAW

RE_PUNTOS_GUIA = re.compile(r"\.{4,}\s*\d+\s*$")   # línea de índice: "Definitions.......3"


def listar_pdfs(carpeta=DATA_RAW):
    return sorted(carpeta.glob("*.pdf")) + sorted(carpeta.glob("*.txt"))


def extraer_texto(ruta):
    '''PDF -> texto, separando páginas con \f para poder citar el número de página.'''
    if ruta.suffix.lower() == ".txt":
        return ruta.read_text(encoding="utf-8", errors="ignore")

    import pdfplumber
    with pdfplumber.open(ruta) as pdf:
        return "\f".join((pagina.extract_text() or "") for pagina in pdf.pages)


def limpiar(texto):
    '''Quita el índice y los encabezados/pies que se repiten en casi todas las páginas.'''
    paginas = texto.split("\f")

    frecuencia = collections.Counter()
    for pagina in paginas:
        for linea in {l.strip() for l in pagina.split("\n") if 3 < len(l.strip()) < 80}:
            frecuencia[linea] += 1

    umbral = max(2, int(len(paginas) * 0.5))
    ruido  = {linea for linea, n in frecuencia.items() if n >= umbral}

    limpias = []
    for pagina in paginas:
        limpias.append("\n".join(
            l for l in pagina.split("\n")
            if not RE_PUNTOS_GUIA.search(l) and l.strip() not in ruido
        ))
    return "\f".join(limpias)


def cargar_documentos(rutas=None):
    docs = []
    for ruta in (rutas if rutas is not None else listar_pdfs()):
        texto = limpiar(extraer_texto(ruta))
        docs.append({
            "doc_id":  ruta.stem,
            "texto":   texto,
            "titulo":  next((l.strip() for l in texto.split("\n") if l.strip()), ruta.stem)[:90],
            "paginas": texto.count("\f") + 1,
        })
    return docs
