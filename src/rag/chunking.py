"""Chunking por cláusula (notebook §2)."""
import re

from .config import MAX_CHARS

# "3. TERMINATION", "ARTICLE V", "SECTION 8 —"
RE_ENCABEZADO = re.compile(
    r"^\s*((?:ARTICLE|SECTION|ART[IÍ]CULO|SECCI[OÓ]N)\s+[IVXLC0-9]+|[0-9]+(?:\.[0-9]+)*)"
    r"\s*[.\-—:)]?\s+(.*)$"
)
# el PDF a veces deja el número solo en una línea y el título en la siguiente
RE_SOLO_NUMERO = re.compile(r"^\s*([0-9]{1,2}|(?:ARTICLE|SECTION)\s+[IVXLC0-9]+)\s*\.?\s*$", re.I)
# subincisos, solo para partir artículos demasiado largos
RE_SUBINCISO = re.compile(r"(?=\((?:[a-z]|[ivx]+|[0-9])\)\s)")

ROMANOS = {r: i + 1 for i, r in enumerate(
    "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX "
    "XXI XXII XXIII XXIV XXV".split())}


def _valor(numero):
    '''Valor numérico del encabezado: "7" -> 7, "ARTICLE V" -> 5.'''
    partes = numero.upper().split()
    if len(partes) == 2:
        return ROMANOS.get(partes[1], int(partes[1]) if partes[1].isdigit() else None)
    return int(numero) if numero.isdigit() else None


def candidatos(texto):
    lineas, pos, encontrados = texto.split("\n"), 0, []
    for i, linea in enumerate(lineas):
        numero = titulo = None
        m = RE_ENCABEZADO.match(linea)
        if m and len(linea.strip()) > 3:
            numero, titulo = m.group(1).strip(), m.group(2).strip()
        else:
            m2 = RE_SOLO_NUMERO.match(linea)
            if m2 and i + 1 < len(lineas) and lineas[i + 1].strip():
                numero, titulo = m2.group(1).strip(), lineas[i + 1].strip()
        if numero:
            encontrados.append({"inicio": pos, "numero": numero, "titulo": titulo,
                                "valor": _valor(numero)})
        pos += len(linea) + 1
    return encontrados


def mejor_secuencia(cands):
    '''La numeración de artículos es la corrida ascendente más larga que arranca en 1.'''
    nivel1 = [c for c in cands if c["valor"] is not None and "." not in c["numero"]]
    mejor = []
    for i, arranque in enumerate(nivel1):
        if arranque["valor"] != 1:
            continue
        corrida, esperado = [arranque], 2
        for c in nivel1[i + 1:]:
            if c["valor"] == esperado:
                corrida.append(c); esperado += 1
            elif c["valor"] == esperado + 1:      # encabezado perdido en la extracción
                corrida.append(c); esperado = c["valor"] + 1
        if len(corrida) > len(mejor):
            mejor = corrida
    return mejor


def segmentar(doc):
    texto  = doc["texto"]
    cands  = candidatos(texto)
    cortes = mejor_secuencia(cands)
    if len(cortes) < 3:   # contrato sin numeración reconocible
        cortes = [c for c in cands if c["valor"] is not None] \
                 or [{"inicio": 0, "numero": "—", "titulo": doc["titulo"]}]

    secciones = []
    for i, c in enumerate(cortes):
        fin = cortes[i + 1]["inicio"] if i + 1 < len(cortes) else len(texto)
        seccion = {**c, "fin": fin, "texto": texto[c["inicio"]:fin].strip()}
        if seccion["texto"]:
            secciones.append(seccion)
    return secciones


def trocear_largos(seccion):
    '''Si un artículo supera el techo se parte por subincisos, nunca por caracteres.'''
    if len(seccion["texto"]) <= MAX_CHARS:
        return [seccion]

    partes, acumulado, inicio = [], "", seccion["inicio"]
    for trozo in RE_SUBINCISO.split(seccion["texto"]):
        if len(acumulado) + len(trozo) > MAX_CHARS and acumulado:
            partes.append({**seccion, "texto": acumulado.strip(),
                           "inicio": inicio, "fin": inicio + len(acumulado)})
            inicio += len(acumulado)
            acumulado = trozo
        else:
            acumulado += trozo
    if acumulado.strip():
        partes.append({**seccion, "texto": acumulado.strip(),
                       "inicio": inicio, "fin": inicio + len(acumulado)})
    return partes


def construir_chunks(documentos):
    chunks = []
    for doc in documentos:
        for seccion in segmentar(doc):
            for parte in trocear_largos(seccion):
                ruta = f"{doc['titulo']} > {parte['numero']} {parte['titulo']}"
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}::{parte['numero']}::{len(chunks)}",
                    "doc_id":   doc["doc_id"],
                    "titulo":   doc["titulo"],
                    "seccion":  parte["numero"],
                    "epigrafe": parte["titulo"][:70],
                    "pagina":   doc["texto"].count("\f", 0, parte["inicio"]) + 1,
                    "inicio":   parte["inicio"],
                    "fin":      parte["fin"],
                    "texto":    parte["texto"],
                    # se indexa con la ruta jerárquica: la cláusula sola es ininteligible
                    "texto_indexado": f"{ruta}\n{parte['texto']}",
                })
    return chunks
