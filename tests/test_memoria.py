"""La memoria a largo plazo con embedder propio (sin LLM ni escritura en disco)."""
from src.crew.memoria import _embedder, textos_lecciones
from src.crew.schemas import Hallazgo, HallazgoFinal


def test_embedder_devuelve_listas_no_ndarray():
    # CrewAI evalúa `if not resultado`: con un ndarray eso lanza ValueError y recall falla en silencio
    vectores = _embedder(["cláusula de exclusividad", "governing law"])
    assert isinstance(vectores, list) and isinstance(vectores[0], list)
    assert len(vectores[0]) == 384                     # multilingual-e5-small


def test_se_aprende_de_descartes_y_rechazos_no_de_ausentes():
    h = Hallazgo(categoria="exclusivity", presente=True, afirmacion="x", seccion="9", cita="y" * 20)
    finales = [
        HallazgoFinal(categoria="exclusivity", estado="descartado", hallazgo=h, razones=["es confidencialidad"]),
        HallazgoFinal(categoria="non_compete", estado="rechazado", hallazgo=h.model_copy(update={"categoria": "non_compete"}),
                      razones=["la cita no existe"]),
        HallazgoFinal(categoria="governing_law", estado="ausente",
                      hallazgo=Hallazgo(categoria="governing_law", presente=False)),
    ]
    lecciones, aprobados = textos_lecciones("DOC", finales, [["cita mal copiada"]])
    assert len(lecciones) == 3 and not aprobados
    assert any("FALSO POSITIVO" in l for l in lecciones)
