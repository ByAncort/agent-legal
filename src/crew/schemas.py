"""Salidas estructuradas de cada tarea del crew (`output_pydantic`). Una entrada por categoría:
la categoría es la clave que une el trabajo de los tres agentes."""
from typing import Literal

from pydantic import BaseModel, Field

from src.esquemas import ClaveCategoria


class Hallazgo(BaseModel):
    categoria: ClaveCategoria
    presente: bool
    afirmacion: str = Field("", description="Qué establece la cláusula y por qué es un riesgo, en español.")
    seccion: str | None = Field(None, description="Número de artículo de la cita.")
    cita: str | None = Field(None, description="Copia literal del contrato, 15 a 300 caracteres.")
    mutua: bool | None = Field(None, description="True si aplica a ambas partes por igual.")
    con_excepciones: bool | None = Field(None, description="True si el mismo texto trae excepciones o límites.")


class Auditoria(BaseModel):
    hallazgos: list[Hallazgo]


class Matiz(BaseModel):
    categoria: ClaveCategoria
    veredicto: Literal["se_sostiene", "se_matiza", "se_descarta"]
    argumento: str
    seccion_contracita: str | None = None
    contracita: str | None = None


class Defensa(BaseModel):
    matices: list[Matiz]


class Verificacion(BaseModel):
    categoria: ClaveCategoria
    cita_existe: bool = Field(..., description="Lo que respondió verificar_cita.")
    cita_respalda: bool = Field(..., description="La cita establece lo que afirma el hallazgo.")
    matiz_respaldado: bool | None = Field(None, description="Vacío si el Defensor no matizó ni descartó.")
    motivo: str


class Verificaciones(BaseModel):
    verificaciones: list[Verificacion]


Estado = Literal["aprobado", "rechazado", "descartado", "ausente"]


class HallazgoFinal(BaseModel):
    '''Resultado de combinar los tres agentes con reglas en código (src/crew/severity.py).'''
    categoria: ClaveCategoria
    estado: Estado
    hallazgo: Hallazgo
    matiz: Matiz | None = None
    verificacion: Verificacion | None = None
    severidad: Literal["Alta", "Media", "Baja"] | None = None
    razones: list[str] = []
    intento: int = 0
