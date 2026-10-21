"""Checklist de riesgo contractual: 10 categorías con etiqueta experta en CUAD.

Cada categoría trae su columna en CUAD (master_clauses.csv), una definición en español para los
prompts y términos en inglés para las búsquedas, porque los contratos están en inglés.

Balance en los 4 contratos del corpus: 21 presentes / 19 ausentes. "Liquidated Damages" está
ausente en los cuatro: es el control negativo para medir falsos positivos.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Categoria:
    clave: str
    cuad: str
    nombre: str
    definicion: str
    terminos: str
    grupo: str         # para repartir el checklist en tareas concurrentes del Auditor


CATEGORIAS = [
    Categoria("uncapped_liability", "Uncapped Liability", "Responsabilidad ilimitada",
              "Alguna parte responde sin tope frente a ciertos incumplimientos (p. ej. el tope "
              "no aplica a indemnizaciones, confidencialidad o dolo).",
              "unlimited liability; cap shall not apply; exclusions from limitation of liability",
              "responsabilidad"),
    Categoria("cap_on_liability", "Cap On Liability", "Tope de responsabilidad",
              "Existe un monto o fórmula máxima de responsabilidad, o se excluyen daños "
              "indirectos/consecuenciales.",
              "limitation of liability; shall not exceed; consequential damages; aggregate liability",
              "responsabilidad"),
    Categoria("liquidated_damages", "Liquidated Damages", "Multas o daños preestablecidos",
              "El contrato fija de antemano una suma a pagar por incumplimiento o terminación "
              "(cláusula penal, multa, liquidated damages).",
              "liquidated damages; penalty; termination fee; shall pay the sum of",
              "responsabilidad"),
    Categoria("termination_for_convenience", "Termination For Convenience",
              "Terminación por conveniencia",
              "Una parte puede terminar el contrato sin causa, solo con aviso previo.",
              "terminate for convenience; without cause; upon written notice; at any time",
              "vigencia"),
    Categoria("renewal_term", "Renewal Term", "Renovación automática",
              "Al vencer el plazo inicial el contrato se renueva por periodos adicionales, "
              "automáticamente o por defecto.",
              "renew automatically; successive periods; renewal term; unless either party gives notice",
              "vigencia"),
    Categoria("anti_assignment", "Anti-Assignment", "Prohibición de cesión",
              "Ceder o transferir el contrato requiere consentimiento o aviso a la otra parte.",
              "assign; transfer; subcontract; prior written consent",
              "vigencia"),
    Categoria("governing_law", "Governing Law", "Ley aplicable",
              "El contrato declara qué ley o jurisdicción lo rige.",
              "governed by the laws of; governing law; construed in accordance with",
              "vigencia"),
    Categoria("exclusivity", "Exclusivity", "Exclusividad",
              "Una parte se obliga a tratar exclusivamente con la otra o le concede derechos "
              "exclusivos (proveedor, territorio, canal).",
              "exclusive; exclusively; sole provider; shall not engage any other",
              "competencia"),
    Categoria("non_compete", "Non-Compete", "No competencia",
              "Una parte no puede competir con la otra o operar en cierto mercado, territorio o "
              "línea de negocio.",
              "non-compete; shall not compete; competing product; competitive business",
              "competencia"),
    Categoria("ip_ownership_assignment", "Ip Ownership Assignment", "Cesión de propiedad intelectual",
              "La propiedad intelectual creada por una parte pasa a la otra (work made for hire, "
              "cesión de derechos).",
              "work made for hire; assigns all right, title and interest; intellectual property ownership",
              "competencia"),
]

POR_CLAVE = {c.clave: c for c in CATEGORIAS}
GRUPOS = {g: [c for c in CATEGORIAS if c.grupo == g] for g in ("responsabilidad", "vigencia", "competencia")}


def checklist(categorias=CATEGORIAS):
    '''Texto para prompts: una línea por categoría con su definición y términos de búsqueda.'''
    return "\n".join(f"- {c.clave} ({c.nombre}): {c.definicion} Términos útiles: {c.terminos}."
                     for c in categorias)
