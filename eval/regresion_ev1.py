"""Regresión de la extracción a src/rag: debe reproducir la tabla del notebook (§8).

    python -m eval.regresion_ev1

Valores de referencia en clausefinder.ipynb: 88 chunks; Recall@10 0,385 / 0,692 / 0,577 / 0,808;
Recall@k del pipeline completo 0,385 / 0,500 / 0,654 / 0,808 para k = 1, 3, 5, 10.
"""
import sys

import numpy as np

from src.rag.corpus import chunks, documentos
from src.rag.retrieve import buscar_bm25, buscar_denso, pipeline_retrieval, recuperar, reordenar

# (pregunta en español, doc_id esperado, artículo esperado) — copiado del notebook
EVAL = [
    # Master Services Agreement — Merit Life (2020)
    ("En el MSA de Merit Life, ¿cómo se compensa al contratista?",                        "MSA_MeritLife_2020", "3"),
    ("En el MSA de Merit Life, ¿quién es dueño de los materiales producidos?",            "MSA_MeritLife_2020", "4"),
    ("En el MSA de Merit Life, ¿qué se considera información confidencial?",              "MSA_MeritLife_2020", "8"),
    ("En el MSA de Merit Life, ¿se puede ceder o subcontratar el contrato?",              "MSA_MeritLife_2020", "10"),
    ("En el MSA de Merit Life, ¿qué seguros debe mantener el contratista?",               "MSA_MeritLife_2020", "14"),
    ("En el MSA de Merit Life, ¿cómo se resuelven las disputas?",                         "MSA_MeritLife_2020", "19"),

    # Co-Branding Agreement — Healthcentral.com (1999)
    ("En el acuerdo de co-branding de Healthcentral, ¿cómo se manejan los ingresos por publicidad?", "CoBranding_Healthcentral_1999", "4"),
    ("En el acuerdo de co-branding de Healthcentral, ¿qué pasa con la información de los clientes?", "CoBranding_Healthcentral_1999", "7"),
    ("En el acuerdo de co-branding de Healthcentral, ¿hay cláusula de exclusividad?",                "CoBranding_Healthcentral_1999", "9"),
    ("En el acuerdo de co-branding de Healthcentral, ¿cuál es el plazo y cómo se termina?",          "CoBranding_Healthcentral_1999", "11"),
    ("En el acuerdo de co-branding de Healthcentral, ¿quién indemniza frente a reclamos de terceros?", "CoBranding_Healthcentral_1999", "14"),
    ("En el acuerdo de co-branding de Healthcentral, ¿cuál es el límite de responsabilidad?",        "CoBranding_Healthcentral_1999", "15"),
    ("En el acuerdo de co-branding de Healthcentral, ¿qué ley lo rige?",                             "CoBranding_Healthcentral_1999", "21"),

    # Licencia de software y mantenimiento — Sparkling Spring (2002)
    ("En la licencia de Sparkling Spring, ¿qué restricciones de uso tiene el software?",             "Licencia_Mantenimiento_SparklingSpring_2002", "3"),
    ("En la licencia de Sparkling Spring, ¿se puede rechazar el software tras las pruebas?",         "Licencia_Mantenimiento_SparklingSpring_2002", "7"),
    ("En la licencia de Sparkling Spring, ¿qué cubren los servicios de mantenimiento y soporte?",    "Licencia_Mantenimiento_SparklingSpring_2002", "10"),
    ("En la licencia de Sparkling Spring, ¿qué obligaciones de confidencialidad hay?",               "Licencia_Mantenimiento_SparklingSpring_2002", "14"),
    ("En la licencia de Sparkling Spring, ¿existe depósito del código fuente en escrow?",            "Licencia_Mantenimiento_SparklingSpring_2002", "16"),
    ("En la licencia de Sparkling Spring, ¿qué secciones sobreviven a la terminación?",              "Licencia_Mantenimiento_SparklingSpring_2002", "18"),
    ("En la licencia de Sparkling Spring, ¿se puede contratar a empleados de la otra parte?",        "Licencia_Mantenimiento_SparklingSpring_2002", "19"),

    # Contrato de mantenimiento — XIMAGE / SAGEM Morpho (1999)
    ("En el contrato de XIMAGE con SAGEM, ¿hay obligación de no divulgación?",            "Mantenimiento_XImage_Sagem_1999", "7"),
    ("En el contrato de XIMAGE con SAGEM, ¿en cuánto tiempo debe responder XIMAGE?",      "Mantenimiento_XImage_Sagem_1999", "8"),
    ("En el contrato de XIMAGE con SAGEM, ¿qué pasa si se requiere intervención en terreno?", "Mantenimiento_XImage_Sagem_1999", "9"),
    ("En el contrato de XIMAGE con SAGEM, ¿qué garantía se otorga sobre el software?",    "Mantenimiento_XImage_Sagem_1999", "14"),
    ("En el contrato de XIMAGE con SAGEM, ¿cuáles son los límites de responsabilidad?",   "Mantenimiento_XImage_Sagem_1999", "15"),
    ("En el contrato de XIMAGE con SAGEM, ¿cómo se resuelven los conflictos?",            "Mantenimiento_XImage_Sagem_1999", "16"),
]


def acierta(idx, doc_esperado, seccion_esperada):
    c = chunks()[idx]
    return c["doc_id"] == doc_esperado and c["seccion"] == seccion_esperada


def evaluar(estrategia, k=10):
    recalls, reciprocos = [], []
    for pregunta, doc, seccion in EVAL:
        indices = estrategia(pregunta)[:k]
        posiciones = [i for i, idx in enumerate(indices) if acierta(idx, doc, seccion)]
        recalls.append(1.0 if posiciones else 0.0)
        reciprocos.append(1.0 / (posiciones[0] + 1) if posiciones else 0.0)
    return np.mean(recalls), np.mean(reciprocos)


ESTRATEGIAS = {
    "Solo BM25 (léxico)":      lambda q: buscar_bm25(q),
    "Solo denso (embeddings)": lambda q: buscar_denso(q),
    "Híbrido (RRF)":           lambda q: recuperar(q),
    "Híbrido + re-ranking":    lambda q: reordenar(q, recuperar(q), k=10),
}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"{len(chunks())} chunks a partir de {len(documentos())} contratos\n")

    demo = "En el contrato de XIMAGE con SAGEM, ¿en cuánto tiempo debe responder XIMAGE?"
    print("Top-5 tras re-ranking (demo del notebook §6):")
    for pos, idx in enumerate(pipeline_retrieval(demo), 1):
        c = chunks()[idx]
        print(f"{pos}. {c['doc_id']:<24} [{c['seccion']}] {c['epigrafe'][:40]}")

    print(f"\n{'Estrategia':<26}{'Recall@10':>12}{'MRR':>10}")
    print("─" * 48)
    for nombre, estrategia in ESTRATEGIAS.items():
        recall, mrr = evaluar(estrategia)
        print(f"{nombre:<26}{recall:>12.3f}{mrr:>10.3f}")

    rankings = [reordenar(q, recuperar(q), k=10) for q, _, _ in EVAL]
    print(f"\n{'k':>4}{'Recall@k':>12}")
    print("─" * 16)
    for k in (1, 3, 5, 10):
        aciertos = [any(acierta(idx, doc, sec) for idx in ranking[:k])
                    for ranking, (_, doc, sec) in zip(rankings, EVAL)]
        print(f"{k:>4}{np.mean(aciertos):>12.3f}")
