#!/usr/bin/env python3
"""Retrieval benchmark: baseline vs enriched DB."""
import sqlite3

baseline = sqlite3.connect('/home/andtega349/lisf-agent/docs/regulation_baseline.db')
enriched = sqlite3.connect('/home/andtega349/lisf-agent/docs/regulation.db')

TESTS = [
    ("reservas tecnicas", ["lisf:121", "lisf:216", "cusf:5.1.1"]),
    ("seguros de danos", ["lisf:27", "cusf:6.4.2"]),
    ("gobierno corporativo", ["lisf:69", "cusf:3.1.1"]),
    ("capital minimo pagado", ["lisf:49", "cusf:6.1.2"]),
    ("reaseguradoras extranjeras", ["lisf:107", "cusf:34.1.1"]),
    ("nota tecnica", ["lisf:201", "cusf:4.2.7"]),
    ("agentes de seguros autorizacion", ["lisf:91", "cusf:32.3.1"]),
    ("liquidacion administrativa", ["lisf:393", "lisf:394"]),
    ("requerimiento de capital de solvencia", ["lisf:232", "cusf:6.1.1"]),
    ("fondos propios admisibles", ["lisf:236", "cusf:7.1.1"]),
    ("prueba de solvencia dinamica", ["lisf:245", "cusf:7.2.1"]),
    ("comite de auditoria", ["lisf:70", "cusf:3.8.1"]),
    ("fianzas de credito", ["cusf:19.1.1", "lisf:165"]),
    ("seguros de pensiones", ["cusf:14.1.1", "lisf:27"]),
    ("contralor medico", ["lisf:73", "cusf:15.3.1"]),
    ("operaciones de reaseguro", ["lisf:256", "cusf:9.4.1"]),
    ("microseguros", ["cusf:4.8.1", "cusf:4.8.2"]),
    ("modelo interno rcs", ["lisf:233", "cusf:6.9.1"]),
    ("sanciones multas", ["lisf:474", "lisf:475"]),
    ("cesion de cartera", ["lisf:270", "cusf:29.1.1"]),
]

def recall_at_k(db, query, expected, k, use_weights=False):
    if use_weights:
        sql = ("SELECT a.law, a.number FROM articles_fts f "
               "JOIN articles a ON a.id = f.rowid "
               "WHERE articles_fts MATCH ? "
               "ORDER BY bm25(articles_fts, 0.0, 5.0, 1.0, 10.0, 8.0) LIMIT ?")
    else:
        sql = ("SELECT a.law, a.number FROM articles_fts f "
               "JOIN articles a ON a.id = f.rowid "
               "WHERE articles_fts MATCH ? "
               "ORDER BY rank LIMIT ?")
    try:
        rows = db.execute(sql, (query, k)).fetchall()
    except Exception:
        return 0.0
    results = set(r[0] + ":" + r[1] for r in rows)
    hits = sum(1 for e in expected if e in results)
    return hits / len(expected) if expected else 0.0

def mrr(db, query, expected, use_weights=False):
    if use_weights:
        sql = ("SELECT a.law, a.number FROM articles_fts f "
               "JOIN articles a ON a.id = f.rowid "
               "WHERE articles_fts MATCH ? "
               "ORDER BY bm25(articles_fts, 0.0, 5.0, 1.0, 10.0, 8.0) LIMIT 10")
    else:
        sql = ("SELECT a.law, a.number FROM articles_fts f "
               "JOIN articles a ON a.id = f.rowid "
               "WHERE articles_fts MATCH ? "
               "ORDER BY rank LIMIT 10")
    try:
        rows = db.execute(sql, (query,)).fetchall()
    except Exception:
        return 0.0
    results = [r[0] + ":" + r[1] for r in rows]
    for exp in expected:
        if exp in results:
            return 1.0 / (results.index(exp) + 1)
    return 0.0

print("| Metric         | Baseline | Enriched | Improvement |")
print("|----------------|----------|----------|-------------|")

for metric_name, k in [("Recall@3", 3), ("Recall@5", 5), ("Recall@10", 10)]:
    base_scores = [recall_at_k(baseline, q, exp, k, False) for q, exp in TESTS]
    enr_scores = [recall_at_k(enriched, q, exp, k, True) for q, exp in TESTS]
    base_avg = sum(base_scores) / len(base_scores)
    enr_avg = sum(enr_scores) / len(enr_scores)
    imp = enr_avg - base_avg
    print(f"| {metric_name:14s} | {base_avg:7.1%}  | {enr_avg:7.1%}  | {imp:+7.1%}      |")

base_mrr = [mrr(baseline, q, exp, False) for q, exp in TESTS]
enr_mrr = [mrr(enriched, q, exp, True) for q, exp in TESTS]
base_avg_mrr = sum(base_mrr) / len(base_mrr)
enr_avg_mrr = sum(enr_mrr) / len(enr_mrr)
imp_mrr = enr_avg_mrr - base_avg_mrr
print(f"| {'MRR':14s} | {base_avg_mrr:7.1%}  | {enr_avg_mrr:7.1%}  | {imp_mrr:+7.1%}      |")

print("\n=== Per-query Recall@5 ===")
for i, (q, exp) in enumerate(TESTS):
    base = recall_at_k(baseline, q, exp, 5, False)
    enr = recall_at_k(enriched, q, exp, 5, True)
    marker = " ++" if enr > base else (" --" if enr < base else "")
    print(f"  {q:40s} base:{base:.0%} enr:{enr:.0%}{marker}")

baseline.close()
enriched.close()
