"""Compara o modelo clássico com o LLM na mesma amostra de avaliações.

Uso:
  python scripts/comparar.py                 # clássico + LLM (precisa de GEMINI_API_KEY)
  python scripts/comparar.py --sem-llm       # só o modelo clássico
  python scripts/comparar.py --amostra 200   # amostra menor, menos chamadas à API

Saídas em resultados/: metricas.json, comparacao.md, predicoes_amostra.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src import config  # noqa: E402
from src.classico import palavras_mais_fortes, prever, treinar  # noqa: E402
from src.dados import amostra_balanceada, carregar, rotular_sentimento  # noqa: E402
from src.metricas import resumo  # noqa: E402

SEED = 42


def _linha(nome: str, m: dict, ms_por_aval: float | None, custo_1000: str) -> str:
    tempo = f"{ms_por_aval * 1000 / 1000:.2f} s" if ms_por_aval is not None else "-"
    return (
        f"| {nome} | {m['acuracia']:.1%} | {m['f1_macro']:.3f} | {m['recall_negativa']:.1%} | "
        f"{m['precisao_negativa']:.1%} | {tempo} | {custo_1000} |"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amostra", type=int, default=400, help="avaliações na comparação (metade de cada classe)")
    parser.add_argument("--sem-llm", action="store_true", help="roda só o modelo clássico")
    args = parser.parse_args()
    config.RESULTADOS_DIR.mkdir(exist_ok=True)

    print("Carregando dados...")
    todas = carregar()
    dados = rotular_sentimento(todas)
    treino, teste = train_test_split(dados, test_size=0.2, stratify=dados["sentimento"], random_state=SEED)
    print(f"{len(treino)} avaliações para treino, {len(teste)} para teste")

    print("Treinando TF-IDF + regressão logística...")
    modelo, seg_treino = treinar(treino["comentario"], treino["sentimento"])
    prev_teste, _ = prever(modelo, teste["comentario"])

    amostra = amostra_balanceada(teste, args.amostra, seed=SEED)
    prev_classico, ms_classico = prever(modelo, amostra["comentario"])

    metricas = {
        "dados": {
            "avaliacoes_com_comentario": len(todas),
            "treino": len(treino),
            "teste": len(teste),
            "amostra_comparacao": len(amostra),
        },
        "classico": {
            "teste_completo": resumo(teste["sentimento"], prev_teste),
            "amostra": resumo(amostra["sentimento"], prev_classico),
            "segundos_treino": round(seg_treino, 2),
            "ms_por_avaliacao": round(ms_classico, 4),
        },
    }
    saida = amostra[["review_id", "nota", "comentario", "sentimento"]].rename(columns={"sentimento": "real"})
    saida["classico"] = prev_classico

    linhas_tabela = [
        _linha(
            "TF-IDF + regressão logística",
            metricas["classico"]["amostra"],
            ms_classico,
            "~US$ 0 (roda na CPU)",
        )
    ]

    if not args.sem_llm:
        from src.llm import ClassificadorLLM, criar_cliente

        print(f"Classificando {len(amostra)} avaliações com {config.LLM_MODEL}...")
        llm = ClassificadorLLM(criar_cliente(), cache=config.RESULTADOS_DIR / "cache_llm.jsonl")
        respostas = llm.classificar(amostra["review_id"], amostra["comentario"])
        saida["llm"] = [r["sentimento"] for r in respostas]
        saida["llm_motivo"] = [r["motivo"] for r in respostas]
        m_llm = resumo(saida["real"], saida["llm"])
        u = llm.uso
        ms_llm = u.segundos * 1000 / len(amostra) if u.chamadas else None
        custo_1000 = u.custo_estimado_usd() / len(amostra) * 1000 if u.chamadas else None
        metricas["llm"] = {
            "modelo": config.LLM_MODEL,
            "amostra": m_llm,
            "chamadas": u.chamadas,
            "tokens_entrada": u.tokens_entrada,
            "tokens_saida": u.tokens_saida,
            "ms_por_avaliacao": round(ms_llm, 1) if ms_llm else None,
            "custo_estimado_usd_por_1000": round(custo_1000, 4) if custo_1000 is not None else None,
            "erros_de_formato": u.erros_de_formato,
        }
        linhas_tabela.append(
            _linha(
                f"LLM ({config.LLM_MODEL})",
                m_llm,
                ms_llm,
                f"~US$ {custo_1000:.3f}" if custo_1000 is not None else "- (resultado do cache)",
            )
        )

    metricas["classico"]["palavras_mais_fortes"] = palavras_mais_fortes(modelo)
    (config.RESULTADOS_DIR / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    saida.to_csv(config.RESULTADOS_DIR / "predicoes_amostra.csv", index=False)

    md = [
        "# Comparação: modelo clássico vs LLM",
        "",
        f"Amostra de **{len(amostra)} avaliações** do conjunto de teste, metade negativas e metade positivas. "
        "Rótulo real = nota do cliente (1-2 negativa, 4-5 positiva).",
        "",
        "| Modelo | Acurácia | F1 macro | Recall negativas | Precisão negativas | Tempo por 1.000 | Custo por 1.000 |",
        "|---|---|---|---|---|---|---|",
        *linhas_tabela,
        "",
        f"No conjunto de teste completo ({len(teste)} avaliações, proporção real de classes), o modelo clássico "
        f"teve acurácia de {metricas['classico']['teste_completo']['acuracia']:.1%} "
        f"e F1 macro de {metricas['classico']['teste_completo']['f1_macro']:.3f}.",
    ]
    if "llm" in saida:
        discordam = saida[saida["classico"] != saida["llm"]]
        md += ["", f"## Onde os dois discordam ({len(discordam)} casos, primeiros 10)", "",
               "| Nota | Real | Clássico | LLM | Comentário |", "|---|---|---|---|---|"]
        for _, r in discordam.head(10).iterrows():
            texto = r["comentario"][:140].replace("|", "/")
            md.append(f"| {r['nota']} | {r['real']} | {r['classico']} | {r['llm']} | {texto} |")
    (config.RESULTADOS_DIR / "comparacao.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md[4 : 6 + len(linhas_tabela)]))
    print("\nResultados em resultados/: metricas.json, comparacao.md, predicoes_amostra.csv")


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
