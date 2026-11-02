"""Classifica com o LLM o motivo de uma amostra de avaliações negativas.

Uso: python scripts/classificar_motivos.py --n 1000   (precisa de GEMINI_API_KEY)
Saída: resultados/motivos.csv, usado pelo dashboard.

Validação embutida: entre as avaliações cujo pedido de fato chegou atrasado,
quantas o LLM classificou como "atraso_na_entrega"? Isso confere o LLM contra
um dado objetivo que ele não viu (as datas de entrega).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.dados import carregar  # noqa: E402
from src.llm import ClassificadorLLM, criar_cliente  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="quantas avaliações negativas classificar")
    args = parser.parse_args()

    negativas = carregar().query("nota <= 2")
    amostra = negativas.sample(min(args.n, len(negativas)), random_state=42).reset_index(drop=True)
    print(f"Classificando o motivo de {len(amostra)} avaliações negativas com {config.LLM_MODEL}...")

    llm = ClassificadorLLM(criar_cliente(), cache=config.RESULTADOS_DIR / "cache_llm.jsonl")
    respostas = llm.classificar(amostra["review_id"], amostra["comentario"])
    amostra["motivo"] = [r["motivo"] for r in respostas]
    amostra["sentimento_llm"] = [r["sentimento"] for r in respostas]

    colunas = ["review_id", "nota", "estado", "mes", "dias_de_atraso", "atrasou", "sentimento_llm", "motivo", "comentario"]
    config.RESULTADOS_DIR.mkdir(exist_ok=True)
    amostra[colunas].to_csv(config.RESULTADOS_DIR / "motivos.csv", index=False)

    print("\nMotivos encontrados:")
    print(amostra["motivo"].value_counts(normalize=True).mul(100).round(1).to_string())
    atrasados = amostra[amostra["atrasou"]]
    if len(atrasados):
        taxa = (atrasados["motivo"].isin(["atraso_na_entrega", "nao_recebido"])).mean()
        print(
            f"\nValidação: dos {len(atrasados)} pedidos que de fato atrasaram, o LLM apontou atraso "
            f"ou não recebimento em {taxa:.0%}."
        )
    print("Salvo em resultados/motivos.csv")


if __name__ == "__main__":
    main()
