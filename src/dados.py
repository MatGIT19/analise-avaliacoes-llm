"""Carga e preparação do dataset público da Olist."""

from pathlib import Path

import pandas as pd

from .config import DATA_DIR

ARQUIVOS = {
    "avaliacoes": "olist_order_reviews_dataset.csv",
    "pedidos": "olist_orders_dataset.csv",
    "clientes": "olist_customers_dataset.csv",
}

# Cópia pública mantida pela própria Olist no GitHub (mesmos arquivos do Kaggle)
URL_BASE = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"


def baixar(pasta: Path = DATA_DIR) -> None:
    """Baixa os três arquivos usados pelo projeto (cerca de 40 MB), se ainda não existirem."""
    import urllib.request

    pasta.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS.values():
        destino = pasta / nome
        if destino.exists():
            continue
        print(f"Baixando {nome}...")
        urllib.request.urlretrieve(URL_BASE + nome, destino)


def _ler(pasta: Path, chave: str) -> pd.DataFrame:
    caminho = pasta / ARQUIVOS[chave]
    if not caminho.exists():
        raise FileNotFoundError(
            f"Não encontrei {caminho}. Rode `python scripts/baixar_dados.py` ou veja data/README.md."
        )
    return pd.read_csv(caminho)


def carregar(pasta: Path = DATA_DIR) -> pd.DataFrame:
    """Uma linha por avaliação COM comentário, já com estado do cliente e dados de entrega.

    Colunas: review_id, nota, comentario, data, mes, estado, dias_de_atraso, atrasou.
    `dias_de_atraso` = entrega real - entrega prevista (negativo = chegou antes).
    """
    avaliacoes = _ler(pasta, "avaliacoes")
    pedidos = _ler(pasta, "pedidos")
    clientes = _ler(pasta, "clientes")

    titulo = avaliacoes["review_comment_title"].fillna("").astype(str).str.strip()
    mensagem = avaliacoes["review_comment_message"].fillna("").astype(str).str.strip()
    comentario = (titulo + ". " + mensagem).where(titulo != "", mensagem)
    comentario = comentario.str.replace(r"\s+", " ", regex=True).str.strip(" .")

    df = pd.DataFrame(
        {
            "review_id": avaliacoes["review_id"],
            "order_id": avaliacoes["order_id"],
            "nota": avaliacoes["review_score"].astype(int),
            "comentario": comentario,
            "data": pd.to_datetime(avaliacoes["review_creation_date"]),
        }
    )
    df = df[df["comentario"].str.len() >= 3]

    pedidos = pedidos.merge(clientes[["customer_id", "customer_state"]], on="customer_id", how="left")
    entregue = pd.to_datetime(pedidos["order_delivered_customer_date"])
    previsto = pd.to_datetime(pedidos["order_estimated_delivery_date"])
    pedidos["dias_de_atraso"] = (entregue - previsto).dt.days
    df = df.merge(
        pedidos[["order_id", "customer_state", "dias_de_atraso"]], on="order_id", how="left"
    ).rename(columns={"customer_state": "estado"})

    df["mes"] = df["data"].dt.to_period("M").astype(str)
    df["atrasou"] = df["dias_de_atraso"] > 0
    # Uma avaliação pode repetir o mesmo review_id em pedidos diferentes: fica a primeira
    return df.drop_duplicates("review_id").drop(columns="order_id").reset_index(drop=True)


def rotular_sentimento(df: pd.DataFrame) -> pd.DataFrame:
    """Rótulo binário a partir da nota: 1-2 = negativa, 4-5 = positiva. Nota 3 sai (ambígua)."""
    out = df[df["nota"] != 3].copy()
    out["sentimento"] = out["nota"].map(lambda n: "negativa" if n <= 2 else "positiva")
    return out.reset_index(drop=True)


def amostra_balanceada(df: pd.DataFrame, n: int, coluna: str = "sentimento", seed: int = 42) -> pd.DataFrame:
    """n linhas com metade de cada classe, para a comparação não favorecer o chute "positiva"."""
    grupos = [g.sample(min(len(g), n // 2), random_state=seed) for _, g in df.groupby(coluna)]
    return pd.concat(grupos).sample(frac=1, random_state=seed).reset_index(drop=True)
