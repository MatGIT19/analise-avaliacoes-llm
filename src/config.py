"""Caminhos e configurações, lidos de variáveis de ambiente ou do .env."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTADOS_DIR = ROOT / "resultados"

try:  # .env é opcional
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

# Os nomes dos modelos mudam com frequência: troque no .env sem mexer no código.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")

# Preço por 1 milhão de tokens (US$), para estimar quanto custaria fora da camada
# gratuita. Confira a tabela atual do modelo escolhido e ajuste no .env.
PRECO_ENTRADA_1M = float(os.getenv("PRECO_ENTRADA_1M", "0.75"))
PRECO_SAIDA_1M = float(os.getenv("PRECO_SAIDA_1M", "3.75"))

# Quantas avaliações vão em cada chamada ao LLM e a pausa entre chamadas,
# para caber nos limites por minuto da camada gratuita.
LOTE_LLM = int(os.getenv("LOTE_LLM", "20"))
PAUSA_ENTRE_CHAMADAS_S = float(os.getenv("PAUSA_ENTRE_CHAMADAS_S", "4"))
