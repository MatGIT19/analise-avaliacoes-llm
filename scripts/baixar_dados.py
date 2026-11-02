"""Baixa os arquivos do dataset da Olist para a pasta data/.  Uso: python scripts/baixar_dados.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dados import baixar  # noqa: E402

if __name__ == "__main__":
    baixar()
    print("Pronto: arquivos em data/")
