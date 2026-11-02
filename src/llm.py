"""Classificação de avaliações com um LLM (Gemini), com saída estruturada em JSON.

Cada chamada manda um LOTE de avaliações numeradas e pede de volta uma lista
JSON com o sentimento e o motivo de cada uma. Lotes reduzem o número de
requisições (importante na camada gratuita) e o JSON com esquema evita ter
de "interpretar" texto livre.

Os resultados ficam num cache em disco: se o limite da API estourar no meio,
basta rodar de novo que ele continua de onde parou.
"""

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, TypeAdapter, ValidationError

from . import config

Sentimento = Literal["negativa", "positiva"]
Motivo = Literal[
    "atraso_na_entrega",
    "nao_recebido",
    "produto_errado_ou_incompleto",
    "produto_com_defeito",
    "qualidade_abaixo_do_esperado",
    "atendimento_do_vendedor",
    "outro",
    "sem_reclamacao",
]
MOTIVOS: list[str] = list(Motivo.__args__)

VERSAO_PROMPT = "v1"  # mude ao alterar o prompt, para não reaproveitar cache antigo

INSTRUCOES = """Você classifica avaliações de clientes de um e-commerce brasileiro.
Para cada avaliação numerada, devolva:
- sentimento: "negativa" se o cliente está insatisfeito, "positiva" se está satisfeito.
- motivo: o principal motivo da reclamação. Use "sem_reclamacao" quando não houver reclamação.
  atraso_na_entrega = chegou depois do prazo ou está demorando
  nao_recebido = o pedido (ou parte dele) não chegou
  produto_errado_ou_incompleto = veio outro produto, cor, tamanho ou faltando itens
  produto_com_defeito = quebrado, com defeito ou danificado
  qualidade_abaixo_do_esperado = diferente do anúncio ou de qualidade ruim
  atendimento_do_vendedor = problema de comunicação, troca, reembolso ou cobrança
  outro = reclamação que não se encaixa acima
Exemplos:
"Chegou rápido e bem embalado, recomendo" -> positiva, sem_reclamacao
"Até agora não recebi o produto e já passou do prazo" -> negativa, nao_recebido
"Comprei dois e veio só um" -> negativa, produto_errado_ou_incompleto
"Gostei do produto, mas demorou demais pra chegar" -> negativa, atraso_na_entrega
Responda só com o JSON pedido, com um item para cada número recebido."""


class Classificacao(BaseModel):
    id: int
    sentimento: Sentimento
    motivo: Motivo


LISTA = TypeAdapter(list[Classificacao])


@dataclass
class Uso:
    chamadas: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    segundos: float = 0.0
    erros_de_formato: int = 0

    def custo_estimado_usd(self) -> float:
        return (
            self.tokens_entrada * config.PRECO_ENTRADA_1M + self.tokens_saida * config.PRECO_SAIDA_1M
        ) / 1_000_000


@dataclass
class ClassificadorLLM:
    cliente: object
    modelo: str = config.LLM_MODEL
    lote: int = config.LOTE_LLM
    pausa_s: float = config.PAUSA_ENTRE_CHAMADAS_S
    cache: Path | None = None
    max_tentativas: int = 5
    uso: Uso = field(default_factory=Uso)

    def __post_init__(self):
        self._cache: dict[str, dict] = {}
        if self.cache and self.cache.exists():
            for linha in self.cache.read_text(encoding="utf-8").splitlines():
                item = json.loads(linha)
                self._cache[item["chave"]] = item

    def _chave(self, review_id: str) -> str:
        return f"{self.modelo}|{VERSAO_PROMPT}|{review_id}"

    def _salvar(self, itens: list[dict]) -> None:
        for item in itens:
            self._cache[item["chave"]] = item
        if self.cache:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            with self.cache.open("a", encoding="utf-8") as f:
                for item in itens:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _chamar(self, textos: list[str]):
        from google.genai import types

        conteudo = "\n".join(f"{i}. {t[:1000]}" for i, t in enumerate(textos))
        configuracao = types.GenerateContentConfig(
            system_instruction=INSTRUCOES,
            response_mime_type="application/json",
            response_schema=list[Classificacao],
            temperature=0,
        )
        espera = 10.0
        for tentativa in range(1, self.max_tentativas + 1):
            try:
                return self.cliente.models.generate_content(
                    model=self.modelo, contents=conteudo, config=configuracao
                )
            except Exception as erro:  # limite de requisições ou instabilidade: tenta de novo
                codigo = getattr(erro, "code", None)
                if codigo not in (429, 500, 503) or tentativa == self.max_tentativas:
                    raise
                print(f"  API respondeu {codigo}; nova tentativa em {espera:.0f} s")
                time.sleep(espera)
                espera *= 2

    def _classificar_lote(self, ids: list[str], textos: list[str]) -> list[dict]:
        inicio = time.perf_counter()
        resposta = self._chamar(textos)
        self.uso.segundos += time.perf_counter() - inicio
        self.uso.chamadas += 1
        meta = getattr(resposta, "usage_metadata", None)
        if meta is not None:
            self.uso.tokens_entrada += getattr(meta, "prompt_token_count", 0) or 0
            self.uso.tokens_saida += getattr(meta, "candidates_token_count", 0) or 0

        try:
            itens = LISTA.validate_json(resposta.text or "[]")
        except ValidationError:
            itens = []
            self.uso.erros_de_formato += 1
        por_posicao = {c.id: c for c in itens}

        resultados = []
        for posicao, review_id in enumerate(ids):
            c = por_posicao.get(posicao)
            resultados.append(
                {
                    "chave": self._chave(review_id),
                    "review_id": review_id,
                    "sentimento": c.sentimento if c else "invalido",
                    "motivo": c.motivo if c else "invalido",
                }
            )
        return resultados

    def classificar(self, ids: Iterable[str], textos: Iterable[str]) -> list[dict]:
        """Devolve, na mesma ordem, {review_id, sentimento, motivo} para cada avaliação."""
        ids, textos = list(ids), list(textos)
        pendentes = [(i, t) for i, t in zip(ids, textos) if self._chave(i) not in self._cache]
        total_lotes = -(-len(pendentes) // self.lote)
        for n, inicio in enumerate(range(0, len(pendentes), self.lote), 1):
            bloco = pendentes[inicio : inicio + self.lote]
            print(f"  lote {n}/{total_lotes} ({len(bloco)} avaliações)")
            self._salvar(self._classificar_lote([i for i, _ in bloco], [t for _, t in bloco]))
            if n < total_lotes:
                time.sleep(self.pausa_s)
        return [
            {k: self._cache[self._chave(i)][k] for k in ("review_id", "sentimento", "motivo")}
            for i in ids
        ]


def criar_cliente():
    from google import genai

    if not config.API_KEY:
        raise RuntimeError("Defina GEMINI_API_KEY no .env (crie em https://aistudio.google.com/apikey).")
    return genai.Client(api_key=config.API_KEY)
