"""Porta da recompensa: o treino depende só daqui; cada linguagem tem seu adaptador.

A recompensa tem três dimensões (ver docs/grpo.md):

* correctness, -1, 0 ou 1: 1 se passa em todos os testes, 0 se falha algum ou usa um
  nome proibido, -1 se não compila (sem nota parcial);
* readability, de 0 a 1, só se correctness = 1;
* efficiency, de 0 a 1, só se correctness = 1.
"""

import importlib
from dataclasses import dataclass, field
from typing import Protocol

ADAPTERS = {"python": "train.rl.reward.python:PythonReward"}


@dataclass
class Scores:
    correctness: float
    readability: float
    efficiency: float
    details: dict = field(default_factory=dict)


@dataclass
class Weights:
    correctness: float = 1.0
    readability: float = 0.2
    efficiency: float = 0.2


class RewardModel(Protocol):
    def score(self, code: str | None, exercise: dict) -> Scores:
        """``exercise``: ``id``, ``tests``, ``name`` (None em nível de módulo), ``response`` (a solução) e ``prompt``."""
        ...

    def state(self) -> dict:
        """O que precisa ir no checkpoint (as referências da efficiency)."""
        ...

    def load(self, state: dict) -> None:
        ...


def combine(scores: Scores, weights: Weights) -> float:
    return (weights.correctness * scores.correctness + weights.readability * scores.readability
            + weights.efficiency * scores.efficiency)


def reward_for(lang: str) -> RewardModel:
    if lang not in ADAPTERS:
        raise ValueError(f"sem adaptador de recompensa para {lang!r} (disponíveis: {', '.join(ADAPTERS)})")
    module, name = ADAPTERS[lang].split(":")
    return getattr(importlib.import_module(module), name)()
