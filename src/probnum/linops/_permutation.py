"""Operators that represent permutations."""
from __future__ import annotations

import functools
from typing import Callable, Optional

import numpy as np
from probnum.linops._linear_operator import LinearOperator

from probnum.typing import DTypeLike, LinearOperatorLike, ArrayLike
import torch

from . import _linear_operator, _utils


class Permutation(_linear_operator.LinearOperator):
    """Permutation operator.

    Parameters
    ----------
    perm :
        Permutation vector. The i-th entry of the vector indicates which row/column
        of the input is mapped to the i-th row/column of the output.
    """

    def __init__(self, perm: ArrayLike, dtype: DTypeLike = np.double):
        perm = np.asarray(perm, dtype=np.int64)
        assert perm.ndim == 1
        if not np.all(np.sort(perm) == np.arange(perm.size)):
            raise ValueError("The input does not represent a permutation.")

        shape = (perm.size, perm.size)
        super().__init__(shape, dtype)

        self._perm = perm

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return x[..., self._perm, :]

    def _matmul_torch(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self._perm, :]

    def _transpose(self) -> Permutation:
        perm_inv = np.empty_like(self._perm)
        perm_inv[self._perm] = np.arange(self._perm.size)
        return Permutation(perm_inv, dtype=self.dtype)

    def _inverse(self) -> LinearOperator:
        return self.T

    def _rank(self) -> np.intp:
        return self.shape[0]

    def _todense(self) -> np.ndarray:
        return np.eye(self.shape[0], dtype=self.dtype)[self._perm, :]

    def _trace(self) -> np.number:
        return np.sum(self._perm == np.arange(self._perm.size))

    def _det(self) -> np.number:
        visited = np.zeros_like(self._perm, dtype=np.bool_)
        cycle_sizes = []
        for i in range(self._perm.size):
            j = i
            cycle_size = 0
            while not visited[j]:
                visited[j] = True
                j = self._perm[j]
                cycle_size += 1
            if cycle_size > 0:
                cycle_sizes.append(cycle_size)
        cycle_sum = np.sum(np.array(cycle_sizes, dtype=np.int32) - 1)
        return (-1) ** cycle_sum
