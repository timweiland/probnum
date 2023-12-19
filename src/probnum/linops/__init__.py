"""Finite-dimensional Linear Operators.

This package implements finite dimensional linear operators. It can be used to
represent linear maps between finite-dimensional vector spaces without explicitly
constructing their matrix representation in memory. This is particularly useful for
sparse and structured matrices and often allows for the definition of a more
efficient matrix-vector product. Linear operators support common algebraic
operations, including matrix-vector products, addition, multiplication, and
transposition.

Several algorithms in the :mod:`probnum.linalg` subpackage are able to operate on
:class:`~probnum.linops.LinearOperator` instances.
"""

from ._block import BlockDiagonalMatrix
from ._kronecker import IdentityKronecker, Kronecker, SymmetricKronecker, Symmetrize
from ._linear_operator import (
    Embedding,
    Identity,
    LambdaLinearOperator,
    LinearOperator,
    Matrix,
    Selection,
)
from ._permutation import Permutation
from ._scaling import Scaling, Zero
from ._utils import LinearOperatorLike, aslinop

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "aslinop",
    "Embedding",
    "LinearOperator",
    "LambdaLinearOperator",
    "Matrix",
    "Identity",
    "IdentityKronecker",
    "Permutation",
    "Scaling",
    "Kronecker",
    "Selection",
    "SymmetricKronecker",
    "Symmetrize",
    "BlockDiagonalMatrix",
    "Zero",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearOperator.__module__ = "probnum.linops"
LambdaLinearOperator.__module__ = "probnum.linops"
Embedding.__module__ = "probnum.linops"
Matrix.__module__ = "probnum.linops"
Identity.__module__ = "probnum.linops"
IdentityKronecker.__module__ = "probnum.linops"
Permutation.__module__ = "probnum.linops"
Scaling.__module__ = "probnum.linops"
Kronecker.__module__ = "probnum.linops"
Selection.__module__ = "probnum.linops"
SymmetricKronecker.__module__ = "probnum.linops"
Symmetrize.__module__ = "probnum.linops"
BlockDiagonalMatrix.__module__ = "probnum.linops"
Zero.__module__ = "probnum.linops"

aslinop.__module__ = "probnum.linops"
