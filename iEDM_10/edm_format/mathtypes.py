"""
Simple math types to represent Vector, Matrix, Quaternion

If the blender mathutils module is available, the blender API types are
used, but otherwise a (minimally) compatible internal type is used instead.

Also included are tools to create from simple lists, and convert between the
EDM and blender axis interpretations.
"""

from __future__ import annotations

import itertools
from typing import Any, Sequence

# mathutils (Blender) ships no type stubs, and the fallback classes below are
# plain tuples used interchangeably with it; there is no meaningful static
# type to give these beyond "whichever of the two is active at runtime".
MatrixLike = Any
VectorLike = Any
QuaternionLike = Any

try:
    from mathutils import Matrix, Quaternion, Vector
except ImportError:
    # We don't have mathutils. Make some very basic replacements.
    class Vector(tuple[Any, ...]):  # type: ignore[no-redef]
        def __repr__(self) -> str:
            return "Vector({})".format(super(Vector, self).__repr__())

    class Matrix(tuple[Any, ...]):  # type: ignore[no-redef]
        def transposed(self) -> "Matrix":
            cols = [[self[j][i] for j in range(len(self))] for i in range(len(self))]
            return Matrix(cols)

        def __repr__(self) -> str:
            return "Matrix({})".format(super(Matrix, self).__repr__())

    class Quaternion(tuple[Any, ...]):  # type: ignore[no-redef]
        def __repr__(self) -> str:
            return "Quaternion({})".format(super(Quaternion, self).__repr__())


__all__ = [
    "Matrix",
    "Vector",
    "Quaternion",
    "MatrixScale",
    "sequence_to_matrix",
    "matrix_to_sequence",
    "sequence_to_quaternion",
    "matrix_to_blender",
    "world_matrix_to_blender",
    "quaternion_to_blender",
    "vector_to_blender",
]


def MatrixScale(vector: Sequence[float]) -> MatrixLike:
    mat = Matrix.Scale(1, 4)
    mat[0][0], mat[1][1], mat[2][2] = vector[:3]
    return mat


def sequence_to_matrix(seq: Sequence[float]) -> MatrixLike:
    return Matrix([seq[:4], seq[4:8], seq[8:12], seq[12:16]]).transposed()


def matrix_to_sequence(mat: MatrixLike) -> tuple[float, ...]:
    xp = mat.transposed()
    return tuple(itertools.chain(xp[0], xp[1], xp[2], xp[3]))


def sequence_to_quaternion(seq: Sequence[float]) -> QuaternionLike:
    return Quaternion((seq[3], seq[0], seq[1], seq[2]))


# Coordinate conversion matrices used for EDM matrix/quaternion transforms.
# EDM stores matrices/quaternions in Y-up space; Blender uses Z-up.
# _R converts a point from EDM (Y-up) to Blender (Z-up).
# _R_inv converts a point from Blender (Z-up) to EDM (Y-up).
# Note: EDM v10 position vectors are already stored in Blender's Z-up axes,
# so vector_to_blender is a passthrough. The matrix/quaternion helpers below
# still perform the basis-change for rotation data.
_R = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))  # Y-up -> Z-up
_R_inv = Matrix(
    ((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1))
)  # Z-up -> Y-up


def matrix_to_blender(matrix: MatrixLike) -> MatrixLike:
    """Converts a LOCAL matrix (basis change)."""
    return _R @ matrix @ _R_inv


def world_matrix_to_blender(matrix: MatrixLike) -> MatrixLike:
    """Converts a WORLD matrix (pre-multiplied global swap)."""
    return _R @ matrix


def quaternion_to_blender(q: QuaternionLike) -> QuaternionLike:
    """Convert quaternion from EDM Y-up to Blender Z-up."""
    return (_R @ q.to_matrix().to_4x4() @ _R_inv).to_quaternion()


def vector_to_blender(v: Sequence[float]) -> VectorLike:
    """EDM v10 position vectors already use Blender's Z-up axes."""
    return Vector(v)
