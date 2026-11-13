"""Small, dependency-free dense linear algebra helpers.

The analysis only ever solves systems with a handful of columns, so a
partial-pivoting Gaussian elimination is sufficient.  Functions return
``None`` instead of raising when a design is numerically singular; callers
turn that into an explicit evidence state rather than a crash.
"""

import math


def solve(matrix, rhs):
    """Solve ``matrix @ x = rhs`` with partial pivoting.

    Returns ``None`` when the system is singular to working precision.
    """
    size = len(rhs)
    if size == 0 or len(matrix) != size:
        return None
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    magnitude = max((abs(value) for row in matrix for value in row), default=0.0)
    if magnitude == 0.0 or not math.isfinite(magnitude):
        return None
    tolerance = 1e-12 * magnitude * max(size, 1)
    for column in range(size):
        pivot = column
        best = abs(augmented[column][column])
        for row in range(column + 1, size):
            candidate = abs(augmented[row][column])
            if candidate > best:
                best = candidate
                pivot = row
        if best <= tolerance:
            return None
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / pivot_value
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    result = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = augmented[row][size]
        for column in range(row + 1, size):
            total -= augmented[row][column] * result[column]
        diagonal = augmented[row][row]
        if diagonal == 0.0:
            return None
        result[row] = total / diagonal
    return result


def inverse(matrix):
    """Invert a square matrix, or return ``None`` if it is singular."""
    size = len(matrix)
    if size == 0:
        return None
    columns = []
    for index in range(size):
        unit = [1.0 if position == index else 0.0 for position in range(size)]
        column = solve(matrix, unit)
        if column is None:
            return None
        columns.append(column)
    return [[columns[column][row] for column in range(size)] for row in range(size)]


def independent_columns(columns, tolerance=1e-8):
    """Return indices of columns that are linearly independent.

    Modified Gram-Schmidt is used in the order given, so earlier columns win
    when two are collinear.  This lets the caller keep known effects (an
    intercept, temperature, a maintenance jump) ahead of estimated change
    points, which are the effects we are willing to drop.
    """
    if not columns:
        return []
    length = len(columns[0])
    scale = max((abs(value) for column in columns for value in column), default=0.0)
    if scale == 0.0:
        return []
    basis = []
    kept = []
    for index, column in enumerate(columns):
        vector = [float(value) for value in column]
        original = math.sqrt(math.fsum(value * value for value in vector))
        if original <= 1e-12 * scale:
            continue
        for direction in basis:
            projection = math.fsum(vector[pos] * direction[pos] for pos in range(length))
            if projection:
                vector = [vector[pos] - projection * direction[pos] for pos in range(length)]
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm <= tolerance * original:
            continue
        basis.append([value / norm for value in vector])
        kept.append(index)
    return kept
