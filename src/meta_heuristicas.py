from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Sequence


ObjectiveFunction = Callable[[Sequence[float]], float]
PenaltyFunction = Callable[[Sequence[float]], float]


@dataclass(frozen=True)
class OptimizationResult:
    position: tuple[float, ...]
    score: float


class CooperativePSO:
    def __init__(
        self,
        objective: ObjectiveFunction,
        bounds: Sequence[tuple[float, float]],
        penalty: PenaltyFunction | None = None,
        particles: int = 16,
        iterations: int = 200,
        w: float = 0.72,
        c1: float = 1.45,
        c2: float = 1.45,
    ) -> None:
        if particles < 2:
            raise ValueError("particles must be >= 2")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.objective = objective
        self.penalty = penalty or (lambda _: 0.0)
        self.bounds = tuple(bounds)
        self.dimensions = len(bounds)
        self.particles = particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _bounded(self, x: float, dim: int) -> float:
        lo, hi = self.bounds[dim]
        return max(lo, min(hi, x))

    def _fitness(self, x: Sequence[float]) -> float:
        return self.objective(x) + self.penalty(x)

    def optimize(self, seed: int | None = None) -> OptimizationResult:
        rng = random.Random(seed)
        context = [rng.uniform(lo, hi) for lo, hi in self.bounds]
        positions: list[list[float]] = []
        velocities: list[list[float]] = []
        pbest_positions: list[list[float]] = []
        pbest_scores: list[list[float]] = []
        gbest_positions: list[float] = []

        for dim, (lo, hi) in enumerate(self.bounds):
            span = hi - lo
            dim_positions = [rng.uniform(lo, hi) for _ in range(self.particles)]
            dim_velocities = [rng.uniform(-0.1 * span, 0.1 * span) for _ in range(self.particles)]
            dim_scores = []
            for value in dim_positions:
                candidate = list(context)
                candidate[dim] = value
                dim_scores.append(self._fitness(candidate))
            best_idx = min(range(self.particles), key=lambda i: dim_scores[i])
            positions.append(dim_positions)
            velocities.append(dim_velocities)
            pbest_positions.append(dim_positions.copy())
            pbest_scores.append(dim_scores.copy())
            gbest_positions.append(dim_positions[best_idx])

        for dim in range(self.dimensions):
            context[dim] = gbest_positions[dim]

        for _ in range(self.iterations):
            for dim in range(self.dimensions):
                for i in range(self.particles):
                    rp = rng.random()
                    rg = rng.random()
                    v = (
                        self.w * velocities[dim][i]
                        + self.c1 * rp * (pbest_positions[dim][i] - positions[dim][i])
                        + self.c2 * rg * (gbest_positions[dim] - positions[dim][i])
                    )
                    x = self._bounded(positions[dim][i] + v, dim)
                    velocities[dim][i] = v
                    positions[dim][i] = x

                    candidate = list(context)
                    candidate[dim] = x
                    score = self._fitness(candidate)
                    if score < pbest_scores[dim][i]:
                        pbest_scores[dim][i] = score
                        pbest_positions[dim][i] = x

                best_idx = min(range(self.particles), key=lambda i: pbest_scores[dim][i])
                gbest_positions[dim] = pbest_positions[dim][best_idx]
                context[dim] = gbest_positions[dim]

        best_position = tuple(context)
        best_score = self.objective(best_position)
        return OptimizationResult(position=best_position, score=best_score)


def positive_constraint_violation(value: float) -> float:
    return max(0.0, value)


def quadratic_penalty(
    constraints: Sequence[Callable[[Sequence[float]], float]],
    weight: float = 1000.0,
) -> PenaltyFunction:
    def penalty(x: Sequence[float]) -> float:
        return weight * sum(positive_constraint_violation(c(x)) ** 2 for c in constraints)

    return penalty
