from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from typing import Callable, Sequence

from src.meta_heuristicas import CooperativePSO, quadratic_penalty


@dataclass(frozen=True)
class ProblemDefinition:
    name: str
    objective: Callable[[Sequence[float]], float]
    bounds: tuple[tuple[float, float], ...]
    penalty_weight: float | None = None
    constraints: tuple[Callable[[Sequence[float]], float], ...] = ()


def tp1_problem_1_config_a(x: Sequence[float]) -> float:
    x1, x2, x3 = x
    return (x1 - 1.5) ** 2 + (x2 + 2.0) ** 2 + (x3 - 0.5) ** 2


def tp2_problem_1(x: Sequence[float]) -> float:
    x1, x2 = x
    return (x1 - 2.0) ** 2 + (x2 - 1.0) ** 2


def tp2_constraints() -> tuple[Callable[[Sequence[float]], float], ...]:
    return (
        lambda x: x[0] + x[1] - 2.2,
        lambda x: -x[0],
        lambda x: -x[1],
    )


PROBLEMS = (
    ProblemDefinition(
        name="tp1_problema1_config_a",
        objective=tp1_problem_1_config_a,
        bounds=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)),
    ),
    ProblemDefinition(
        name="tp2_problema1_com_restricao",
        objective=tp2_problem_1,
        bounds=((0.0, 5.0), (0.0, 5.0)),
        penalty_weight=1000.0,
        constraints=tp2_constraints(),
    ),
)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _quantiles(values: list[float]) -> tuple[float, float, float, float, float]:
    ordered = sorted(values)
    q1, q2, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return ordered[0], q1, q2, q3, ordered[-1]


def write_boxplot_svg(values: list[float], title: str, output_file: Path) -> None:
    vmin, q1, median, q3, vmax = _quantiles(values)
    width, height = 520, 180
    margin_x, mid_y = 40, 90
    scale = (width - 2 * margin_x) / (vmax - vmin if vmax != vmin else 1.0)

    def x_pos(v: float) -> float:
        return margin_x + (v - vmin) * scale

    box_left = x_pos(q1)
    box_right = x_pos(q3)
    med_x = x_pos(median)
    min_x = x_pos(vmin)
    max_x = x_pos(vmax)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">{title}</text>
<line x1="{min_x}" y1="{mid_y}" x2="{box_left}" y2="{mid_y}" stroke="black"/>
<line x1="{box_right}" y1="{mid_y}" x2="{max_x}" y2="{mid_y}" stroke="black"/>
<line x1="{min_x}" y1="{mid_y-18}" x2="{min_x}" y2="{mid_y+18}" stroke="black"/>
<line x1="{max_x}" y1="{mid_y-18}" x2="{max_x}" y2="{mid_y+18}" stroke="black"/>
<rect x="{box_left}" y="{mid_y-24}" width="{max(2.0, box_right-box_left)}" height="48" fill="#e6f0ff" stroke="black"/>
<line x1="{med_x}" y1="{mid_y-24}" x2="{med_x}" y2="{mid_y+24}" stroke="black"/>
<text x="{margin_x}" y="{height-14}" text-anchor="start" font-family="monospace" font-size="12">min={vmin:.6f}</text>
<text x="{width-margin_x}" y="{height-14}" text-anchor="end" font-family="monospace" font-size="12">max={vmax:.6f}</text>
</svg>
"""
    output_file.write_text(svg, encoding="utf-8")


def run_experiments(
    runs: int = 30,
    output_dir: Path | None = None,
    seed_base: int = 2026,
) -> dict[str, dict[str, object]]:
    out = output_dir or Path("results")
    boxplot_dir = out / "boxplots"
    boxplot_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, object]] = {}

    for p_idx, problem in enumerate(PROBLEMS):
        scores: list[float] = []
        run_data: list[dict[str, object]] = []
        best_run: dict[str, object] | None = None
        penalty = (
            quadratic_penalty(problem.constraints, problem.penalty_weight or 0.0)
            if problem.constraints
            else None
        )

        for run in range(runs):
            seed = seed_base + 1000 * p_idx + run
            optimizer = CooperativePSO(
                objective=problem.objective,
                bounds=problem.bounds,
                penalty=penalty,
            )
            result = optimizer.optimize(seed=seed)
            run_entry = {
                "run": run + 1,
                "score": result.score,
                "position": list(result.position),
            }
            run_data.append(run_entry)
            scores.append(result.score)
            if best_run is None or result.score < float(best_run["score"]):
                best_run = run_entry

        assert best_run is not None
        stats = _stats(scores)
        write_boxplot_svg(
            values=scores,
            title=problem.name,
            output_file=boxplot_dir / f"{problem.name}.svg",
        )
        summary[problem.name] = {
            "stats": stats,
            "best_solution": best_run,
            "runs": run_data,
            "problem": {
                "name": problem.name,
                "bounds": [list(b) for b in problem.bounds],
                "constraints_count": len(problem.constraints),
                "penalty_weight": problem.penalty_weight,
            },
        }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
