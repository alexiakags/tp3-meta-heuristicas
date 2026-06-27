from pathlib import Path

from src.experiments import run_experiments


def main() -> None:
    run_experiments(runs=30, output_dir=Path("results"))


if __name__ == "__main__":
    main()
