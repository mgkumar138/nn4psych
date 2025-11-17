#!/usr/bin/env python3
"""
Master Pipeline Runner

Executes all data processing stages in sequence.
"""

import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent

# Define pipeline stages with scripts and descriptions
PIPELINE_STAGES = [
    {
        'stage': '01 - BEHAVIOR EXTRACTION',
        'scripts': [
            ('01_extract_model_behavior.py', 'Extract behavioral data from trained models'),
        ]
    },
    {
        'stage': '02 - METRICS COMPUTATION',
        'scripts': [
            ('02_compute_learning_metrics.py', 'Compute learning rates and performance metrics'),
        ]
    },
    {
        'stage': '03 - HYPERPARAMETER ANALYSIS',
        'scripts': [
            ('03_analyze_hyperparameter_sweeps.py', 'Analyze hyperparameter sweep results'),
        ]
    },
]


def run_pipeline(start_stage: int = 1, end_stage: int = None):
    """
    Run the data processing pipeline.

    Parameters
    ----------
    start_stage : int
        Stage number to start from (1-indexed).
    end_stage : int, optional
        Stage number to end at (inclusive). If None, run all remaining stages.
    """
    print("=" * 80)
    print("NN4PSYCH DATA PROCESSING PIPELINE")
    print("=" * 80)

    if end_stage is None:
        end_stage = len(PIPELINE_STAGES)

    stages_to_run = PIPELINE_STAGES[start_stage - 1:end_stage]

    for stage_info in stages_to_run:
        print(f"\n{'=' * 80}")
        print(f"{stage_info['stage']}")
        print(f"{'=' * 80}\n")

        for script_name, description in stage_info['scripts']:
            script_path = PIPELINE_DIR / script_name

            if not script_path.exists():
                print(f"WARNING: Script not found: {script_path}")
                print(f"Skipping: {description}")
                continue

            print(f"Running: {description}")
            print(f"Script: {script_name}")
            print("-" * 40)

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PIPELINE_DIR.parent.parent),
            )

            if result.returncode != 0:
                print(f"\nERROR: {script_name} failed with return code {result.returncode}")
                user_input = input("Continue with next script? (y/n): ")
                if user_input.lower() != 'y':
                    print("Pipeline execution stopped.")
                    return False

            print()

    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    return True


def main():
    """Main entry point with command-line argument handling."""
    import argparse

    parser = argparse.ArgumentParser(description="Run NN4Psych data processing pipeline")
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='Stage number to start from (default: 1)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='Stage number to end at (default: run all)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all pipeline stages'
    )

    args = parser.parse_args()

    if args.list:
        print("Pipeline Stages:")
        for i, stage_info in enumerate(PIPELINE_STAGES, 1):
            print(f"\nStage {i}: {stage_info['stage']}")
            for script_name, description in stage_info['scripts']:
                print(f"  - {script_name}: {description}")
        return

    run_pipeline(start_stage=args.start, end_stage=args.end)


if __name__ == '__main__':
    main()
