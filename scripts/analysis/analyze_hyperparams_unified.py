#!/usr/bin/env python3
"""
Unified hyperparameter analysis script.

This replaces the 5 duplicate analyze_hyperparams_*.py scripts with a single
parameterized version.
"""

import argparse
from pathlib import Path

from nn4psych.analysis.hyperparams import HyperparamAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweeps")

    parser.add_argument(
        '--param',
        type=str,
        required=True,
        choices=['gamma', 'rollout', 'preset', 'scale'],
        help='Hyperparameter to analyze'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./model_params/',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='V3',
        help='Model version string'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of evaluation epochs'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=5.0,
        help='Performance threshold for model inclusion'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./plots/',
        help='Directory for output plots'
    )
    parser.add_argument(
        '--reset_memory',
        action='store_true',
        help='Reset memory between epochs'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Analyzing hyperparameter: {args.param}")
    print(f"Model directory: {args.model_dir}")
    print(f"Version: {args.version}")
    print()

    # Create analyzer
    analyzer = HyperparamAnalyzer(
        param_name=args.param,
        model_dir=args.model_dir,
        version=args.version,
    )

    # Run analysis
    results = analyzer.analyze_sweep(
        n_epochs=args.epochs,
        reset_memory=args.reset_memory,
        performance_threshold=args.threshold,
    )

    # Generate report
    report = analyzer.generate_summary_report(results)
    print(report)

    # Save report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"analysis_{args.param}_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
