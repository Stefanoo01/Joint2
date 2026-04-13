"""
Entry point for training ∂ILP-ST on standard ILP tasks.

Usage:
    python -m dilp_st.main --task member
    python -m dilp_st.main --task plus
    python -m dilp_st.main --task append
    python -m dilp_st.main --task plus --epochs 1000 --lr 0.005
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import torch

from .configs.append import build_append_problem
from .configs.member import build_member_problem
from .configs.plus import build_plus_problem
from .learning.model import ILPLearner, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TASKS = {
    "member": build_member_problem,
    "plus": build_plus_problem,
    "append": build_append_problem,
}


def main():
    parser = argparse.ArgumentParser(description="∂ILP-ST: Differentiable ILP for Structured Examples")
    parser.add_argument("--task", type=str, default="member", choices=list(TASKS.keys()),
                        help="ILP task to solve (default: member)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--program-size", type=int, default=3,
                        help="m: target number of clauses in the program")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Softor smoothing parameter (only used in logsumexp mode)")
    parser.add_argument("--inference-steps", type=int, default=3,
                        help="T: number of forward-chaining steps")
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.1)
    parser.add_argument("--entropy-coeff", type=float, default=0.001)
    parser.add_argument("--n-beam", type=int, default=20,
                        help="Beam width for clause generation")
    parser.add_argument("--t-beam", type=int, default=3,
                        help="Number of beam search refinement steps")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Maximum term depth during refinement")
    parser.add_argument("--max-body", type=int, default=3,
                        help="Maximum body atoms per clause")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', 'mps', or 'auto'")
    parser.add_argument("--softor-mode", type=str, default="probabilistic",
                        choices=["probabilistic", "logsumexp", "max"],
                        help="Softor mode for logical OR aggregation")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    # Device selection.
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Task: {args.task}")
    logger.info(f"Device: {device}")

    # Build problem.
    problem = TASKS[args.task]()
    logger.info(f"E+: {len(problem.positive_examples)}, E-: {len(problem.negative_examples)}, "
                f"B: {len(problem.background)}")

    # Configure training.
    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        program_size=args.program_size,
        gamma=args.gamma,
        inference_steps=args.inference_steps,
        temperature_start=args.temp_start,
        temperature_end=args.temp_end,
        entropy_coeff=args.entropy_coeff,
        softor_mode=args.softor_mode,
        n_beam=args.n_beam,
        t_beam=args.t_beam,
        max_depth=args.max_depth,
        max_body=args.max_body,
        log_interval=args.log_interval,
    )

    # Build learner (symbolic preprocessing + tensor construction).
    t0 = time.perf_counter()
    learner = ILPLearner(problem, config, device=device)
    t_preprocess = time.perf_counter() - t0
    logger.info(f"Preprocessing time: {t_preprocess:.2f}s")

    # Train.
    t0 = time.perf_counter()
    loss_history = learner.train()
    t_train = time.perf_counter() - t0
    logger.info(f"Training time: {t_train:.2f}s")

    # Extract program.
    logger.info("\n=== Learned Program ===")
    program = learner.extract_program_with_probabilities()
    for clause, prob in program:
        logger.info(f"  {clause}  (p={prob:.4f})")

    # Evaluate on training data.
    metrics = learner.evaluate(
        problem.positive_examples,
        problem.negative_examples,
    )
    logger.info(f"\nTraining metrics: {metrics}")

    logger.info(f"\nTotal time: {t_preprocess + t_train:.2f}s")


if __name__ == "__main__":
    main()
