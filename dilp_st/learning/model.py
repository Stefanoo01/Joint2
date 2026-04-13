"""
ILP Learner — top-level model combining symbolic preprocessing with
differentiable inference.

Handles:
  - One-time symbolic preprocessing (clause generation, enumeration, encoding)
  - Training loop with cross-entropy loss (Eq. 15)
  - Temperature annealing and entropy regularisation
  - Program extraction to human-readable clauses
  - Interface for future neural concept extractor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..logic.clause_generator import beam_search
from ..logic.ground_enumerator import enumerate_ground_atoms
from ..logic.language import BOTTOM, TOP, Atom, Clause
from ..problems.ilp_problem import ILPProblem
from .infer import DifferentiableInference
from .tensor_encoder import build_fact_mask, build_index_tensor

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration for the training procedure."""
    epochs: int = 500
    lr: float = 1e-2
    program_size: int = 3
    gamma: float = 0.05
    inference_steps: int = 3
    temperature_start: float = 2.0
    temperature_end: float = 0.1
    entropy_coeff: float = 0.001
    softor_mode: str = "probabilistic"
    # Clause generation parameters
    n_beam: int = 20
    t_beam: int = 3
    max_depth: int = 4
    max_body: int = 3
    # Logging
    log_interval: int = 50


class ILPLearner:
    """End-to-end ILP learner wrapping symbolic preprocessing + differentiable
    inference.

    Usage::

        problem = ILPProblem(...)
        learner = ILPLearner(problem, config, device)
        learner.train()
        program = learner.extract_program()
    """

    def __init__(
        self,
        problem: ILPProblem,
        config: TrainConfig,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.problem = problem
        self.config = config

        logger.info("=== Symbolic Preprocessing (CPU) ===")

        # 1. Clause generation via beam search (Algorithm 1).
        logger.info("Generating clauses via beam search ...")
        self.clauses = beam_search(
            initial_clauses=problem.initial_clauses,
            language=problem.language,
            positive=problem.positive_examples,
            negative=problem.negative_examples,
            background=problem.background,
            n_beam=config.n_beam,
            t_beam=config.t_beam,
            max_depth=config.max_depth,
            max_body=config.max_body,
        )
        logger.info(f"  Generated {len(self.clauses)} candidate clauses.")

        # 2. Ground atom enumeration (Algorithm 2).
        logger.info("Enumerating ground atoms ...")
        self.ground_atoms = enumerate_ground_atoms(
            positive=problem.positive_examples,
            negative=problem.negative_examples,
            background=problem.background,
            clauses=self.clauses,
            T=config.inference_steps,
        )
        logger.info(f"  Enumerated {len(self.ground_atoms)} ground atoms.")

        # Build lookup.
        self.atom_to_idx: Dict[Atom, int] = {
            a: i for i, a in enumerate(self.ground_atoms)
        }

        # 3. Tensor encoding (CPU → GPU).
        logger.info("Building index tensor ...")
        X = build_index_tensor(self.clauses, self.ground_atoms, device=device)
        fact_mask = build_fact_mask(self.clauses, self.ground_atoms, device=device)
        logger.info(f"  X shape: {X.shape}")

        # 4. Differentiable inference module (GPU).
        self.inference = DifferentiableInference(
            X=X,
            fact_mask=fact_mask,
            program_size=config.program_size,
            gamma=config.gamma,
            T=config.inference_steps,
            softor_mode=config.softor_mode,
        ).to(device)

        # 5. Build training data.
        self._build_training_data()

        logger.info(f"  Device: {device}")
        logger.info("=== Preprocessing Complete ===\n")

    def _build_training_data(self) -> None:
        """Build (atom_index, label) pairs from E+ and E- (Eq. 13)."""
        indices: list[int] = []
        labels: list[float] = []

        for e in self.problem.positive_examples:
            if e in self.atom_to_idx:
                indices.append(self.atom_to_idx[e])
                labels.append(1.0)

        for e in self.problem.negative_examples:
            if e in self.atom_to_idx:
                indices.append(self.atom_to_idx[e])
                labels.append(0.0)

        self.train_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        self.train_labels = torch.tensor(labels, dtype=torch.float32, device=self.device)

    def _build_v0(
        self,
        soft_facts: Optional[Dict[Atom, float]] = None,
    ) -> torch.Tensor:
        """Build the initial valuation vector v0 (Eq. 5).

        Parameters
        ----------
        soft_facts : dict[Atom, float], optional
            Soft probability values for atoms (from a neural concept extractor).
            If None, only hard background knowledge is used.
        """
        v0 = torch.zeros(len(self.ground_atoms), dtype=torch.float32, device=self.device)

        # ⊤ always has valuation 1.
        v0[self.atom_to_idx[TOP]] = 1.0

        # Background knowledge (hard facts).
        for a in self.problem.background:
            if a in self.atom_to_idx:
                v0[self.atom_to_idx[a]] = 1.0

        # Soft facts from concept extractor.
        if soft_facts is not None:
            for atom, prob in soft_facts.items():
                if atom in self.atom_to_idx:
                    v0[self.atom_to_idx[atom]] = prob

        return v0

    def set_soft_facts_tensor(self, concept_probs: torch.Tensor, atoms: List[Atom]) -> torch.Tensor:
        """Interface for a neural concept extractor.

        Given a tensor of concept probabilities and the corresponding atoms,
        build a v0 that merges the soft probabilities with the hard background.

        Parameters
        ----------
        concept_probs : torch.Tensor
            Probability values [num_concepts], on the same device.
        atoms : list[Atom]
            Corresponding atoms for each probability.

        Returns
        -------
        torch.Tensor
            The initial valuation v0 [|G|].
        """
        v0 = torch.zeros(len(self.ground_atoms), dtype=torch.float32, device=self.device)
        v0[self.atom_to_idx[TOP]] = 1.0

        for a in self.problem.background:
            if a in self.atom_to_idx:
                v0[self.atom_to_idx[a]] = 1.0

        for prob, atom in zip(concept_probs, atoms):
            if atom in self.atom_to_idx:
                idx = self.atom_to_idx[atom]
                v0[idx] = torch.max(v0[idx], prob)

        return v0

    def set_soft_facts_tensor_batched(self, concept_probs: torch.Tensor, atoms: List[Atom]) -> torch.Tensor:
        """Interface for a batched neural concept extractor.

        Parameters
        ----------
        concept_probs : torch.Tensor
            Probability values [B, num_concepts], on the same device.
        atoms : list[Atom]
            Corresponding atoms for each probability along dim 1.

        Returns
        -------
        torch.Tensor
            The initial valuation v0 [B, |G|].
        """
        B = concept_probs.shape[0]
        v0 = torch.zeros((B, len(self.ground_atoms)), dtype=torch.float32, device=self.device)
        v0[:, self.atom_to_idx[TOP]] = 1.0

        for a in self.problem.background:
            if a in self.atom_to_idx:
                v0[:, self.atom_to_idx[a]] = 1.0

        for i, atom in enumerate(atoms):
            if atom in self.atom_to_idx:
                idx = self.atom_to_idx[atom]
                v0[:, idx] = torch.max(v0[:, idx], concept_probs[:, i])

        return v0

    def train(self) -> List[float]:
        """Run the training loop.

        Returns
        -------
        list[float]
            Loss history.
        """
        config = self.config
        optimizer = torch.optim.Adam(self.inference.parameters(), lr=config.lr)

        v0 = self._build_v0()
        loss_history: list[float] = []

        for epoch in range(config.epochs):
            # Temperature annealing: linear from start → end.
            progress = epoch / max(config.epochs - 1, 1)
            temperature = config.temperature_start + progress * (
                config.temperature_end - config.temperature_start
            )

            # Forward pass.
            vT = self.inference(v0, temperature=temperature)

            # Extract predictions for training atoms (Eq. 14).
            predictions = vT[self.train_indices]
            # Clamp to avoid log(0).
            predictions = predictions.clamp(1e-7, 1.0 - 1e-7)

            # Cross-entropy loss (Eq. 15).
            loss = F.binary_cross_entropy(predictions, self.train_labels)

            # Entropy regularisation.
            if config.entropy_coeff > 0:
                entropy_loss = torch.tensor(0.0, device=self.device)
                for l in range(config.program_size):
                    probs = F.softmax(self.inference.W[l] / temperature, dim=0)
                    entropy_loss = entropy_loss - (probs * (probs + 1e-10).log()).sum()
                loss = loss + config.entropy_coeff * entropy_loss

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if epoch % config.log_interval == 0 or epoch == config.epochs - 1:
                logger.info(
                    f"Epoch {epoch:4d}/{config.epochs} | "
                    f"loss={loss_val:.6f} | temp={temperature:.4f}"
                )

        return loss_history

    def extract_program(self) -> List[Clause]:
        """Extract the discrete program by taking argmax of clause weights.

        Returns
        -------
        list[Clause]
            Human-readable clauses forming the learned program.
        """
        indices = self.inference.extract_program_indices(
            temperature=self.config.temperature_end
        )
        program: list[Clause] = []
        seen: set[int] = set()
        for idx in indices:
            if idx not in seen:
                program.append(self.clauses[idx])
                seen.add(idx)
        return program

    def extract_program_with_probabilities(
        self,
    ) -> List[Tuple[Clause, float]]:
        """Extract program with probability of each selected clause."""
        probs_list = self.inference.get_clause_probabilities(
            temperature=self.config.temperature_end
        )
        result: list[Tuple[Clause, float]] = []
        seen: set[int] = set()
        for l, probs in enumerate(probs_list):
            idx = probs.argmax().item()
            if idx not in seen:
                result.append((self.clauses[idx], probs[idx].item()))
                seen.add(idx)
        return result

    def evaluate(
        self,
        test_positive: List[Atom],
        test_negative: List[Atom],
    ) -> Dict[str, float]:
        """Evaluate the model on test atoms.

        Returns accuracy and MSE metrics.
        """
        v0 = self._build_v0()
        with torch.no_grad():
            vT = self.inference(v0, temperature=self.config.temperature_end)

        correct = 0
        total = 0
        mse = 0.0

        for e in test_positive:
            if e in self.atom_to_idx:
                pred = vT[self.atom_to_idx[e]].item()
                correct += int(pred > 0.5)
                mse += (1.0 - pred) ** 2
                total += 1

        for e in test_negative:
            if e in self.atom_to_idx:
                pred = vT[self.atom_to_idx[e]].item()
                correct += int(pred <= 0.5)
                mse += pred ** 2
                total += 1

        accuracy = correct / max(total, 1)
        mse = mse / max(total, 1)
        return {"accuracy": accuracy, "mse": mse, "total": total}
