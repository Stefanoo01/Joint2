"""
Differentiable inference engine — GPU-accelerated forward chaining.

Implements Equations 6–12 of the paper:
  - gather        (Eq. 8)
  - clause_fn     (Eq. 7)  — product t-norm across body
  - weighted_sum  (Eq. 9)  — softmax-weighted combination
  - softor        (Eq. 11) — smooth logical OR (probabilistic sum)
  - infer_step    (Eq. 12) — one step of forward chaining
  - infer         (Eq. 6)  — T-step forward chaining

All operations are fully vectorised PyTorch tensor ops — no Python loops over
atoms or clauses during the forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableInference(nn.Module):
    """Differentiable forward-chaining inference module.

    Parameters
    ----------
    X : torch.LongTensor
        Index tensor of shape ``[num_clauses, num_atoms, max_body]``.
    fact_mask : torch.BoolTensor
        Boolean mask ``[num_clauses, num_atoms]`` — True where clause i is a
        unit fact matching atom j.
    program_size : int
        ``m`` — target number of clauses in the program.
    gamma : float
        Smoothing parameter for logsumexp softor mode (only used when
        ``softor_mode='logsumexp'``).  Smaller γ → closer to max.
    T : int
        Number of forward-chaining inference steps.
    softor_mode : str
        Which softor to use: ``'probabilistic'`` (default, bias-free),
        ``'logsumexp'`` (parametric, has γ·log(n) bias), or ``'max'``.
    """

    def __init__(
        self,
        X: torch.LongTensor,
        fact_mask: torch.BoolTensor,
        program_size: int,
        gamma: float = 0.05,
        T: int = 2,
        softor_mode: str = "probabilistic",
    ):
        super().__init__()
        num_clauses = X.shape[0]

        # Register as buffers (moved with .to(device) but not parameters).
        self.register_buffer("X", X)
        self.register_buffer("fact_mask", fact_mask)

        self.num_clauses = num_clauses
        self.num_atoms = X.shape[1]
        self.max_body = X.shape[2]
        self.program_size = program_size
        self.gamma = gamma
        self.T = T
        self.softor_mode = softor_mode

        # Learnable weights: m weight vectors, each of dim |C|.
        # W[l] is used to softly select the l-th clause of the target program.
        self.W = nn.ParameterList(
            [nn.Parameter(torch.randn(num_clauses)) for _ in range(program_size)]
        )

    def forward(self, v0: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Perform T-step differentiable forward-chaining inference.

        Parameters
        ----------
        v0 : torch.Tensor
            Initial valuation vector of shape ``[num_atoms]``.
            ``v0[j]`` is the soft truth value of ground atom j.
        temperature : float
            Temperature for softmax over clause weights (for annealing).

        Returns
        -------
        torch.Tensor
            Final valuation vector ``vT`` of shape ``[num_atoms]``.
        """
        v = v0
        for _t in range(self.T):
            v = self._infer_step(v, temperature)
        return v

    def _infer_step(self, v: torch.Tensor, temperature: float) -> torch.Tensor:
        """One inference step: Eqs. 7–12.

        Computes r(v) and amalgamates with current valuation.
        """
        is_batched = v.dim() == 2

        if is_batched:
            # --- Eq. 7: Compute clause functions c_i(v) for all clauses at once ---
            # X shape: [C, G, b],  v shape: [B, G]
            # gathered shape: [B, C, G_atoms, b]
            gathered = v[:, self.X]
            # clause_vals shape: [B, C, G_atoms]
            clause_vals = gathered.prod(dim=-1)
            # Handle unit clauses
            fact_mask_exp = self.fact_mask.unsqueeze(0)
            clause_vals = torch.where(fact_mask_exp, torch.ones_like(clause_vals), clause_vals)

            # --- Eq. 9: Weighted sum for each program slot ---
            W_mat = torch.stack([F.softmax(self.W[l] / temperature, dim=0) for l in range(self.program_size)]) # [m, C]
            # h shape: [B, m, G]
            h = torch.einsum('mc, bcg -> bmg', W_mat, clause_vals)

            # --- Eq. 10–11: softor across program slots ---
            # r(v) = softor(h_1, ..., h_m) over dim = 1 (m)
            r = self._softor(h, dim=1)  # [B, G]

            # --- Eq. 12: Amalgamate with previous valuation ---
            stacked = torch.stack([v, r], dim=1)  # [B, 2, G]
            v_next = self._softor(stacked, dim=1)  # [B, G]
            
        else:
            # --- Eq. 7: Compute clause functions c_i(v) for all clauses at once ---
            gathered = v[self.X]  # [C, G, b]
            clause_vals = gathered.prod(dim=-1) # [C, G]
            clause_vals = torch.where(self.fact_mask, torch.ones_like(clause_vals), clause_vals)

            # --- Eq. 9: Weighted sum for each program slot ---
            W_mat = torch.stack([F.softmax(self.W[l] / temperature, dim=0) for l in range(self.program_size)])
            h = W_mat @ clause_vals # [m, G]

            # --- Eq. 10–11: softor across program slots ---
            r = self._softor(h, dim=0)  # [G]

            # --- Eq. 12: Amalgamate with previous valuation ---
            stacked = torch.stack([v, r], dim=0)  # [2, G]
            v_next = self._softor(stacked, dim=0)  # [G]

        return v_next

    def _softor(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Smooth logical OR over the given dimension.

        Three modes available:

        * ``'probabilistic'`` (default): ``1 - ∏(1 - x_i)``
        * ``'logsumexp'``: ``γ · log(Σ exp(x_i/γ))``
        * ``'max'``: ``max(x_i)``

        Parameters
        ----------
        x : torch.Tensor
            Tensor of valuations to OR together.
        dim : int
            The dimension to reduce across.

        Returns
        -------
        torch.Tensor
            Clamped to [0, 1].
        """
        if self.softor_mode == "probabilistic":
            result = 1.0 - torch.prod(1.0 - x, dim=dim)
            return result.clamp(0.0, 1.0)
        elif self.softor_mode == "logsumexp":
            result = self.gamma * torch.logsumexp(x / self.gamma, dim=dim)
            return result.clamp(0.0, 1.0)
        elif self.softor_mode == "max":
            return x.max(dim=dim).values.clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unknown softor mode: {self.softor_mode}")

    def extract_program_indices(self, temperature: float = 1.0) -> list[int]:
        """Extract the discrete program: argmax of each weight vector.

        Returns list of clause indices (one per program slot).
        """
        indices = []
        for l in range(self.program_size):
            probs = F.softmax(self.W[l] / temperature, dim=0)
            idx = probs.argmax().item()
            indices.append(idx)
        return indices

    def get_clause_probabilities(self, temperature: float = 1.0) -> list[torch.Tensor]:
        """Return the probability distribution over clauses for each slot."""
        return [F.softmax(self.W[l] / temperature, dim=0) for l in range(self.program_size)]
