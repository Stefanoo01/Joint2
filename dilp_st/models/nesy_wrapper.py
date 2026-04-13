import torch
import torch.nn as nn
from typing import List

from ..learning.model import ILPLearner
from ..configs.mnist_add import img1, img2, nums, _digit, _add

class NeSyWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, learner: ILPLearner, n_images: int = 2):
        super().__init__()
        self.encoder = encoder
        self.learner = learner
        self.n_images = n_images
        
        # Atoms representing digit(img1, d) and digit(img2, d)
        self.atoms_img1 = [_digit(img1, nums[d]) for d in range(10)]
        self.atoms_img2 = [_digit(img2, nums[d]) for d in range(10)]
        
        # We need a flat list for set_soft_facts_tensor_batched
        self.atoms_concepts = self.atoms_img1 + self.atoms_img2
        
        # Atoms for the target sum add(img1, img2, s)
        self.atoms_target = [_add(img1, img2, nums[s]) for s in range(19)]
        
        # Verify target indices exist in the statically grounded DILP graph
        self.target_indices = [self.learner.atom_to_idx[a] for a in self.atoms_target]
        
    def _normalize_concepts(self, cs: torch.Tensor) -> torch.Tensor:
        """Apply softmax to concept logits."""
        # cs is [B, 2, 10]
        prob_digit1 = nn.Softmax(dim=1)(cs[:, 0, :])
        prob_digit2 = nn.Softmax(dim=1)(cs[:, 1, :])
        return torch.stack([prob_digit1, prob_digit2], dim=1)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        x can be [B, C, H, W*2] (if concatenated) or a batch of images.
        We adopt RSBench's approach: splitting along dim=-1.
        """
        cs = []
        # RSBench standard concat dimension is the width (dim=-1)
        if x.size(-1) > 28 and self.n_images == 2:
            xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        else:
            # If they are stacked in dim=1 as [B, 2, 1, 28, 28] maybe?
            if x.dim() == 5:
                xs = [x[:, i] for i in range(self.n_images)]
            else:
                raise ValueError(f"Unexpected image shape {x.shape}")
        
        for i in range(self.n_images):
            # rsbench encoder returns (c, mu, logvar)
            lc, _, _ = self.encoder(xs[i])
            cs.append(lc)
            
        # len = 2, each is [B, 10]
        # stack to [B, 2, 10]
        cs = torch.stack(cs, dim=1)
        pCs = self._normalize_concepts(cs) # [B, 2, 10]
        
        # flatten concepts to [B, 20] to match self.atoms_concepts
        concept_probs = pCs.view(-1, 20)
        
        # inject into DILP
        v0 = self.learner.set_soft_facts_tensor_batched(concept_probs, self.atoms_concepts)
        
        # forward chaining inference
        vT = self.learner.inference(v0, temperature=temperature) # [B, G]
        
        # extract probabilities for `add(img1, img2, 0...18)`
        target_probs = vT[:, self.target_indices] # [B, 19]
        
        return target_probs
