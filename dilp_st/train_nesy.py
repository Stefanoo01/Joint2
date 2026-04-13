import sys
import os
import argparse
import time
import logging

import torch
import torch.nn.functional as F

# 1. Add RSBench-code to PYTHONPATH so we can import its modules
# We assume the user executes from `Joint2/dilp_st` or `Joint2`
current_dir = os.path.abspath(os.path.dirname(__file__))
rsbench_rss_path = os.path.join(current_dir, "..", "rsbench-code", "rsseval", "rss")

# Add to sys.path so 'datasets' can be imported
sys.path.append(rsbench_rss_path)

# RSBench imports
from datasets.shortcutmnist import SHORTMNIST
from backbones.addmnist_single import MNISTSingleEncoder

# DILP imports
from .configs.mnist_add import build_mnist_add_problem
from .learning.model import ILPLearner, TrainConfig
from .models.nesy_wrapper import NeSyWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

class MockArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def main():
    parser = argparse.ArgumentParser(description="NeSy Joint Training: DILP + CBM")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dilp-lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=16) # keep low due to batch DILP memory
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--entropy-coeff", type=float, default=0.001)
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.1)
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # 1. Initialize RSBench SHORTMNIST Dataset
    # Provide the necessary mock arguments for the RSBench Dataset constructor
    ds_args = MockArgs(
        model="mnistnn",  # bypass their internal CBM logic
        task="addition",
        c_sup=0,
        which_c=-1,
        batch_size=args.batch_size,
        dataset="shortmnist",
        joint=False,
        backbone="none",
        n_epochs=args.epochs,
        lr=args.lr,
        exp_decay=0.99,
        warmup_steps=0,
        finetuning=False,
        validate=True,
    )
    
    logger.info("Loading RSBench SHORTMNIST dataset...")
    dataset = SHORTMNIST(ds_args)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()

    # 2. Build DILP ILP problem
    logger.info("Initializing DILP Problem & Graph...")
    problem = build_mnist_add_problem()
    config = TrainConfig(
        epochs=args.epochs,
        lr=args.dilp_lr,
        program_size=1,            # we only need 1 clause for `add`
        inference_steps=2,         # steps=2 is enough: add <- digit, digit, plus
        temperature_start=args.temp_start,
        temperature_end=args.temp_end,
        entropy_coeff=args.entropy_coeff,
        n_beam=5,                  # Fast beam search 
    )
    learner = ILPLearner(problem, config, device=device)

    # 3. Build RSBench CBM Encoder
    logger.info("Loading RSBench CNN conceptizer...")
    cbm = MNISTSingleEncoder().to(device)

    # 4. Wrap them together
    model = NeSyWrapper(encoder=cbm, learner=learner, n_images=2).to(device)

    # Prepare Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': args.lr},
        {'params': model.learner.inference.parameters(), 'lr': args.dilp_lr}
    ])

    # 5. Training Loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Anneal temperature
        progress = epoch / max(args.epochs - 1, 1)
        temperature = args.temp_start + progress * (args.temp_end - args.temp_start)

        t0 = time.time()
        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # target_probs is [B, 19] bounding the probability for each sum from 0 to 18
            # The values are smoothly clamped to [0, 1] by DILP SoftOR
            target_probs = model(images, temperature=temperature)
            
            # Since probabilities are in [0, 1] and sum roughly to 1 (not strictly guaranteed in fuzzy logic but bounded)
            # We use NLL Loss over the log probabilities: L = - log( p_true + eps )
            
            # Gather the probabilities of the correct targets
            # targets is [B]
            gathered_probs = target_probs.gather(1, targets.unsqueeze(-1).long()).squeeze(-1)
            loss = - torch.log(gathered_probs.clamp(min=1e-7)).mean()
            
            # Entropy regularisation over clause weights
            if config.entropy_coeff > 0:
                entropy_loss = torch.tensor(0.0, device=device)
                for l in range(config.program_size):
                    probs = F.softmax(model.learner.inference.W[l] / temperature, dim=0)
                    entropy_loss = entropy_loss - (probs * (probs + 1e-10).log()).sum()
                loss = loss + config.entropy_coeff * entropy_loss
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(targets)
            
            # Accuracy metric
            preds = target_probs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += len(targets)
            
        train_time = time.time() - t0
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                target_probs = model(images, temperature=args.temp_end)
                preds = target_probs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += len(targets)
        
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch:2d}/{args.epochs} [{train_time:.1f}s] | Temp: {temperature:.3f} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # --- DEBUG LOGGING ---
        logger.info("  === Learned Program (Current State) ===")
        program = learner.extract_program_with_probabilities()
        for clause, prob in program:
            if prob > 0.05:  # Only print clauses with meaningful weight
                logger.info(f"    {clause}  (p={prob:.4f})")
                
        # --- CONCEPT MAPPING VIEW ---
        # Pick one batch from validation to peek inside the CBM
        viz_images, viz_targets, viz_concepts = next(iter(val_loader))
        viz_images = viz_images.to(device)
        with torch.no_grad():
            if viz_images.size(-1) > 28:
                xs = torch.split(viz_images, viz_images.size(-1) // 2, dim=-1)
            else:
                xs = [viz_images[:, i] for i in range(2)]
                
            lc1, _, _ = model.encoder(xs[0])
            lc2, _, _ = model.encoder(xs[1])
            pred_digits1 = F.softmax(lc1.squeeze(1), dim=-1).argmax(dim=-1)
            pred_digits2 = F.softmax(lc2.squeeze(1), dim=-1).argmax(dim=-1)
            
        logger.info("  --- Concept Shortcut Analysis (First 5 Val samples) ---")
        logger.info(f"    True Digits    : {viz_concepts[:5, 0].tolist()} + {viz_concepts[:5, 1].tolist()}")
        logger.info(f"    Learned Digits : {pred_digits1[:5].tolist()} + {pred_digits2[:5].tolist()}")
        logger.info(f"    True Sum Target: {viz_targets[:5].tolist()}")

        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    logger.info("=== Learned Program ===")
    program = learner.extract_program_with_probabilities()
    for clause, prob in program:
        logger.info(f"  {clause}  (p={prob:.4f})")

if __name__ == "__main__":
    main()
