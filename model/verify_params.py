"""Verify model parameter count hits ~30M target."""

import torch
from model.architecture import NeuralSpellModel


def main():
    model = NeuralSpellModel()
    total = model.count_parameters()

    print(f"NeuralSpell Model Parameter Count")
    print(f"{'=' * 50}")

    # Breakdown by component
    embedding_params = model.token_embedding.weight.numel()
    print(f"Token embeddings: {embedding_params:>12,} (tied with output)")

    layer_params = 0
    for i, layer in enumerate(model.layers):
        lp = sum(p.numel() for p in layer.parameters())
        layer_params += lp
        print(f"Layer {i}: {lp:>18,}")

    norm_params = sum(p.numel() for p in model.norm.parameters())
    print(f"Final norm: {norm_params:>14,}")

    print(f"{'=' * 50}")
    print(f"Total (unique):    {total:>12,}")
    print(f"Target:            {'~30,000,000':>12}")
    print(f"{'=' * 50}")

    if 25_000_000 <= total <= 35_000_000:
        print("PASS: Parameter count within target range (25M-35M)")
    else:
        print(f"WARN: Parameter count {total:,} outside target range")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randint(0, 32000, (2, 128))
    mask = torch.ones(2, 128)
    with torch.no_grad():
        logits = model(x, mask)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    assert logits.shape == (2, 128, 32000)
    print("Forward pass OK")


if __name__ == "__main__":
    main()
