"""Verify encoder-decoder model parameter count hits ~385M target."""

import torch
from model.architecture import NeuralSpellModel


def main():
    model = NeuralSpellModel()
    total = model.count_parameters()

    print(f"NeuralSpell Encoder-Decoder Parameter Count")
    print(f"{'=' * 55}")

    # Shared embedding (tied 3-way, counted once)
    embedding_params = model.shared_embedding.weight.numel()
    print(f"Shared embeddings:  {embedding_params:>14,} (3-way tied)")

    # Encoder breakdown
    enc_total = 0
    for i, layer in enumerate(model.encoder_layers):
        lp = sum(p.numel() for p in layer.parameters())
        enc_total += lp
        if i == 0:
            print(f"Encoder layer 0:    {lp:>14,}")
    enc_norm = sum(p.numel() for p in model.encoder_norm.parameters())
    print(f"Encoder layers 1-11:{'':>1}{enc_total - sum(p.numel() for p in model.encoder_layers[0].parameters()):>13,}")
    print(f"Encoder norm:       {enc_norm:>14,}")
    print(f"Encoder total:      {enc_total + enc_norm:>14,}")

    # Decoder breakdown
    dec_total = 0
    for i, layer in enumerate(model.decoder_layers):
        lp = sum(p.numel() for p in layer.parameters())
        dec_total += lp
        if i == 0:
            print(f"Decoder layer 0:    {lp:>14,}")
    dec_norm = sum(p.numel() for p in model.decoder_norm.parameters())
    print(f"Decoder layers 1-11:{'':>1}{dec_total - sum(p.numel() for p in model.decoder_layers[0].parameters()):>13,}")
    print(f"Decoder norm:       {dec_norm:>14,}")
    print(f"Decoder total:      {dec_total + dec_norm:>14,}")

    print(f"{'=' * 55}")
    print(f"Total (unique):     {total:>14,}")
    print(f"Target:             {'~385,000,000':>14}")
    print(f"{'=' * 55}")

    if 370_000_000 <= total <= 400_000_000:
        print("PASS: Parameter count within target range (370M-400M)")
    else:
        print(f"WARN: Parameter count {total:,} outside target range")

    # Test encode()
    print("\nTesting encode()...")
    enc_input = torch.randint(0, 32000, (2, 128))
    enc_mask = torch.ones(2, 128, dtype=torch.long)
    with torch.no_grad():
        encoder_output = model.encode(enc_input, enc_mask)
    print(f"  Encoder input:  {tuple(enc_input.shape)}")
    print(f"  Encoder output: {tuple(encoder_output.shape)}")
    assert encoder_output.shape == (2, 128, model.hidden_size)
    print("  OK")

    # Test decode()
    print("Testing decode()...")
    dec_input = torch.randint(0, 32000, (2, 64))
    dec_mask = torch.ones(2, 64, dtype=torch.long)
    with torch.no_grad():
        logits = model.decode(dec_input, dec_mask, encoder_output, enc_mask)
    print(f"  Decoder input:  {tuple(dec_input.shape)}")
    print(f"  Logits output:  {tuple(logits.shape)}")
    assert logits.shape == (2, 64, 32000)
    print("  OK")

    # Test full forward()
    print("Testing forward()...")
    with torch.no_grad():
        logits = model(enc_input, enc_mask, dec_input, dec_mask)
    assert logits.shape == (2, 64, 32000)
    print(f"  Full forward: {tuple(enc_input.shape)} + {tuple(dec_input.shape)} -> {tuple(logits.shape)}")
    print("  OK")

    # Test gradient checkpointing
    print("Testing gradient checkpointing...")
    model.enable_gradient_checkpointing()
    model.train()
    logits = model(enc_input, enc_mask, dec_input, dec_mask)
    loss = logits.sum()
    loss.backward()
    print("  Gradient checkpointing OK")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
