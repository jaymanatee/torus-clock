import torch
from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltOutput
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def load_model(model_name: str="chronos-bolt-small") -> ChronosBoltModelForForecasting:
    """Load a Chronos-Bolt model from models/ in eval mode."""
    return ChronosBoltModelForForecasting.from_pretrained(f"../models/{model_name}").eval()


def capture(model: ChronosBoltModelForForecasting, context: torch.Tensor) -> tuple[dict[str, torch.Tensor], ChronosBoltOutput]:
    """
    Run one forward pass and return the residual stream at every encoder layer.

    Layer 0 is the patch embedding, [B, N, 512]. Layer i is the output of
    encoder block i, [B, N+1, 512], with the [REG] token last.
    """
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[name] = h.detach()
        return hook

    handles = [model.input_patch_embedding.register_forward_hook(make_hook("layer 0"))]
    for i, block in enumerate(model.encoder.block):
        handles.append(block.register_forward_hook(make_hook(f"layer {i+1}")))
    handles.append(model.encoder.final_layer_norm.register_forward_hook(make_hook("encoder out")))
    try:
        with torch.no_grad():
            out = model(context)
    finally:
        for h in handles:
            h.remove()
    return captured, out
