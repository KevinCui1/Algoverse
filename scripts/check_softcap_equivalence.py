"""GPU preflight for the deferred Gemma 2 final-logit softcap.

This compares transformers' native operation with the readout's deferred
operation on the same checkpoint, input, LM-head shape and dtype.  It is a
deployment check for the memory-path workaround, not a scientific measurement.
"""

from __future__ import annotations

import argparse

import torch

from hiringcue import stage0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    tokenizer = stage0.load_tokenizer(args.model, args.model_path)
    model = stage0.load_model(args.model, args.model_path)
    encoded = tokenizer(
        ["Answer with one word: Yes or No."], return_tensors="pt"
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    cap = model.config.final_logit_softcapping
    if cap is None:
        raise RuntimeError("checkpoint has no final_logit_softcapping to check")

    with torch.inference_mode():
        native = model(**encoded, use_cache=False, logits_to_keep=1).logits
        model.config.final_logit_softcapping = None
        try:
            deferred = model(**encoded, use_cache=False, logits_to_keep=1).logits
        finally:
            model.config.final_logit_softcapping = cap
        deferred = deferred / cap
        deferred = torch.tanh(deferred)
        deferred = deferred * cap

    maximum_delta = float((native - deferred).abs().max().item())
    print(
        {
            "model": args.model,
            "dtype": str(native.dtype),
            "values": native.numel(),
            "bitwise_equal": bool(torch.equal(native, deferred)),
            "maximum_absolute_delta": maximum_delta,
            "config_restored": model.config.final_logit_softcapping == cap,
        }
    )
    if not torch.equal(native, deferred):
        raise RuntimeError(
            f"deferred softcap differs from native softcap (max delta {maximum_delta})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
