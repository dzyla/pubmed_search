"""
Builds the 'precise ranking' model (reranker.py): MedCPT cross-encoder as int8 ONNX.
Run on a machine with torch, transformers and onnxruntime, then copy OUT_DIR to the
search backend's MSS_RERANKER_DIR (default <repo>/models/medcpt-ce-int8):

    python tools/export_reranker.py OUT_DIR

Per-channel weight quantization of the MatMuls only: default dynamic int8 quantization
visibly changes the ranking; this keeps it (Spearman 0.99 vs fp32 on the benchmark).
"""
import os
import sys

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NAME = "ncbi/MedCPT-Cross-Encoder"


def main(out):
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NAME).eval()
    example = tok(["query"], ["a document"], return_tensors="pt")
    names = list(example.keys())

    class Wrap(torch.nn.Module):          # forward() takes the tokenizer outputs in this order
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args):
            return self.m(**dict(zip(names, args))).logits

    f32 = os.path.join(out, "model.fp32.onnx")
    torch.onnx.export(Wrap(model), tuple(example[n] for n in names), f32, input_names=names,
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={**{n: {0: "b", 1: "s"} for n in names}, "logits": {0: "b"}})
    quantize_dynamic(f32, os.path.join(out, "model.onnx"), weight_type=QuantType.QInt8, per_channel=True,
                     reduce_range=True, op_types_to_quantize=["MatMul"])
    os.remove(f32)
    tok.save_pretrained(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
