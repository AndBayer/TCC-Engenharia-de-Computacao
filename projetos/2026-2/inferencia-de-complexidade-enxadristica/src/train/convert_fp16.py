#!/usr/bin/env python3
"""Converte o modelo ONNX para FP16 (tensor cores), com entrada e saídas em float32.

Com keep_io_types=True, a entrada e as saídas continuam em float32, então o motor C++ não muda; basta apontar WeightsFile para o modelo convertido.

Uso:
    python convert_fp16.py model.onnx model_fp16.onnx
    python convert_fp16.py model.onnx model_fp16.onnx --check  # paridade
"""

import argparse
import sys

import onnx
from onnxconverter_common import float16


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("entrada", help="modelo float32 (.onnx)")
    ap.add_argument("saida", help="modelo fp16 de saida (.onnx)")
    ap.add_argument("--check", action="store_true", help="compara as saidas fp32 vs fp16 em entradas aleatorias")
    args = ap.parse_args()

    model = onnx.load(args.entrada)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,  # entrada e saídas continuam em float32 --> o motor C++ não muda
    )
    onnx.save(model_fp16, args.saida)
    print(f"[OK] {args.entrada} -> {args.saida} (pesos fp16, I/O float32)")

    if not args.check:
        return

    import numpy as np
    import onnxruntime as ort

    # sessão no CUDA, ou na CPU se o CUDA não estiver disponível
    def session(path):
        return ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    s32, s16 = session(args.entrada), session(args.saida)
    inp = s32.get_inputs()[0]
    # dimensão dinâmica de lote --> usa 64
    shape = [d if isinstance(d, int) else 64 for d in inp.shape]
    rng = np.random.default_rng(42)
    # planos binários (0/1), mais próximos da entrada real da rede que ruído gaussiano
    x = rng.integers(0, 2, size=shape).astype(np.float32)

    o32 = s32.run(None, {inp.name: x})
    o16 = s16.run(None, {inp.name: x})

    pol32, val32 = o32[0], o32[1]
    pol16, val16 = o16[0], o16[1]

    # a paridade que importa para o MCTS é a ordem dos lances (argmax/top-k) e o valor
    top1_igual = (pol32.argmax(axis=1) == pol16.argmax(axis=1)).mean()
    # top-k --> fração dos k lances de maior logit que aparecem nos dois modelos
    k = 5
    topk32 = np.argsort(-pol32, axis=1)[:, :k]
    topk16 = np.argsort(-pol16, axis=1)[:, :k]
    topk_jaccard = np.mean([
        len(set(a) & set(b)) / k for a, b in zip(topk32, topk16)
    ])
    dval = np.abs(val32 - val16).max()

    print(f"[CHECK] top-1 igual: {top1_igual:.1%}  "
          f"top-{k} sobreposicao: {topk_jaccard:.1%}  "
          f"|delta valor| max: {dval:.5f}")
    # aceita com top-1 igual em pelo menos 99% das entradas e diferença de valor de no máximo 0,01
    if top1_igual < 0.99 or dval > 0.01:
        print("[AVISO] divergencia acima do esperado - validar com model_benchmark antes de usar em medicao", file=sys.stderr)
        sys.exit(1)
    print("[OK] paridade adequada para uso no motor")


if __name__ == "__main__":
    main()
