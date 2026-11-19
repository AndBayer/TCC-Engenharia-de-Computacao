"""Exporta checkpoints .pt para .onnx, o formato lido pelo motor.

Uso:
    python export_onnx.py --checkpoints-dir <pasta>
    python export_onnx.py --single checkpoint_epoch12.pt
"""
import argparse
import os
import sys
import glob
import torch

from train import ChessNet

def export_one(pt_path, onnx_path, channels=128, res_blocks=12, opset=17, dynamic_batch=True):
    """Exporta um checkpoint em float32, com a entrada 'board' e as saídas 'policy' e 'value' que o motor usa."""
    print(f"[EXPORT] {os.path.basename(pt_path)} -> {os.path.basename(onnx_path)}")

    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=True)
    model = ChessNet(channels=channels, num_res_blocks=res_blocks)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # entrada de exemplo; o export roda a rede uma vez com ela para montar o grafo
    dummy_input = torch.randn(1, 19, 8, 8, dtype=torch.float32)

    # tamanho do lote livre
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'board':  {0: 'batch'},
            'policy': {0: 'batch'},
            'value':  {0: 'batch'},
        }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['board'],
        output_names=['policy', 'value'],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        dynamo=False
    )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  -> OK ({size_mb:.1f} MB)")


def main():
    """Exporta um checkpoint (--single) ou todos os checkpoint_epoch*.pt de uma pasta."""
    parser = argparse.ArgumentParser(description="Exporta checkpoints PT -> ONNX")
    parser.add_argument('--checkpoints-dir', type=str, default='./', help='Pasta com os arquivos *.pt')
    parser.add_argument('--output-dir', type=str, default=None, help='Pasta de saida dos .onnx')
    parser.add_argument('--single', type=str, default=None, help='Exporta apenas o arquivo .pt especificado')
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--res-blocks', type=int, default=12)
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    args = parser.parse_args()

    if args.single:
        if not os.path.isfile(args.single):
            print(f"ERRO: {args.single} nao existe")
            sys.exit(1)
        pt_files = [args.single]
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.single))
    else:
        if not os.path.isdir(args.checkpoints_dir):
            print(f"ERRO: pasta {args.checkpoints_dir} nao existe")
            sys.exit(1)
        pattern = os.path.join(args.checkpoints_dir, 'checkpoint_epoch*.pt')
        pt_files = sorted(glob.glob(pattern))
        if not pt_files:
            print(f"ERRO: nenhum arquivo bate em {pattern}")
            sys.exit(1)
        out_dir = args.output_dir or args.checkpoints_dir

    os.makedirs(out_dir, exist_ok=True)

    print(f"Exportando {len(pt_files)} checkpoint(s)")
    print(f"Saida: {out_dir}\n")

    for pt in pt_files:
        base = os.path.basename(pt).replace('.pt', '.onnx')
        onnx_path = os.path.join(out_dir, base)
        # erro num checkpoint não interrompe os outros
        try:
            export_one(pt, onnx_path, channels=args.channels, res_blocks=args.res_blocks, opset=args.opset)
        except Exception as e:
            print(f"  -> FALHOU: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nConcluido. Arquivos .onnx em: {out_dir}")

if __name__ == '__main__':
    main()
