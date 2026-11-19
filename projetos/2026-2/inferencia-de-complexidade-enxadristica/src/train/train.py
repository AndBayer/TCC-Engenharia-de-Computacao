import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import ctypes
from tqdm import tqdm
import os
import csv
import argparse

ENTRY_SIZE = 46  # bytes por registro do dataset (ChessData do miner)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda_lib = None  # decoder.dll, carregada em init_cuda_lib

def init_cuda_lib(dll_path):
    """Carrega a DLL do decoder (decoder.cu) e declara os argumentos de decode_cuda."""
    global cuda_lib
    cuda_lib = ctypes.CDLL(dll_path)
    cuda_lib.decode_cuda.argtypes = [
        ctypes.c_void_p, # input
        ctypes.c_void_p, # output
        ctypes.c_void_p, # policy_out
        ctypes.c_void_p, # value_out
        ctypes.c_int     # batch_size
    ]

class ChessIterableDataset(IterableDataset):
    """Lê o .bin em blocos e entrega lotes de batch_size registros crus (46 bytes cada); o lote incompleto do fim é descartado."""
    def __init__(self, file_path, batch_size=1024, chunk_size=102400, start_record=0, end_record=None):
        self.file_path = file_path
        self.entry_size = ENTRY_SIZE
        self.batch_size = batch_size
        self.chunk_size = (chunk_size // batch_size) * batch_size
        self.start_record = start_record
        self.end_record = end_record

    def __iter__(self):
        with open(self.file_path, 'rb') as f:
            f.seek(self.start_record * self.entry_size)
            bytes_limit = (
                (self.end_record - self.start_record) * self.entry_size
                if self.end_record is not None else None
            )
            bytes_consumed = 0

            while True:
                if bytes_limit is not None:
                    remaining = bytes_limit - bytes_consumed
                    if remaining <= 0:
                        break
                    to_read = min(self.chunk_size * self.entry_size, remaining)
                else:
                    to_read = self.chunk_size * self.entry_size

                raw = f.read(to_read)
                if not raw:
                    break

                bytes_consumed += len(raw)
                n = len(raw) // self.entry_size
                chunk = np.frombuffer(raw, dtype=np.uint8).reshape((n, self.entry_size)).copy()

                for i in range(0, n, self.batch_size):
                    batch = chunk[i : i + self.batch_size]
                    if batch.shape[0] < self.batch_size:
                        break
                    yield batch

class SEBlock(nn.Module):
    """Bloco SE --> calcula um peso de 0 a 1 para cada canal, a partir das médias de todos os canais, e multiplica cada canal pelo seu peso."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResSEBlock(nn.Module):
    """Bloco residual --> duas convoluções 3x3 com BatchNorm, bloco SE e soma com a entrada."""
    def __init__(self, channels):
        super(ResSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)

class ChessNet(nn.Module):
    """Entrada com 19 planos 8x8; saídas --> política com 4224 logits e valor entre -1 e 1."""
    def __init__(self, channels=128, num_res_blocks=12):
        super(ChessNet, self).__init__()
        self.conv_inicial = nn.Conv2d(19, channels, kernel_size=3, padding=1, bias=False)
        self.bn_inicial = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *[ResSEBlock(channels) for _ in range(num_res_blocks)]
        )
        # 66 planos de 8x8 achatados = 4224 saídas, na ordem dos índices do encoding.hpp
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 66, kernel_size=1),
            nn.Flatten()
        )
        # 1 plano de 8x8 --> 64 --> 128 --> 1, com Tanh no fim
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.bn_inicial(self.conv_inicial(x)))
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

def custom_loss(policy_pred, policy_target, value_pred, value_target):
    """Perda --> entropia cruzada da política + erro quadrático do valor, com o mesmo peso."""
    loss_policy = F.cross_entropy(policy_pred, policy_target)
    loss_value = F.mse_loss(value_pred.view(-1), value_target.float())
    total_loss = loss_policy + loss_value
    return total_loss, loss_policy, loss_value

def run_batch(model, compact_batch_numpy):
    """Manda o lote cru para a GPU, decodifica com a DLL, roda a rede e calcula a perda."""
    # o DataLoader já converteu o array do lote em tensor uint8 [B, 46]
    compact_gpu = compact_batch_numpy.to(DEVICE).contiguous()
    current_batch_size = compact_gpu.shape[0]

    inflated_gpu = torch.empty((current_batch_size, 19, 64), dtype=torch.float32, device=DEVICE)
    policy_target = torch.empty(current_batch_size, dtype=torch.long, device=DEVICE)
    value_target = torch.empty(current_batch_size, dtype=torch.float32, device=DEVICE)

    # a DLL recebe os ponteiros da GPU e preenche a entrada da rede, o lance alvo e o valor alvo
    cuda_lib.decode_cuda(
        compact_gpu.data_ptr(),
        inflated_gpu.data_ptr(),
        policy_target.data_ptr(),
        value_target.data_ptr(),
        current_batch_size
    )

    tabuleiros_input = inflated_gpu.view(current_batch_size, 19, 8, 8)

    # precisão mista --> o PyTorch usa float16 nas operações em que é seguro
    with torch.amp.autocast('cuda'):
        policy_pred, value_pred = model(tabuleiros_input)
        loss, l_ce, l_mse = custom_loss(
            policy_pred, policy_target, value_pred, value_target
        )

    return loss, l_ce, l_mse

def train(args):
    """A cada época --> treina, valida, grava o log e salva o checkpoint."""
    init_cuda_lib(args.decoder_path)

    model = ChessNet(channels=args.channels, num_res_blocks=args.res_blocks).to(DEVICE)
    # base_model é a rede sem o torch.compile (que gera uma versão mais rápida da mesma rede), usada para salvar e carregar os pesos
    base_model = model
    if args.compile:
        torch._dynamo.config.cache_size_limit = 64
        model = torch.compile(model, dynamic=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # escala a perda para os gradientes (ajustes dos pesos) em float16 não zerarem
    scaler = torch.amp.GradScaler()

    # registros de treino e de validação, em arquivos separados pelo split_dataset.py
    n_train = os.path.getsize(args.train_path) // ENTRY_SIZE
    n_val = os.path.getsize(args.test_path) // ENTRY_SIZE

    # --max-samples limita os registros --> 90% do limite no treino e 10% na validação
    if args.max_samples:
        n_train = min(n_train, int(args.max_samples * 0.9))
        n_val = min(n_val, int(args.max_samples * 0.1))

    # retoma do último checkpoint
    start_epoch = 0
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE, weights_only=True)
        base_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']

        # ao retomar, a taxa de aprendizado vem do --lr, não do checkpoint
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        print(f"Resumindo da época {start_epoch + 1} com LR={args.lr}!")

    total_train_batches = n_train // args.batch_size
    total_val_batches = n_val // args.batch_size

    # perdas médias de treino e de validação por época
    log_path = 'training_log.csv'
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch',
                'train_ce', 'train_mse',
                'val_ce',   'val_mse'
            ])

    print(f"Dataset Total Usado: {n_train + n_val:,} registros | Treino: {n_train:,} | Val: {n_val:,}")
    print("Iniciando treinamento...")

    for epoch in range(start_epoch, args.epochs):
        # treino da época
        train_dataset = ChessIterableDataset(
            args.train_path, 
            batch_size=args.batch_size,
            start_record=0, 
            end_record=n_train
        )
        train_loader = DataLoader(train_dataset, batch_size=None)

        model.train()
        pbar = tqdm(enumerate(train_loader), total=total_train_batches, desc=f"Treino {epoch+1}/{args.epochs}", unit="batch")

        # soma as perdas na GPU, sem sincronizar com a CPU a cada lote
        sum_train_ce  = torch.zeros((), device=DEVICE)
        sum_train_mse = torch.zeros((), device=DEVICE)
        n_train_steps = 0

        for _, compact_batch_numpy in pbar:
            loss, l_ce, l_mse = run_batch(model, compact_batch_numpy)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            sum_train_ce += l_ce.detach()
            sum_train_mse += l_mse.detach()
            n_train_steps += 1

            # sincroniza com a CPU só a cada log_every lotes, para mostrar as perdas
            if n_train_steps % args.log_every == 0:
                pbar.set_postfix({
                    'CE': f"{l_ce.item():.3f}",
                    'MSE': f"{l_mse.item():.3f}"
                })

        avg_train_ce = (sum_train_ce / max(n_train_steps, 1)).item()
        avg_train_mse = (sum_train_mse / max(n_train_steps, 1)).item()

        # validação da época 
        val_dataset = ChessIterableDataset(
            args.test_path, 
            batch_size=args.batch_size,
            start_record=0, 
            end_record=n_val
        )
        val_loader = DataLoader(val_dataset, batch_size=None)

        model.eval()
        pbar_val = tqdm(val_loader, total=total_val_batches, desc=f"Val {epoch+1}/{args.epochs}", unit="batch")

        sum_val_ce  = torch.zeros((), device=DEVICE)
        sum_val_mse = torch.zeros((), device=DEVICE)
        n_val_steps = 0

        with torch.no_grad():
            for compact_batch_numpy in pbar_val:
                _, l_ce, l_mse = run_batch(model, compact_batch_numpy)
                sum_val_ce  += l_ce.detach()
                sum_val_mse += l_mse.detach()
                n_val_steps += 1
                if n_val_steps % args.log_every == 0:
                    pbar_val.set_postfix({
                        'CE': f"{l_ce.item():.3f}",
                        'MSE': f"{l_mse.item():.3f}"
                    })

        avg_val_ce = (sum_val_ce / max(n_val_steps, 1)).item()
        avg_val_mse = (sum_val_mse / max(n_val_steps, 1)).item()

        # grava as perdas médias da época no log 
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1,
                f"{avg_train_ce:.6f}", 
                f"{avg_train_mse:.6f}",
                f"{avg_val_ce:.6f}",
                f"{avg_val_mse:.6f}"
            ])

        print(f"Época {epoch+1}: "
              f"train_ce={avg_train_ce:.4f}  train_mse={avg_train_mse:.4f} | "
              f"val_ce={avg_val_ce:.4f}  val_mse={avg_val_mse:.4f}")

        # checkpoint_latest.pt para retomar o treino e um arquivo por época
        checkpoint = {
            'epoch': epoch + 1,
            'model': base_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
        }
        torch.save(checkpoint, 'checkpoint_latest.pt')
        torch.save(checkpoint, f'checkpoint_epoch{epoch+1}.pt')
        print(f"Modelo salvo: checkpoint_epoch{epoch+1}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinamento da Rede Neural AlphaZero para Xadrez')
    
    parser.add_argument('--train-path', type=str, required=True, help='Caminho do dataset de treino')
    parser.add_argument('--test-path', type=str, required=True, help='Caminho do dataset de teste')
    parser.add_argument('--decoder-path', type=str, default=os.path.join(os.path.dirname(__file__), '..', '..', 'bin', 'decoder.dll'), help='Caminho para a DLL do CUDA (default: <projeto>/bin/decoder.dll)')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint_latest.pt', help='Caminho de restore')
    
    parser.add_argument('--batch-size', type=int, default=4096, help='Tamanho do batch')
    parser.add_argument('--epochs', type=int, default=12, help='Número de épocas')
    parser.add_argument('--lr', type=float, default=1e-3, help='Taxa de aprendizado (Learning Rate)')
    parser.add_argument('--max-samples', type=int, default=None, help='Limitar registros')

    parser.add_argument('--channels', type=int, default=128, help='Canais da ResNet')
    parser.add_argument('--res-blocks', type=int, default=12, help='Blocos residuais')
    parser.add_argument('--compile', action='store_true', help='Ativa torch.compile')
    parser.add_argument('--log-every', type=int, default=50, help='Atualiza a barra de progresso a cada N batches')
    
    args = parser.parse_args()
        
    train(args)
