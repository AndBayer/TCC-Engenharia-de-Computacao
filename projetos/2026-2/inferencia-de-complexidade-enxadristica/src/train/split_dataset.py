import os
import time
import argparse
import numpy as np

ENTRY_SIZE = 46  # tamanho de cada registro em bytes

def split_dataset(input_path, train_path, val_path, val_ratio=0.1, record_size=ENTRY_SIZE, chunk_records=1_000_000, seed=42):
    """Separa o .bin em treino e validação, sorteando o destino de cada registro; a semente fixa repete a mesma divisão."""
    np.random.seed(seed)
    print(f"Lendo: {input_path} (Seed: {seed} | Val ratio: {val_ratio:.2f})")

    total_bytes = os.path.getsize(input_path)
    total_records = total_bytes // record_size
    print(f"Tamanho total: {total_bytes / (1024**3):.2f} GB ({total_records:,} registros)")

    if total_bytes % record_size:
        print(f"AVISO: {total_bytes % record_size} bytes residuais (não múltiplo de {record_size})")

    chunk_bytes = chunk_records * record_size

    train_count = 0
    val_count = 0
    bytes_read = 0
    start_time = time.time()

    with open(input_path, 'rb') as f_in, \
         open(train_path, 'wb') as f_train, \
         open(val_path, 'wb') as f_val:

        while True:
            chunk = f_in.read(chunk_bytes)
            if not chunk:
                break

            num_records = len(chunk) // record_size

            # cada registro vira um elemento de record_size bytes (dtype void), para separar o bloco sem laço
            records = np.frombuffer(chunk, dtype=f'V{record_size}', count=num_records)

            # máscara aleatória por registro --> uma fração val_ratio (10%) vai para a validação
            mask = np.random.rand(num_records) < val_ratio

            f_val.write(records[mask].tobytes())
            f_train.write(records[~mask].tobytes())

            train_count += int(np.count_nonzero(~mask))
            val_count += int(np.count_nonzero(mask))
            bytes_read += len(chunk)

            progress = (bytes_read / total_bytes) * 100
            print(f"Progresso: {progress:.1f}% | Treino: {train_count:,} | Val: {val_count:,}", end='\r')

    print(f"\nConcluído em {time.time() - start_time:.1f} segundos!")
    print(f"Treino: {train_path} ({train_count:,} registros)")
    print(f"Val:    {val_path} ({val_count:,} registros)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split treino/validação de um dataset .bin de registros de tamanho fixo')

    parser.add_argument('-i', '--input', type=str, required=True, help='Dataset de entrada (.bin)')
    parser.add_argument('--train-path', type=str, default='dataset_train.bin', help='Saída do dataset de treino')
    parser.add_argument('--val-path', type=str, default='dataset_val.bin', help='Saída do dataset de validação')

    parser.add_argument('--val-ratio', type=float, default=0.1, help='Fração para validação')
    parser.add_argument('--record-size', type=int, default=ENTRY_SIZE, help=f'Tamanho de cada registro em bytes')
    parser.add_argument('--chunk-records', type=int, default=1_000_000, help='Registros por bloco de leitura')
    parser.add_argument('--seed', type=int, default=42, help='Seed do RNG')

    args = parser.parse_args()

    split_dataset(
        args.input, args.train_path, args.val_path,
        val_ratio=args.val_ratio, record_size=args.record_size,
        chunk_records=args.chunk_records, seed=args.seed,
    )
