// decodificador do treino, que expande os registros de 46 bytes na GPU e gera a entrada [19, 8, 8] e os alvos (lance e valor) de cada posição
// mesmos canais do motor (engine/src/inference_cuda_kernels.cu); tabuleiro já canonizado pelo miner, com 1-6 peças de quem joga e 7-12 do adversário

#include <cstdint>
#include <cuda_runtime.h>
#include <cmath>

// exporta decode_cuda como função C para o ctypes do Python
extern "C" __declspec(dllexport) void
decode_cuda(const uint8_t *input, float *output, long long *policy_out, float *value_out, int batch_size);

// lê os 4 bits de uma casa
__device__ __forceinline__ uint8_t get_piece(const uint8_t *board, int sq)
{
  uint8_t b = board[sq >> 1];              // sq / 2 --> byte que tem essa casa
  return (sq & 1) ? (b >> 4) : (b & 0x0F); // casa ímpar --> 4 bits altos; casa par --> 4 bits baixos
}

// converte o lance (promo<<12 | to<<6 | from) no índice da política (0-4223)
// mesmo layout do encoding::uciToIndex do motor
__device__ int32_t remap_move_cuda(uint16_t move)
{
  int from_sq = move & 0x3F;
  int to_sq = (move >> 6) & 0x3F;
  int promo = (move >> 12) & 0x0F;

  if (promo == 0)
  {
    return to_sq * 64 + from_sq;
  }
  else
  {
    int from_file = from_sq & 7;
    int to_file = to_sq & 7;
    int direction = to_file - from_file + 1; // 0 = captura à esquerda, 1 = avanço, 2 = captura à direita
    return 4096 + from_file * 12 + direction * 4 + (promo - 1);
  }
}

// um bloco por posição, uma thread por casa
__global__ void decode_kernel(const uint8_t *__restrict__ input, float *__restrict__ output, long long *__restrict__ policy_out, float *__restrict__ value_out, int batch)
{
  int b = blockIdx.x;   // índice da posição no lote
  int sq = threadIdx.x; // índice da casa no tabuleiro (0-63)

  if (b >= batch || sq >= 64)
    return;

#pragma unroll
  // zera os 19 canais da casa
  for (int c = 0; c < 19; c++)
    output[(b * 19 + c) * 64 + sq] = 0.0f;

  // início do registro de 46 bytes
  const uint8_t *entry = input + b * 46;

  // tabuleiro, en passant e roques compartilhados pelas 64 threads
  __shared__ uint8_t board[64];
  __shared__ uint8_t ep;
  __shared__ uint8_t rK, rQ, rk, rq;

  // só a thread 0 lê roques, en passant e alvos
  if (sq == 0)
  {
    // bytes de cada campo do registro ChessData (46 bytes)
    // zobrist 0-7, data 8-39, move 40-41, cp 42-43, flags 44, en_passant 45
    uint8_t flags = entry[44];
    ep = entry[45];
    rK = (flags & 0x01) ? 1 : 0;
    rQ = (flags & 0x02) ? 1 : 0;
    rk = (flags & 0x04) ? 1 : 0;
    rq = (flags & 0x08) ? 1 : 0;

    // alvos do treino
    uint16_t move_raw = *(uint16_t *)(entry + 40);
    int16_t cp_raw = *(int16_t *)(entry + 42);

    // a política é o índice do lance do Stockfish
    // o valor é o cp comprimido para [-1, 1]
    policy_out[b] = (long long)remap_move_cuda(move_raw);
    value_out[b] = tanhf((float)cp_raw / 600.0f);
  }

  // cada thread decodifica a sua casa (tabuleiro começa no byte 8)
  board[sq] = get_piece(entry + 8, sq);

  // espera todas as threads antes de ler casas vizinhas
  __syncthreads();

  // escreve no canal c da casa
  auto write = [&](int c, float v)
  { output[(b * 19 + c) * 64 + sq] = v; };

  // canais 0-11 com as peças
  uint8_t p = board[sq];
  if (p >= 1 && p <= 12)
    write(p - 1, 1.0f);

  // canal 12 com o en passant
  if (ep == sq && ep < 64)
    write(12, 1.0f);

  // canais 13-16 com os roques
  if (rK)
    write(13, 1.0f);
  if (rQ)
    write(14, 1.0f);
  if (rk)
    write(15, 1.0f);
  if (rq)
    write(16, 1.0f);

  // canais 17-18 com os atacantes da casa (w = quem joga, b_ = adversário)
  float w = 0.0f;
  float b_ = 0.0f;
  int r0 = sq >> 3;
  int f0 = sq & 7;

  // cavalos
  const int nd[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
#pragma unroll
  for (int i = 0; i < 8; i++)
  {
    int r = r0 + nd[i][0];
    int f = f0 + nd[i][1];
    if (r < 0 || r > 7 || f < 0 || f > 7)
      continue;
    uint8_t q = board[r * 8 + f];
    if (q == 2)
      w += 1.0f;
    if (q == 8)
      b_ += 1.0f;
  }

  // reis (kd também serve de direções para as deslizantes)
  const int kd[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
#pragma unroll
  for (int i = 0; i < 8; i++)
  {
    int r = r0 + kd[i][0];
    int f = f0 + kd[i][1];
    if (r < 0 || r > 7 || f < 0 || f > 7)
      continue;
    uint8_t q = board[r * 8 + f];
    if (q == 6)
      w += 1.0f;
    if (q == 12)
      b_ += 1.0f;
  }

  // o peão de quem joga fica em r0 - 1, o do adversário em r0 + 1
  if (r0 > 0)
  {
    if (f0 > 0 && board[(r0 - 1) * 8 + (f0 - 1)] == 1)
      w += 1.0f;
    if (f0 < 7 && board[(r0 - 1) * 8 + (f0 + 1)] == 1)
      w += 1.0f;
  }
  if (r0 < 7)
  {
    if (f0 > 0 && board[(r0 + 1) * 8 + (f0 - 1)] == 7)
      b_ += 1.0f;
    if (f0 < 7 && board[(r0 + 1) * 8 + (f0 + 1)] == 7)
      b_ += 1.0f;
  }

#pragma unroll
  // nas peças deslizantes, só conta a primeira peça em cada direção
  // diagonal = bispo e dama, reta = torre e dama
  for (int d = 0; d < 8; d++)
  {
    for (int s = 1; s < 8; s++)
    {
      int r = r0 + kd[d][0] * s;
      int f = f0 + kd[d][1] * s;
      if (r < 0 || r > 7 || f < 0 || f > 7)
        break;
      uint8_t q = board[r * 8 + f];
      if (!q)
        continue;

      bool diag = kd[d][0] && kd[d][1];
      if (diag)
      {
        if (q == 3 || q == 5)
          w += 1.0f;
        if (q == 9 || q == 11)
          b_ += 1.0f;
      }
      else
      {
        if (q == 4 || q == 5)
          w += 1.0f;
        if (q == 10 || q == 11)
          b_ += 1.0f;
      }
      break;
    }
  }

  write(17, w);
  write(18, b_);
}

// lança o kernel (função que roda na GPU) com um bloco por posição e 64 threads por bloco; os ponteiros já estão na GPU
void decode_cuda(const uint8_t *input, float *output, long long *policy_out, float *value_out, int batch)
{
  decode_kernel<<<batch, 64>>>(input, output, policy_out, value_out, batch);
}