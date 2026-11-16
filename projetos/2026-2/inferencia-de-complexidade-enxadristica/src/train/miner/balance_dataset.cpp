#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <string>
#include <random>

// registro de 46 bytes do dataset.bin, o mesmo do miner.cpp; posição canonizada (espelhada quando as pretas jogam)
#pragma pack(push, 1)
struct ChessData
{
    uint64_t zobrist;   // 8   hash zobrist da posição canonizada (identifica a posição em 64 bits)
    uint8_t data[32];   // 32  tabuleiro canonizado, 4 bits por casa
    uint16_t move;      // 2   melhor lance (promo<<12 | to<<6 | from)
    int16_t cp;         // 2   avaliação em centipawns (centésimos de peão) na visão de quem joga (+-4000 = mate, 0 = igual)
    uint8_t flags;      // 1   bit0-3 = roques KQkq (canônicos), bit4 = turno original
    uint8_t en_passant; // 1   casa de en passant (0-63) ou 255 = nenhuma
};
#pragma pack(pop)
static_assert(sizeof(ChessData) == 46, "ChessData must be 46 bytes");

// descarta ao acaso parte das posições com valor perto de 0, o pico do histograma
int main(int argc, char *argv[])
{
    std::string in_str, out_str;

    auto usage = []()
    {
        std::cerr << "Uso: balance_dataset --input <dataset_in.bin> --output <dataset_out.bin>\n"
                  << "  -i, --input   dataset de entrada (.bin)\n"
                  << "  -o, --output  dataset balanceado de saida (.bin)\n";
    };

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto val = [&](const std::string &fl) -> std::string
        {
            if (i + 1 >= argc)
            {
                std::cerr << "ERRO: faltou valor para " << fl << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-i" || a == "--input")
            in_str = val(a);
        else if (a == "-o" || a == "--output")
            out_str = val(a);
        else if (a == "-h" || a == "--help")
        {
            usage();
            return 0;
        }
        else
        {
            std::cerr << "ERRO: argumento desconhecido: " << a << "\n";
            usage();
            return 2;
        }
    }
    if (in_str.empty() || out_str.empty())
    {
        std::cerr << "ERRO: --input e --output sao obrigatorios\n";
        usage();
        return 1;
    }

    FILE *fin = fopen(in_str.c_str(), "rb");
    if (!fin)
    {
        std::cerr << "Erro ao abrir entrada: " << in_str << "\n";
        return 1;
    }

    FILE *fout = fopen(out_str.c_str(), "wb");
    if (!fout)
    {
        std::cerr << "Erro ao criar saida: " << out_str << "\n";
        fclose(fin);
        return 1;
    }

    // semente fixa, então o dataset balanceado sai sempre igual
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    const size_t buf_size = 1000000;
    std::vector<ChessData> read_buf(buf_size);
    std::vector<ChessData> write_buf(buf_size);

    uint64_t total_read = 0;
    uint64_t total_written = 0;

    std::cerr << "Iniciando balanceamento (Gaussian Drop no pico W=0)...\n";

    while (true)
    {
        size_t n = fread(read_buf.data(), sizeof(ChessData), buf_size, fin);
        if (n == 0)
            break;

        size_t w_count = 0;
        for (size_t i = 0; i < n; i++)
        {
            int16_t cp = read_buf[i].cp;

            // mesmo valor do treino, tanh(cp / 600)
            float w = std::tanh((float)cp / 600.0f);

            // curva gaussiana invertida; no pico (w = 0) descarta ~67%, e o descarte cai para 0 longe dele
            // P_drop(w) = 0.67 * exp(-w^2 / (2 * 0.15^2))
            // curva suave para a rede não aprender o corte
            float drop_prob = 0.67f * std::exp(-(w * w) / (2.0f * 0.15f * 0.15f));
            float keep_prob = 1.0f - drop_prob;

            if (dis(gen) < keep_prob)
            {
                write_buf[w_count++] = read_buf[i];
            }
        }

        if (w_count > 0)
        {
            fwrite(write_buf.data(), sizeof(ChessData), w_count, fout);
            total_written += w_count;
        }

        total_read += n;
        if (total_read % 20000000 == 0)
        {
            std::cerr << "Lidos: " << total_read << " | Mantidos: " << total_written << " (" << (total_written * 100 / total_read) << "%)\n";
        }
    }

    fclose(fin);
    fclose(fout);

    std::cerr << "==========================================\n";
    std::cerr << "Concluido!\n";
    std::cerr << "Total Lidos    : " << total_read << "\n";
    std::cerr << "Total Mantidos : " << total_written << " (" << (total_written * 100 / total_read) << "%)\n";
    std::cerr << "Total Excluidos: " << (total_read - total_written) << "\n";

    return 0;
}
