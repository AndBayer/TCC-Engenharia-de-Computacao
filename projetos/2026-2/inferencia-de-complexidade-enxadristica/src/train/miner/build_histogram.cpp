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

// conta as posições do dataset em 41 faixas de tanh(cp / 600) e grava o histograma em CSV
int main(int argc, char *argv[])
{
    std::string in_str, out_str;

    auto usage = []()
    {
        std::cerr << "Uso: build_histogram --input <dataset.bin> --output <saida.csv>\n"
                  << "  -i, --input   dataset de entrada (.bin)\n"
                  << "  -o, --output  CSV de saida\n";
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

    const char *in_path = in_str.c_str();
    const char *out_path = out_str.c_str();

    FILE *f = fopen(in_path, "rb");
    if (!f)
    {
        std::cerr << "Nao abriu " << in_path << "\n";
        return 1;
    }

    // contagens em 41 faixas de 0,05 (centradas) no intervalo [-1, 1]
    std::vector<uint64_t> hist_orig(41, 0);

    const size_t buf_size = 1000000; // lê 1 milhão de registros por vez (46 MB)
    std::vector<ChessData> buf(buf_size);

    uint64_t total = 0;
    while (true)
    {
        size_t n = fread(buf.data(), sizeof(ChessData), buf_size, f);
        if (n == 0)
            break;

        for (size_t i = 0; i < n; i++)
        {
            int16_t cp = buf[i].cp;
            if (cp >= -4000 && cp <= 4000)
            {
                // mesmo valor do treino, tanh(cp / 600)
                float w = std::tanh((float)cp / 600.0f);

                // w de [-1, 1] --> índice de 0 a 40 (faixas centradas)
                int bin = std::floor((w + 1.0f) / 0.05f + 0.5f);
                if (bin < 0)
                    bin = 0;
                if (bin >= 41)
                    bin = 40;

                hist_orig[bin]++;
            }
        }
        total += n;
        if (total % 50000000 == 0)
        {
            std::cerr << "Processado: " << total << " registros...\n";
        }
    }
    fclose(f);

    std::cerr << "Total processado: " << total << "\nEscrevendo " << out_path << "...\n";
    FILE *out = fopen(out_path, "w");
    if (!out)
    {
        std::cerr << "Erro ao criar arquivo de saida: " << out_path << "\n";
        return 1;
    }
    fprintf(out, "w,freq\n");
    for (int i = 0; i <= 40; i++)
    {
        float bucket_w = -1.0f + i * 0.05f;
        fprintf(out, "%.2f,%llu\n", bucket_w, hist_orig[i]);
    }
    fclose(out);
    std::cerr << "Pronto!\n";
    return 0;
}
