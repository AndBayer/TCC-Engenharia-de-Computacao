#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

static constexpr size_t RECORD_SIZE = 46; // registro ChessData do miner
static constexpr size_t CHUNK_RECS = 1'000'000; // 46 MB por bloco de leitura
static constexpr size_t CHUNK_BYTES = CHUNK_RECS * RECORD_SIZE;
static constexpr size_t WBUF_BYTES = CHUNK_BYTES; // buffer de escrita (46 MB)

// funções auxiliares para arquivos grandes (> 2 GB) no Windows
#ifdef _WIN32
#define fseek64 _fseeki64
#define ftell64 _ftelli64
#else
#define fseek64 fseeko
#define ftell64 ftello
#endif

// copia o dataset sem as posições que aparecem nos puzzles, para a validação não usar posições vistas no treino
int main(int argc, char *argv[])
{
    std::string src_str, hashes_str, dst_str = "dataset_train.bin";

    auto usage = []()
    {
        fprintf(stderr,
                "Uso: filter_dataset --src <dataset.bin> --hashes <puzzle_hashes.bin> [--output <dataset_train.bin>]\n"
                "  -s, --src     dataset de entrada (.bin)\n"
                "  -H, --hashes  binario de hashes de puzzle (.bin)\n"
                "  -o, --output  dataset filtrado de saida (default: dataset_train.bin)\n");
    };

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto val = [&](const std::string &f) -> std::string
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "ERRO: faltou valor para %s\n", f.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-s" || a == "--src")
            src_str = val(a);
        else if (a == "-H" || a == "--hashes")
            hashes_str = val(a);
        else if (a == "-o" || a == "--output")
            dst_str = val(a);
        else if (a == "-h" || a == "--help")
        {
            usage();
            return 0;
        }
        else
        {
            fprintf(stderr, "ERRO: argumento desconhecido: %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (src_str.empty() || hashes_str.empty())
    {
        fprintf(stderr, "ERRO: --src e --hashes sao obrigatorios\n");
        usage();
        return 1;
    }

    const char *src_path = src_str.c_str();
    const char *hashes_path = hashes_str.c_str();
    const char *dst_path = dst_str.c_str();

    // carrega os hashes dos puzzles
    FILE *fh = fopen(hashes_path, "rb");
    if (!fh)
    {
        fprintf(stderr, "ERRO: nao abriu %s\n", hashes_path);
        return 1;
    }
    fseek64(fh, 0, SEEK_END);
    const uint64_t hsize = (uint64_t)ftell64(fh);
    const uint64_t n_hashes = hsize / sizeof(uint64_t);
    rewind(fh);
    std::vector<uint64_t> hashes(n_hashes);
    fread(hashes.data(), sizeof(uint64_t), (size_t)n_hashes, fh);
    fclose(fh);
    fprintf(stderr, "[filter] %llu hashes de puzzle carregados (%.0f MB)\n", n_hashes, (double)hsize / 1e6);

    // abre o dataset de entrada
    FILE *fin = fopen(src_path, "rb");
    if (!fin)
    {
        fprintf(stderr, "ERRO: nao abriu %s\n", src_path);
        return 1;
    }
    fseek64(fin, 0, SEEK_END);
    const uint64_t src_size = (uint64_t)ftell64(fin);
    const uint64_t total_rec = src_size / RECORD_SIZE;
    rewind(fin);
    if (src_size % RECORD_SIZE)
        fprintf(stderr, "[filter] AVISO: %llu bytes residuais em %s\n", (uint64_t)(src_size % RECORD_SIZE), src_path);
    fprintf(stderr, "[filter] dataset : %s  (%llu registros, %.2f GB)\n", src_path, total_rec, (double)src_size / 1e9);

    // abre a saída
    FILE *fout = fopen(dst_path, "wb");
    if (!fout)
    {
        fprintf(stderr, "ERRO: nao abriu %s\n", dst_path);
        return 1;
    }
    fprintf(stderr, "[filter] saida   : %s\n\n", dst_path);

    // processa em blocos
    static uint8_t chunk[CHUNK_BYTES]; // buffer de leitura estático, fora da pilha
    static uint8_t wbuf[WBUF_BYTES];   // buffer de escrita

    uint64_t n_read = 0;
    uint64_t n_written = 0;
    uint64_t n_excluded = 0;
    size_t wbuf_used = 0;

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    while (true)
    {
        const size_t bytes_read = fread(chunk, 1, CHUNK_BYTES, fin);
        if (bytes_read == 0)
            break;
        const size_t recs = bytes_read / RECORD_SIZE;

        for (size_t i = 0; i < recs; i++)
        {
            const uint8_t *rec = chunk + i * RECORD_SIZE;

            // zobrist = primeiros 8 bytes do registro (little-endian)
            uint64_t z;
            memcpy(&z, rec, 8);

            // o puzzle_hasher grava os hashes ordenados, então a busca binária funciona
            if (std::binary_search(hashes.cbegin(), hashes.cend(), z))
            {
                n_excluded++;
            }
            else
            {
                memcpy(wbuf + wbuf_used, rec, RECORD_SIZE);
                wbuf_used += RECORD_SIZE;
                n_written++;
                // grava quando o buffer enche
                if (wbuf_used + RECORD_SIZE > WBUF_BYTES)
                {
                    fwrite(wbuf, 1, wbuf_used, fout);
                    wbuf_used = 0;
                }
            }
        }

        n_read += (uint64_t)recs;

        // progresso a cada ~10 milhões de registros
        if (n_read % 10000000 < (uint64_t)recs)
        {
            const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
            const double pct = 100.0 * (double)n_read / (double)total_rec;
            const double rate = (double)n_read / elapsed;
            const double eta = ((double)total_rec - (double)n_read) / rate;
            fprintf(stderr,
                    "[filter] %12llu / %llu  (%.1f%%)  excluidos=%llu  escritos=%llu  %.1f M rec/s  ETA %.1f min\n", n_read, total_rec, pct, n_excluded, n_written, rate / 1e6, eta / 60.0);
        }
    }

    // grava o que sobrou no buffer
    if (wbuf_used > 0)
        fwrite(wbuf, 1, wbuf_used, fout);

    fclose(fin);
    fclose(fout);

    // tamanho da saída
    FILE *fo2 = fopen(dst_path, "rb");
    fseek64(fo2, 0, SEEK_END);
    const uint64_t out_size = (uint64_t)ftell64(fo2);
    fclose(fo2);

    const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, "[filter] CONCLUIDO em %.1f min\n", elapsed / 60.0);
    fprintf(stderr, "[filter] Registros processados : %12llu\n", n_read);
    fprintf(stderr, "[filter] Excluidos (puzzles)   : %12llu  (%.2f%%)\n", n_excluded, 100.0 * (double)n_excluded / (double)n_read);
    fprintf(stderr, "[filter] Escritos (treino)     : %12llu  (%.2f%%)\n", n_written, 100.0 * (double)n_written / (double)n_read);
    fprintf(stderr, "[filter] Tamanho saida         : %.2f GB  -> %s\n", (double)out_size / 1e9, dst_path);
    fprintf(stderr, "============================================================\n");

    return 0;
}
