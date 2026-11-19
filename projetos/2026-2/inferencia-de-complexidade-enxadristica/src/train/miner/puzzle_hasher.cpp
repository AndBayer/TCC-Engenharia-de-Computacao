#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <bit> // define __cpp_lib_bitops --> chess.hpp usa std::popcount em vez de _mm_popcnt_u64
#include "chess.hpp"

// hash zobrist da posição canonizada (espelhada quando as pretas jogam), com a mesma conta do miner.cpp, para casar com o campo zobrist do dataset
static uint64_t canonical_zobrist(const chess::Board &board)
{
    const bool flip = (board.sideToMove() == chess::Color::BLACK);
    uint64_t z = 0;

    for (int isq = 0; isq < 64; isq++)
    {
        const auto p = board.at(chess::Square(isq));
        if (p == chess::Piece::NONE)
            continue;

        const int pt = static_cast<int>(p.type().internal());   // 0-5
        const int col = static_cast<int>(p.color().internal()); // 0=WHITE 1=BLACK

        const int can_sq = flip ? (isq ^ 56) : isq;
        const int can_col = flip ? (1 - col) : col;

        z ^= chess::Zobrist::RANDOM_ARRAY[chess::Zobrist::MAP_HASH_PIECE[can_col * 6 + pt] * 64 + can_sq];
    }

    // roques canonizados
    using Side = chess::Board::CastlingRights::Side;
    const auto cr = board.castlingRights();
    bool K_can, Q_can, k_can, q_can;
    if (flip)
    {
        K_can = cr.has(chess::Color::BLACK, Side::KING_SIDE);
        Q_can = cr.has(chess::Color::BLACK, Side::QUEEN_SIDE);
        k_can = cr.has(chess::Color::WHITE, Side::KING_SIDE);
        q_can = cr.has(chess::Color::WHITE, Side::QUEEN_SIDE);
    }
    else
    {
        K_can = cr.has(chess::Color::WHITE, Side::KING_SIDE);
        Q_can = cr.has(chess::Color::WHITE, Side::QUEEN_SIDE);
        k_can = cr.has(chess::Color::BLACK, Side::KING_SIDE);
        q_can = cr.has(chess::Color::BLACK, Side::QUEEN_SIDE);
    }
    if (K_can)
        z ^= chess::Zobrist::RANDOM_ARRAY[768];
    if (Q_can)
        z ^= chess::Zobrist::RANDOM_ARRAY[769];
    if (k_can)
        z ^= chess::Zobrist::RANDOM_ARRAY[770];
    if (q_can)
        z ^= chess::Zobrist::RANDOM_ARRAY[771];

    // en passant canonizado (só a coluna importa)
    const auto ep = board.enpassantSq();
    if (ep != chess::Square::NO_SQ)
    {
        const int ep_idx = ep.index();
        const int ep_can = flip ? (ep_idx ^ 56) : ep_idx;
        z ^= chess::Zobrist::RANDOM_ARRAY[772 + (ep_can % 8)];
    }

    return z;
}

// extrai os campos 1 (FEN) e 2 (Moves) de uma linha do CSV
static bool parse_csv_line(const std::string &line, std::string &fen, std::string &moves_str)
{
    int field = 0;
    size_t start = 0;
    for (size_t i = 0; i <= line.size(); i++)
    {
        if (i == line.size() || line[i] == ',')
        {
            if (field == 1)
                fen = line.substr(start, i - start);
            if (field == 2)
            {
                moves_str = line.substr(start, i - start);
                return true;
            }
            field++;
            start = i + 1;
        }
    }
    return false;
}

// lê o CSV de puzzles do Lichess e grava o zobrist canônico de cada posição dos puzzles, ordenado e sem repetição
int main(int argc, char *argv[])
{
    std::string out_path = "puzzle_hashes.bin";
    std::string input_path; // vazio = stdin

    auto usage = []()
    {
        std::fprintf(stderr,
                     "Uso: puzzle_hasher [--output <hashes.bin>] [--input <puzzles.csv>]\n"
                     "  -o, --output  binario de hashes de saida (default: puzzle_hashes.bin)\n"
                     "  -i, --input   CSV de puzzles (default: stdin)\n");
    };

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto val = [&](const std::string &fl) -> std::string
        {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "ERRO: faltou valor para %s\n", fl.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-o" || a == "--output")
            out_path = val(a);
        else if (a == "-i" || a == "--input")
            input_path = val(a);
        else if (a == "-h" || a == "--help")
        {
            usage();
            return 0;
        }
        else
        {
            std::fprintf(stderr, "ERRO: argumento desconhecido: %s\n", a.c_str());
            usage();
            return 2;
        }
    }

    // puzzles vêm do arquivo (--input) ou do stdin
    std::ifstream input_file;
    if (!input_path.empty())
    {
        input_file.open(input_path);
        if (!input_file)
        {
            std::fprintf(stderr, "ERRO: nao abriu o CSV: %s\n", input_path.c_str());
            return 1;
        }
    }
    std::istream &in = input_path.empty() ? std::cin : input_file;

    std::vector<uint64_t> hashes;
    hashes.reserve(30'000'000);

    std::string line;
    bool header_skipped = false;
    size_t n_puzzles = 0;
    size_t last_report = 0;
    constexpr size_t REPORT_EVERY = 100'000;

    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    std::fprintf(stderr, "[puzzle_hasher] lendo %s -> %s\n", input_path.empty() ? "stdin" : input_path.c_str(), out_path.c_str());

    while (std::getline(in, line))
    {
        if (!header_skipped)
        {
            header_skipped = true;
            continue;
        } // pula o cabeçalho
        if (line.empty())
            continue;

        std::string fen, moves_str;
        if (!parse_csv_line(line, fen, moves_str))
            continue;

        chess::Board board;
        try
        {
            board = chess::Board(fen);
        }
        catch (...)
        {
            continue;
        }

        // posição inicial (antes do lance forçador)
        hashes.push_back(canonical_zobrist(board));

        // aplica cada lance e coleta a posição resultante
        std::istringstream ss(moves_str);
        std::string uci;
        while (ss >> uci)
        {
            const auto move = chess::uci::uciToMove(board, uci);
            if (move == chess::Move::NO_MOVE)
                break;
            board.makeMove(move);
            hashes.push_back(canonical_zobrist(board));
        }

        n_puzzles++;
        if (n_puzzles - last_report >= REPORT_EVERY)
        {
            last_report = n_puzzles;
            const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
            const double rate = n_puzzles / elapsed;
            std::fprintf(stderr, "[puzzle_hasher] %8zu puzzles  %10zu hashes  %.1f k puzzles/s\n", n_puzzles, hashes.size(), rate / 1000.0);
        }
    }

    const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
    std::fprintf(stderr, "[puzzle_hasher] %zu puzzles processados, %zu hashes brutos (%.1fs)\n", n_puzzles, hashes.size(), elapsed);

    // ordena e remove duplicatas
    std::fprintf(stderr, "[puzzle_hasher] ordenando e deduplicando...\n");
    std::sort(hashes.begin(), hashes.end());
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
    std::fprintf(stderr, "[puzzle_hasher] %zu posicoes unicas  (%.1f MB)\n", hashes.size(), hashes.size() * 8.0 / 1e6);

    // grava em binário, cada hash como uint64_t little-endian
    std::ofstream out(out_path, std::ios::binary);
    if (!out)
    {
        std::fprintf(stderr, "ERRO: nao foi possivel abrir %s\n", out_path.c_str());
        return 1;
    }
    out.write(reinterpret_cast<const char *>(hashes.data()), static_cast<std::streamsize>(hashes.size() * sizeof(uint64_t)));
    std::fprintf(stderr, "[puzzle_hasher] salvo em %s\n", out_path.c_str());

    return 0;
}
