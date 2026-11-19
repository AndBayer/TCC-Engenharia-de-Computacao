#include <iostream>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <climits>
#include <algorithm>

#include <bit> // define __cpp_lib_bitops --> chess.hpp usa std::popcount em vez de _mm_popcnt_u64
#include "chess.hpp"

constexpr int CP_CLAMP = 4000; // cp limitado a +-4000; mate vira +-4000
constexpr int MAX_EVALS = 10;  // máximo de avaliações lidas por linha do JSON

// bits de ChessData::flags
constexpr uint8_t FLAG_CASTLE_K = 0x01; // roque pequeno das brancas (canônico)
constexpr uint8_t FLAG_CASTLE_Q = 0x02; // roque grande das brancas (canônico)
constexpr uint8_t FLAG_CASTLE_k = 0x04; // roque pequeno das pretas (canônico)
constexpr uint8_t FLAG_CASTLE_q = 0x08; // roque grande das pretas (canônico)
constexpr uint8_t FLAG_TURN = 0x10;     // turno original (0=brancas, 1=pretas)

// hash zobrist --> número de 64 bits que identifica a posição; é o XOR de um número aleatório fixo para cada peça na sua casa, cada roque e o en passant
// os números aleatórios ficam no chess::Zobrist::RANDOM_ARRAY (chess.hpp), com 781 valores:
//   0-767 --> 12 peças x 64 casas, um bloco de 64 por peça, na ordem p, P, n, N, b, B, r, R, q, Q, k, K
//   768-771 --> roques K, Q, k, q; 772-779 --> en passant, um por coluna; 780 --> vez de jogar, não usada (a posição é sempre espelhada para as brancas jogarem)
// NIBBLE_ZOFFSET --> início do bloco de cada peça, pelo código do miner (1-6 = P N B R Q K, 7-12 = p n b r q k); ex.: cavalo branco (2) em e4 (casa 28) usa RANDOM_ARRAY[192 + 28]
static constexpr int NIBBLE_ZOFFSET[13] = {-1, 64, 192, 320, 448, 576, 704, 0, 128, 256, 384, 512, 640};

// troca a cor da peça (1-6 <--> 7-12)
static constexpr uint8_t FLIP_PIECE[13] = {0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6};
// tabela caractere ASCII --> código da peça (4 bits), preenchida por initCharToNibble
static uint8_t CHAR_TO_NIBBLE[128];

// letra da FEN --> código da peça (1-6 brancas, 7-12 pretas); o resto fica 0
static void initCharToNibble()
{
    std::memset(CHAR_TO_NIBBLE, 0, sizeof(CHAR_TO_NIBBLE));
    CHAR_TO_NIBBLE['P'] = 1;
    CHAR_TO_NIBBLE['N'] = 2;
    CHAR_TO_NIBBLE['B'] = 3;
    CHAR_TO_NIBBLE['R'] = 4;
    CHAR_TO_NIBBLE['Q'] = 5;
    CHAR_TO_NIBBLE['K'] = 6;
    CHAR_TO_NIBBLE['p'] = 7;
    CHAR_TO_NIBBLE['n'] = 8;
    CHAR_TO_NIBBLE['b'] = 9;
    CHAR_TO_NIBBLE['r'] = 10;
    CHAR_TO_NIBBLE['q'] = 11;
    CHAR_TO_NIBBLE['k'] = 12;
}

// registro de 46 bytes do dataset.bin; posição canonizada (espelhada quando as pretas jogam)
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

// acha key no buffer e devolve o ponteiro logo depois dela (nullptr se não achar)
static inline const char *seek(const char *buf, const char *key, size_t klen)
{
    const char *p = strstr(buf, key);
    return p ? p + klen : nullptr;
}

// acha needle só entre start e end (nullptr se não achar)
static inline const char *find_str(const char *start, const char *end, const char *needle, size_t nlen)
{
    if (nlen == 0)
        return start;
    for (const char *p = start; p + nlen <= end; ++p)
    {
        if (*p == needle[0] && std::memcmp(p, needle, nlen) == 0)
            return p;
    }
    return nullptr;
}

// lê um inteiro com sinal; ep recebe onde a leitura parou
static inline int readInt(const char *p, const char **ep = nullptr)
{
    int sign = 1, v = 0;
    if (*p == '-')
    {
        sign = -1;
        ++p;
    }
    while ((unsigned)(*p - '0') <= 9u)
        v = v * 10 + (*p++ - '0');
    if (ep)
        *ep = p;
    return sign * v;
}

// uma linha (pv) da avaliação --> cp ou mate e o primeiro lance
struct RawPV
{
    int cp;
    int mate_v;
    char move[6];
};

// uma avaliação do JSON --> até 5 linhas, profundidade e knodes
struct EvalEntry
{
    RawPV pvs[5];
    int n_pvs;
    int depth;
    uint32_t knodes;
};

// lê fen e evals de uma linha do JSON do Lichess sem biblioteca de JSON; avaliação sem pvs é ignorada
static bool fastParse(const char *raw, char fen_out[128], EvalEntry evals_out[MAX_EVALS], int &n_evals_out)
{
    const char *p = seek(raw, "\"fen\":\"", 7);
    if (!p)
        return false;
    int fi = 0;
    while (*p && *p != '"' && fi < 127)
        fen_out[fi++] = *p++;
    fen_out[fi] = 0;
    if (!fi)
        return false;

    p = seek(raw, "\"evals\":[", 9);
    if (!p)
        return false;

    n_evals_out = 0;
    while (n_evals_out < MAX_EVALS && *p)
    {
        while (*p && *p != '{' && *p != ']')
            ++p;
        if (!*p || *p == ']')
            break;

        const char *obj = p++;
        int d = 1;
        while (*p && d > 0)
        {
            if (*p == '{')
                ++d;
            else if (*p == '}')
                --d;
            ++p;
        }
        const char *obj_end = p;

        EvalEntry &ev = evals_out[n_evals_out];
        ev.n_pvs = 0;
        ev.depth = 0;
        ev.knodes = 0;

        const char *pvp = find_str(obj, obj_end, "\"pvs\":[", 7);
        if (pvp && pvp < obj_end)
        {
            pvp += 7;
            while (ev.n_pvs < 5 && pvp < obj_end)
            {
                while (pvp < obj_end && *pvp != '{' && *pvp != ']')
                    ++pvp;
                if (pvp >= obj_end || *pvp == ']')
                    break;

                const char *pv_obj = pvp++;
                int d2 = 1;
                while (pvp < obj_end && d2 > 0)
                {
                    if (*pvp == '{')
                        ++d2;
                    else if (*pvp == '}')
                        --d2;
                    ++pvp;
                }
                const char *pv_end = pvp;

                RawPV &pv = ev.pvs[ev.n_pvs];
                pv.cp = INT_MIN;
                pv.mate_v = INT_MIN;
                pv.move[0] = 0;

                const char *cp_p = find_str(pv_obj, pv_end, "\"cp\":", 5);
                if (cp_p)
                    pv.cp = readInt(cp_p + 5);

                const char *mt_p = find_str(pv_obj, pv_end, "\"mate\":", 7);
                if (mt_p)
                    pv.mate_v = readInt(mt_p + 7);

                const char *ln_p = find_str(pv_obj, pv_end, "\"line\":\"", 8);
                if (ln_p)
                {
                    ln_p += 8;
                    int mi = 0;
                    while (*ln_p && *ln_p != '"' && *ln_p != ' ' && mi < 5)
                        pv.move[mi++] = *ln_p++;
                    pv.move[mi] = 0;
                }
                if (pv.move[0])
                    ++ev.n_pvs;
            }
        }

        if (ev.n_pvs == 0)
        {
            if (*p == ',')
                ++p;
            continue;
        }

        const char *dp = find_str(obj, obj_end, "\"depth\":", 8);
        if (dp)
            ev.depth = readInt(dp + 8);
        const char *kp = find_str(obj, obj_end, "\"knodes\":", 9);
        if (kp)
            ev.knodes = (uint32_t)readInt(kp + 9);

        ++n_evals_out;
        if (*p == ',')
            ++p;
    }
    return n_evals_out > 0;
}

// lance UCI --> 16 bits (promo<<12 | to<<6 | from), espelhado se as pretas jogam; promo 1=n, 2=b, 3=r, 4=q
static inline uint16_t moveToU16(const char *s, bool flip)
{
    if (!s || !s[0] || !s[1] || !s[2] || !s[3])
        return 0;
    uint16_t from = (s[1] - '1') * 8 + (s[0] - 'a');
    uint16_t to = (s[3] - '1') * 8 + (s[2] - 'a');
    if (flip)
    {
        from ^= 56;
        to ^= 56;
    }
    uint16_t promo = (s[4] == 'n') ? 1 : (s[4] == 'b') ? 2
                                     : (s[4] == 'r')   ? 3
                                     : (s[4] == 'q')   ? 4
                                                       : 0;
    return (promo << 12) | (to << 6) | from;
}

// cp canonizado de uma linha da avaliação (mate = +-CP_CLAMP)
static inline int pvCp(const RawPV &pv, bool flip)
{
    if (pv.cp != INT_MIN)
    {
        int v = flip ? -pv.cp : pv.cp;
        return std::max(-CP_CLAMP, std::min(CP_CLAMP, v));
    }
    if (pv.mate_v != INT_MIN)
    {
        int m = flip ? -pv.mate_v : pv.mate_v;
        return m > 0 ? CP_CLAMP : -CP_CLAMP;
    }
    return 0;
}

// FEN --> ChessData canonizado (tabuleiro, roques, en passant e zobrist); retorna true se as pretas jogam
static bool parseFenToStruct(const char *fen, ChessData &out)
{
    const char *p = fen;
    std::memset(&out, 0, sizeof(out));
    out.en_passant = 255;

    // peças, 4 bits por casa
    int rank = 7, file = 0;
    while (*p && *p != ' ')
    {
        char c = *p++;
        if (c == '/')
        {
            rank--;
            file = 0;
        }
        else if (c >= '1' && c <= '8')
        {
            file += (c - '0');
        }
        else
        {
            int sq = rank * 8 + file++;
            uint8_t piece = CHAR_TO_NIBBLE[(uint8_t)c];
            if (sq & 1)
                out.data[sq >> 1] |= piece << 4;
            else
                out.data[sq >> 1] |= piece;
        }
    }
    if (*p)
        ++p;

    bool flip = (*p == 'b');
    if (flip)
        out.flags |= FLAG_TURN;
    p += 2;

    // roques; se as pretas jogam, os roques das pretas viram os de quem joga
    bool rK = false, rQ = false, rk = false, rq = false;
    while (*p && *p != ' ')
    {
        char c = *p++;
        if (c == 'K')
            rK = true;
        else if (c == 'Q')
            rQ = true;
        else if (c == 'k')
            rk = true;
        else if (c == 'q')
            rq = true;
    }
    if (*p)
        ++p;
    if (flip)
    {
        if (rk)
            out.flags |= FLAG_CASTLE_K;
        if (rq)
            out.flags |= FLAG_CASTLE_Q;
        if (rK)
            out.flags |= FLAG_CASTLE_k;
        if (rQ)
            out.flags |= FLAG_CASTLE_q;
    }
    else
    {
        if (rK)
            out.flags |= FLAG_CASTLE_K;
        if (rQ)
            out.flags |= FLAG_CASTLE_Q;
        if (rk)
            out.flags |= FLAG_CASTLE_k;
        if (rq)
            out.flags |= FLAG_CASTLE_q;
    }

    if (*p && *p != '-')
    {
        int ep = (p[1] - '1') * 8 + (p[0] - 'a');
        out.en_passant = flip ? (uint8_t)(ep ^ 56) : (uint8_t)ep;
    }
    while (*p && *p != ' ')
        ++p;
    if (*p)
        ++p;

    // espelha o tabuleiro e troca as cores quando as pretas jogam
    if (flip)
    {
        uint8_t tmp[32];
        std::memcpy(tmp, out.data, 32);
        std::memset(out.data, 0, 32);
        for (int sq = 0; sq < 64; ++sq)
        {
            uint8_t b = tmp[sq >> 1];
            uint8_t piece = (sq & 1) ? (b >> 4) : (b & 0x0F);
            if (!piece)
                continue;
            int t = sq ^ 56;
            uint8_t fp = FLIP_PIECE[piece];
            if (t & 1)
                out.data[t >> 1] |= fp << 4;
            else
                out.data[t >> 1] |= fp;
        }
    }

    // zobrist da posição canonizada (peças, roques e en passant)
    out.zobrist = 0;
    for (int sq = 0; sq < 64; ++sq)
    {
        uint8_t b = out.data[sq >> 1];
        uint8_t piece = (sq & 1) ? (b >> 4) : (b & 0x0F);
        if (piece)
            out.zobrist ^= chess::Zobrist::RANDOM_ARRAY[NIBBLE_ZOFFSET[piece] + sq];
    }
    if (out.flags & FLAG_CASTLE_K)
        out.zobrist ^= chess::Zobrist::RANDOM_ARRAY[768];
    if (out.flags & FLAG_CASTLE_Q)
        out.zobrist ^= chess::Zobrist::RANDOM_ARRAY[769];
    if (out.flags & FLAG_CASTLE_k)
        out.zobrist ^= chess::Zobrist::RANDOM_ARRAY[770];
    if (out.flags & FLAG_CASTLE_q)
        out.zobrist ^= chess::Zobrist::RANDOM_ARRAY[771];
    if (out.en_passant != 255)
        out.zobrist ^= chess::Zobrist::RANDOM_ARRAY[772 + (out.en_passant % 8)];
    // RANDOM_ARRAY[780] (vez de jogar) não entra, pois a posição está sempre canonizada para as brancas

    return flip;
}

// lê o JSONL de avaliações do Lichess pela entrada padrão e grava dataset.bin, um registro por posição
int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    initCharToNibble();

    FILE *fout = fopen("dataset.bin", "wb");
    if (!fout)
    {
        std::cerr << "Erro: nao foi possivel criar dataset.bin\n";
        return 1;
    }

    // grava em blocos de 65536 registros
    static ChessData buffer[65536];
    size_t bufCount = 0;
    uint64_t processed = 0, skipped = 0;

    static char fen_buf[128];
    static EvalEntry evals_buf[MAX_EVALS];
    int n_evals = 0;

    static char line[131072]; // buffer de 128 KB por linha

    while (fgets(line, sizeof(line), stdin))
    {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;

        if (!fastParse(line, fen_buf, evals_buf, n_evals))
        {
            ++skipped;
            continue;
        }

        // usa a primeira avaliação do array, assumindo que é a de maior profundidade
        const EvalEntry &deep = evals_buf[0];

        ChessData cd;
        bool flip = parseFenToStruct(fen_buf, cd);

        // alvos --> primeiro lance e avaliação da melhor linha
        cd.move = moveToU16(deep.pvs[0].move, flip);
        cd.cp = (int16_t)pvCp(deep.pvs[0], flip);

        buffer[bufCount++] = cd;
        if (bufCount == 65536)
        {
            fwrite(buffer, sizeof(ChessData), bufCount, fout);
            bufCount = 0;
        }

        if (++processed % 100000 == 0)
            std::cerr << "\rProcessed " << processed << " | Skipped " << skipped << std::flush;
    }

    if (bufCount)
        fwrite(buffer, sizeof(ChessData), bufCount, fout);
    fclose(fout);
    std::cerr << "\nDone. Total: " << processed << " | Skipped: " << skipped << "\n";
    return 0;
}
