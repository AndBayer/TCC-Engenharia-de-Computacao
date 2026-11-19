#include "board_encoder.hpp"

#include "encoding.hpp"

#include <cstring>

namespace encoding
{

    // tabuleiro --> tensor 19x64; se as pretas jogam, espelha o tabuleiro
    bool encodeBoard(const chess::Board &board, float *output, bool &flip)
    {
        flip = board.sideToMove() == chess::Color::BLACK;
        std::memset(output, 0, INPUT_SIZE * sizeof(float));

        // canais 0-5 --> peças de quem joga; 6-11 --> peças do adversário
        for (int source = 0; source < BOARD_SQ; ++source)
        {
            const chess::Piece piece = board.at(chess::Square(source));
            if (piece == chess::Piece::NONE)
                continue;

            int channel = static_cast<int>(piece);
            int target = source;
            if (flip)
            {
                channel = channel < 6 ? channel + 6 : channel - 6;
                target ^= 56;
            }
            output[channel * BOARD_SQ + target] = 1.0f;
        }

        // canal 12 --> casa de en passant
        const chess::Square enPassant = board.enpassantSq();
        if (enPassant != chess::Square::NO_SQ)
        {
            const int target = enPassant.index() ^ (flip ? 56 : 0);
            output[12 * BOARD_SQ + target] = 1.0f;
        }

        // canais 13-16 --> roques (pequeno e grande de quem joga, pequeno e grande do adversário), com o plano inteiro em 1
        const auto rights = board.castlingRights();
        const chess::Color us = board.sideToMove();
        const chess::Color opponent = ~us;
        using Side = chess::Board::CastlingRights::Side;
        const bool planes[4] = {
            rights.has(us, Side::KING_SIDE),
            rights.has(us, Side::QUEEN_SIDE),
            rights.has(opponent, Side::KING_SIDE),
            rights.has(opponent, Side::QUEEN_SIDE),
        };
        for (int plane = 0; plane < 4; ++plane)
        {
            if (!planes[plane])
                continue;
            for (int square = 0; square < BOARD_SQ; ++square)
                output[(13 + plane) * BOARD_SQ + square] = 1.0f;
        }

        // canais 17 e 18 com quantas peças de quem joga (17) e do adversário (18) atacam cada casa
        for (int source = 0; source < BOARD_SQ; ++source)
        {
            const int target = source ^ (flip ? 56 : 0);
            const chess::Square square(source);
            output[17 * BOARD_SQ + target] = static_cast<float>(chess::attacks::attackers(board, us, square).count());
            output[18 * BOARD_SQ + target] = static_cast<float>(chess::attacks::attackers(board, opponent, square).count());
        }

        return true;
    }

} // namespace encoding
