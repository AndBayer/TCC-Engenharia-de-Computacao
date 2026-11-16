#include "compact_position.hpp"

#include "encoding.hpp"

namespace encoding
{
    // posição compacta (34 bytes) que vai para a GPU
    namespace
    {
        // duas casas por byte --> casa par nos 4 bits baixos, casa ímpar nos 4 bits altos
        void setPiece(CompactPosition &position, int square, uint8_t piece)
        {
            uint8_t &packed = position.data[square >> 1];
            if ((square & 1) != 0)
                packed = static_cast<uint8_t>((packed & 0x0F) | (piece << 4));
            else
                packed = static_cast<uint8_t>((packed & 0xF0) | piece);
        }
    } // namespace

    CompactPosition compactPosition(const chess::Board &board)
    {
        CompactPosition compact;
        // se as pretas jogam, espelha o tabuleiro e troca as cores
        const bool flip = board.sideToMove() == chess::Color::BLACK;

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
            // casa vazia é 0, então a peça é gravada como canal + 1
            setPiece(compact, target, static_cast<uint8_t>(channel + 1));
        }

        // roques em bit0 (pequeno), bit1 (grande), bit2 e bit3 (do adversário)
        const auto rights = board.castlingRights();
        const chess::Color us = board.sideToMove();
        const chess::Color opponent = ~us;
        using Side = chess::Board::CastlingRights::Side;
        if (rights.has(us, Side::KING_SIDE))
            compact.flags |= 0x01;
        if (rights.has(us, Side::QUEEN_SIDE))
            compact.flags |= 0x02;
        if (rights.has(opponent, Side::KING_SIDE))
            compact.flags |= 0x04;
        if (rights.has(opponent, Side::QUEEN_SIDE))
            compact.flags |= 0x08;

        // casa de en passant, espelhada junto com o tabuleiro; sem en passant fica 255
        const chess::Square enPassant = board.enpassantSq();
        if (enPassant != chess::Square::NO_SQ)
            compact.enPassant = static_cast<uint8_t>(enPassant.index() ^ (flip ? 56 : 0));

        return compact;
    }


} // namespace encoding
