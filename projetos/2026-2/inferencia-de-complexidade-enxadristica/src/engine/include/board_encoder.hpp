#pragma once

#include "chess.hpp"

namespace encoding
{
    // codifica o tabuleiro da biblioteca direto no formato 19x64 da rede, sem passar por FEN
    bool encodeBoard(const chess::Board &board, float *output, bool &flip);

} // namespace encoding
