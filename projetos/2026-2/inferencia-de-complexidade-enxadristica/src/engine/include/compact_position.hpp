#pragma once

#include "chess.hpp"
#include "compact_position_data.hpp"

namespace encoding
{

    // tabuleiro --> posição compacta de 34 bytes, já espelhada quando as pretas jogam
    CompactPosition compactPosition(const chess::Board &board);

} // namespace encoding
