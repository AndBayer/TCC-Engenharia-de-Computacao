#pragma once

#include <cstdint>

namespace encoding
{

    // 34 bytes por posição --> data com 4 bits por casa, flags com os 4 roques e enPassant (255 = sem en passant)
    struct CompactPosition
    {
        uint8_t data[32]{};
        uint8_t flags = 0;
        uint8_t enPassant = 255;
    };

    static_assert(sizeof(CompactPosition) == 34, "CompactPosition must contain exactly the inference features");

} // namespace encoding
