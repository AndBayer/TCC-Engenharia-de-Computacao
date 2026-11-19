
// formato da entrada da rede (tensor 19x64) e lance UCI --> índice da política
//   - 19 canais x 64 casas, ordem (canal, casa), casa = rank*8+file (a1=0)
//   - se as pretas jogam, espelha o tabuleiro (sq ^ 56) e troca as cores (flip=true), assim a rede sempre trata quem joga como brancas
//   - canais 0-11 peças (P,N,B,R,Q,K,p,n,b,r,q,k), 12 en passant, 13-16 roques (pequeno e grande de quem joga, pequeno e grande do adversário), 17-18 quantas peças de quem joga e do adversário atacam a casa
//   - sem canal de turno, pois o flip já define quem joga
//   - política com 4224 = 4096 (to*64+from) + 96 promoções + 32 sem uso
//   - roque no dataset (Lichess) é rei captura torre (e1h1/e1a1, não e1g1/e1c1)

#pragma once

#include <string>

namespace encoding
{

    constexpr int CHANNELS = 19;
    constexpr int BOARD_SQ = 64;
    constexpr int POLICY_SIZE = 4224;
    constexpr int INPUT_SIZE = CHANNELS * BOARD_SQ;

    // lance UCI --> índice da política (0-4223), com flip; -1 se o lance não tem 4 ou 5 caracteres
    // 0-4095 = destino * 64 + origem
    // 4096-4191 = promoções (8 colunas x 3 direções x 4 peças); 4192-4223 sem uso
    inline int uciToIndex(const std::string &uci, bool flip)
    {
        if (uci.length() < 4)
            return -1;
        int from = (uci[1] - '1') * 8 + (uci[0] - 'a');
        int to = (uci[3] - '1') * 8 + (uci[2] - 'a');
        if (flip)
        {
            from ^= 56;
            to ^= 56;
        }

        if (uci.length() == 4)
            return to * 64 + from;
        else if (uci.length() == 5)
        {
            int p = 0; // 'n'
            if (uci[4] == 'b')
                p = 1;
            else if (uci[4] == 'r')
                p = 2;
            else if (uci[4] == 'q')
                p = 3;
            // coluna de origem (0-7), direção (0 captura à esquerda, 1 avanço, 2 captura à direita) e peça
            int ff = from - 48;
            int dir = to - 56 - ff + 1;
            int rem = dir * 4 + p;
            return 4096 + (ff * 12 + rem);
        }
        return -1;
    }

} // namespace encoding
