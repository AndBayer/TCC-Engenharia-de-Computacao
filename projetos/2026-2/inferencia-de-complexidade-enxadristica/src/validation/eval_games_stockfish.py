#!/usr/bin/env python3
"""Anota com Stockfish as partidas humanas do Lichess (<pasta dos PGNs>/*.pgn).

Uso:
    python eval_games_stockfish.py <pasta dos PGNs> <pasta de saída> [--depth 16] [--stockfish "stockfish.exe"]

Saída: <pasta de saída>/bracket_<faixa>.pgn, um arquivo por faixa de rating, no formato de anotação do Lichess. Cada lance recebe, no comentário, a avaliação da posição após o lance, na visão das brancas:

    1. e4 { [%eval 0.31,20] [%best e7e5] [%clk 0:10:00] }

  [%eval x,d] avaliação em peões (visão das brancas); mate = #n; d = profundidade
  [%best uci] melhor lance do Stockfish na posição após o lance anotado
              (ou seja, a alternativa do Stockfish para o próximo lance)
  [%clk ...]  relógio original do Lichess (preservado)

A posição inicial vai no comentário antes do primeiro lance (eval + best do primeiro lance).
A posição final de mate ou empate não recebe [%eval] (não há busca).

Todos os lances de todas as partidas são avaliados; filtros (janela de lances, |eval| etc.) ficam para o pós-processamento.

Retoma de onde parou, inclusive após queda de energia (partida cortada no meio da escrita é descartada e refeita).
"""
import argparse, os, glob, io
from concurrent.futures import ProcessPoolExecutor, as_completed
import chess, chess.pgn, chess.engine

STOCKFISH = "stockfish"    # no PATH; ou --stockfish <caminho>
PGN_DIR   = None
OUT_DIR   = None          # um PGN por faixa
DEPTH     = 16            # profundidade alvo por posição
TIME_CAP  = 60            # teto de segundos por posição (só contra travamento)
WORKERS   = 4             # processos Stockfish em paralelo
THREADS   = max(1, ((os.cpu_count() or 2) - 1) // WORKERS)  # threads por processo
HASH_MB   = 256           # por processo
# com várias threads o Stockfish não é reproduzível bit a bit (busca paralela não determinística)
# configuração para relatar --> "SF depth 16, WORKERS x THREADS threads"

_eng = None   # Stockfish do processo, aberto em _init

def _init(sf=STOCKFISH, depth=DEPTH):
    """Roda uma vez em cada processo do pool e abre o Stockfish desse processo."""
    global _eng, DEPTH
    DEPTH = depth                # processos filhos no Windows não herdam o valor do main
    _eng = chess.engine.SimpleEngine.popen_uci(sf)
    _eng.configure({"Threads": THREADS, "Hash": HASH_MB})

def _gid(headers):
    """Id da partida, o fim da URL do Lichess no header Site."""
    return headers.get("Site", "").rsplit("/", 1)[-1]

def _annotate(node, board, gid):
    """Grava [%eval] e [%best] da posição 'board' no comentário de 'node'."""
    if board.is_game_over():
        return
    info = _eng.analyse(board, chess.engine.Limit(depth=DEPTH, time=TIME_CAP),
                        game=gid)  # partida nova (gid) --> ucinewgame; a mesma partida mantém a tabela de transposição
    node.set_eval(info["score"], info.get("depth"))
    if info.get("pv"):
        node.comment = (node.comment + f" [%best {info['pv'][0].uci()}]").strip()

def eval_game(pgn_text):
    """Anota uma partida num processo do pool e devolve (id, PGN anotado)."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    gid = _gid(game.headers)
    game.headers["Annotator"] = f"Stockfish depth {DEPTH}"
    _annotate(game, game.board(), gid)                 # posição inicial
    for node in game.mainline():
        _annotate(node, node.board(), gid)             # posição após o lance
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    return gid, game.accept(exporter)

def load_games():
    """Lê as partidas de todos os PGNs da pasta de entrada; a faixa vem do nome do arquivo (bracket_<faixa>.pgn)."""
    games = {}   # faixa --> [(gid, pgn_text)]
    for path in sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn"))):
        bracket = os.path.splitext(os.path.basename(path))[0].replace("bracket_", "")
        with open(path, encoding="utf-8") as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                games.setdefault(bracket, []).append((_gid(g.headers), str(g)))
    return games

def _out_path(bracket):
    return os.path.join(OUT_DIR, f"bracket_{bracket}.pgn")

def _complete(game):
    """Partida inteira anotada --> todo lance (exceto mate ou empate final) tem [%eval]."""
    if not game.headers.get("Annotator") or game.errors:
        return False
    nodes = [game] + list(game.mainline())
    return all(n.eval() is not None or n.board().is_game_over() for n in nodes)

def _clean(path):
    """Mantém só as partidas completas no PGN de saída e devolve os ids delas."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    blocks = [b for b in text.split("\n\n\n") if b.strip()]   # um bloco por partida
    keep, done = [], set()
    for b in blocks:
        g = chess.pgn.read_game(io.StringIO(b))
        if g is not None and _complete(g) and _gid(g.headers) not in done:
            done.add(_gid(g.headers))
            keep.append(b.strip() + "\n")
    if len(keep) != len(blocks):
        print(f"  limpando {os.path.basename(path)}: "
              f"{len(blocks) - len(keep)} partidas incompletas descartadas")
    # grava num arquivo temporário e troca, para uma queda não deixar o PGN pela metade
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(k + "\n\n" for k in keep))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    return done

def main():
    global PGN_DIR, OUT_DIR, DEPTH
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--stockfish", default=STOCKFISH)
    args = ap.parse_args()
    PGN_DIR, OUT_DIR, DEPTH, sf = args.pgn_dir, args.out_dir, args.depth, args.stockfish
    games = load_games()
    os.makedirs(OUT_DIR, exist_ok=True)

    # só entram as partidas que ainda não estão completas na saída
    todo = []
    total = 0
    for bracket, lst in games.items():
        done = _clean(_out_path(bracket))
        total += len(lst)
        todo += [(bracket, gid, t) for gid, t in lst if gid not in done]
    print(f"{total} partidas ({total - len(todo)} ja feitas) | depth={DEPTH} "
          f"workers={WORKERS}x{THREADS}threads hash={HASH_MB}MB | todos os lances")

    files = {}   # faixa --> PGN de saída, aberto para anexar
    try:
        with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(sf, DEPTH)) as ex:
            futs = {ex.submit(eval_game, t): b for b, _, t in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                bracket = futs[fut]
                gid, pgn = fut.result()
                if bracket not in files:
                    files[bracket] = open(_out_path(bracket), "a", encoding="utf-8")
                fout = files[bracket]
                fout.write(pgn + "\n\n\n")    # partida inteira de uma vez
                fout.flush()
                os.fsync(fout.fileno())       # garante no disco (queda de energia)
                print(f"  {k}/{len(todo)}  {bracket}  {gid}", flush=True)
    finally:
        for fout in files.values():
            fout.close()
    print(f"pronto -> {OUT_DIR}")

if __name__ == "__main__":
    main()
