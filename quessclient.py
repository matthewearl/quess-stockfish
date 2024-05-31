import argparse
import asyncio
import concurrent.futures
import datetime
import logging
import os
import re

import chess
import chess.pgn
import numpy as np
import stockfish

import pyquake.aiodgram
import pyquake.bsp
import pyquake.client
import pyquake.pak
import pyquake.proto


logger = logging.getLogger(__name__)


class _Impulse:
    UNSELECT = 20
    SELECT = 100
    PASS = 60


# Frames for each piece when idle.
_idle_frames = {
    chess.PAWN: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    chess.ROOK: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    chess.KNIGHT: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    chess.BISHOP: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    chess.QUEEN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    chess.KING: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
}


# Frames for the pawn when waiting for a promotion.
_promotion_frames = [
    28, 29, 30
]


# Order of pieces in the promotion selection cycle.
_promotion_order = [
    chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP
]


# Models for each piece.
_model_to_piece_type = {
    "progs/knight.mdl": chess.PAWN,
    "progs/ogre.mdl": chess.ROOK,
    "progs/demon.mdl": chess.KNIGHT,
    "progs/hknight.mdl": chess.BISHOP,
    "progs/shambler.mdl": chess.QUEEN,
    "progs/zombie.mdl": chess.KING,
}


def _mirror_move(move: chess.Move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        move.promotion
    )


class _AsyncStockfish:
    def __init__(self, depth):
        if depth is None:
           depth = 15
        self._stockfish = stockfish.Stockfish(depth=depth)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def get_move(self, board: chess.Board,
                       black_first: bool) -> chess.Move | None:
        if black_first:
            board = board.mirror()
        logger.info('thinking...')
        loop = asyncio.get_event_loop()
        move = await loop.run_in_executor(self._executor, self._get_best_move,
                                          board)
        if black_first:
            move = _mirror_move(move)
        return move

    def _get_best_move(self, board):
        self._stockfish.set_fen_position(board.fen())
        return chess.Move.from_uci(self._stockfish.get_best_move())


def _parse_pgn(pgn):
    board = chess.Board()
    sans = ''.join(re.split('\d+\.', pgn)).split()
    black_first = sans and sans[0] == '..'
    if black_first:
        board = board.mirror()
        sans = sans[1:]

    if sans[-1] == '*':
        sans = sans[:-1]

    moves = []
    for san in sans:
        move = board.parse_san(san)
        board.push(move)
        moves.append(move)

    return black_first, moves


def _get_pieces_moved(board):
    return board != chess.Board() and board != chess.Board().mirror()


class _PgnPlayer:
    def __init__(self, pgn: str):
        self._black_first, self._moves = _parse_pgn(pgn)
        self._move_number = 0
        self._next_board = chess.Board()
        logger.info('black first: %s, moves: %s',
                    self._black_first, self._moves)

    async def get_move(self, board: chess.Board,
                       black_first: bool) -> chess.Move | None:
        pieces_moved = _get_pieces_moved(board)
        if self._move_number == 0 and pieces_moved:
            self._move_number += 1

        if pieces_moved or self._black_first == (board.turn == chess.BLACK):
            if self._move_number >= len(self._moves):
                raise Exception("reached end of pgn")
            move = self._moves[self._move_number]
            logger.info('played move %d', self._move_number)
            self._move_number += 2
        else:
            move = None
            self._move_number = 1
            logger.info('passing')

        return move


def _color_name(color: chess.Color):
    if color == chess.WHITE:
        return "white"
    else:
        return "black"


def _ent_to_piece_type(client, ent: pyquake.client.Entity) -> \
        chess.PieceType | None:
    model = client.models[ent.model_num - 1]
    if model in _model_to_piece_type:
        piece_type = _model_to_piece_type[model]
    else:
        piece_type = None
    return piece_type


def _get_board(client) -> chess.BaseBoard:
    """Get a base board from current entity positions."""
    board = chess.BaseBoard.empty()

    for ent in client.entities.values():
        piece_type = _ent_to_piece_type(client, ent)
        if piece_type is not None:
            coords = np.round((np.array(ent.origin[:2]) + 224) / 64).astype(int)
            color = chess.WHITE if ent.skin != 1 else chess.BLACK
            if (np.all(coords >= 0) and np.all(coords < 8)
                    and ent.frame in _idle_frames[piece_type]):
                board.set_piece_at(
                    coords[0] + coords[1] * 8,
                    chess.Piece(piece_type, color)
                )
    return board


def _get_move_from_diff(board_before: chess.Board,
                        board_after: chess.BaseBoard) -> chess.Move:
    """Find the move that transitions between two given boards."""

    def apply_move(b, m):
        b = b.copy()
        b.push(m)
        return b

    moves = [m
             for m in board_before.legal_moves
             if apply_move(board_before, m) == board_after]

    # There should be exactly one move that works
    if len(moves) == 0:
        raise Exception('invalid move')
    assert len(moves) == 1
    return moves[0]


def _square_to_angles(square, client, board_height):
    x = square % 8
    y = square // 8
    view_origin = np.array(client.player_entity.origin)
    target = (np.array([x, y, board_height]) - [3.5, 3.5, 0]) * [64, 64, 1]
    if y == 0 and view_origin[1] < 0:
        target[1] += 8
    elif y == 7 and view_origin[1] > 0:
        target[1] -= 8
    dir_ = target - view_origin
    yaw = np.arctan2(dir_[1], dir_[0])
    pitch = np.arctan2(-dir_[2], np.linalg.norm(dir_[:2]))

    return pitch, yaw


def _square_to_highlight_model_num(square):
    x = square % 8
    y = square // 8
    return 9 - x + y * 8


async def _wait_until_first_turn(client, color: chess.Color, board_height):
    board = _get_board(client)
    # Look at the square below the king, until it is highlighted.
    king_square = board.king(color)
    while all(ent.origin[2] == 0
              for ent in client.entities.values()
              if ent.model_num == _square_to_highlight_model_num(king_square)):
        pitch, yaw = _square_to_angles(king_square, client, board_height)
        client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.UNSELECT)
        await client.wait_for_update()
        king_square = board.king(color)


async def _wait_until_turn(client):
    msg = None
    while msg is None or msg not in ("Your turn", "You have lost.."):
        msg = (await client.wait_for_center_print()).strip()


async def _find_color(client):
    # Work out which color we are.
    while client.view_entity not in client.entities:
        await client.wait_for_update()
    player_origin = client.player_entity.origin

    if player_origin[1] < 0:
        color = chess.WHITE
    else:
        color = chess.BLACK
    return color


async def _wait_for_promotion_anim(client):
    done = False
    while not done:
        await client.wait_for_update()
        for ent in client.entities.values():
            if _ent_to_piece_type(client, ent) == chess.PAWN:
                if ent.frame in _promotion_frames:
                    done = True
                    break


def _log_pgn(board):
    game = chess.pgn.Game.from_board(board)
    pgn_str = str(game).strip().split('\n')[-1]
    logger.info('pgn: %s', pgn_str)


async def _play_bot_move(client, bot, board, black_first, board_height):
    """Play the bot's move.

    It must be the bot's turn when this function is called.
    """

    logger.info('%.3f bot (%s) to move:\n%s',
                client.time, _color_name(board.turn),
                board.unicode(empty_square='-', invert_color=True))
    _log_pgn(board)

    # Get the move we should play, according to the bot.
    move = await bot.get_move(board, black_first)

    logger.info('%.3f playing move %s', client.time, move)
    if move is None:
        pitch, yaw = _square_to_angles(36, client, board_height)
        client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.PASS)
        await client.wait_for_update()

        assert not _get_pieces_moved(board), "Can only pass on first turn"
        board.apply_mirror()
    else:
        # Send commands to apply this move.
        pitch, yaw = _square_to_angles(move.from_square, client, board_height)
        client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.UNSELECT)
        await client.wait_for_update()

        client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.SELECT)
        await client.wait_for_update()

        pitch, yaw = _square_to_angles(move.to_square, client, board_height)
        client.move(pitch, yaw, 0, 0, 0, 0, 0, 0)
        await client.wait_for_update()

        client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.SELECT)
        await client.wait_for_update()

        if move.promotion is not None:
            # Select correct piece from promotion menu.
            await _wait_for_promotion_anim(client)
            for _ in range(_promotion_order.index(move.promotion)):
                client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.UNSELECT)
                await client.wait_for_update()
            client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.SELECT)
            await client.wait_for_update()

        # Update our board state.
        board.push(move)


def _update_board_after_other_move(client, board, board_after):
    move = _get_move_from_diff(board, board_after)
    logger.info('%.3f other player moved: %s', client.time, move)
    board.push(move)
    assert board == board_after


async def _wait_for_other_move(client, board):
    logger.info('%.3f other player (%s) to move:\n%s',
                client.time, _color_name(board.turn),
                board.unicode(empty_square='-', invert_color=True))
    _log_pgn(board)
    await _wait_until_turn(client)
    _update_board_after_other_move(client, board, _get_board(client))


def _walk_planes(node):
    if not isinstance(node, pyquake.bsp.Leaf):
        yield node.plane
        for child_num in range(2):
            yield from _walk_planes(node.get_child(child_num))


def _trace_down(model, x, y, z_hi):
    """Intersect a downwards ray with collision hull."""

    candidate_zs = (
        plane.normal[2] * plane.dist
        for plane in _walk_planes(model.node)
        if abs(plane.normal[2]) == 1
    )
    return max(
        z
        for z in candidate_zs
        if z <= z_hi
        if model.get_leaf_from_point([x, y, z + 1e-5]).contents == -1
        if model.get_leaf_from_point([x, y, z - 1e-5]).contents != -1
    )


def _find_board_height(client, fs):
    """Find the height of the playing field."""

    # Load and parse the bsp file.
    with fs.open(client.models[0]) as f:
        bsp = pyquake.bsp.Bsp(f)

    # Find a point we know is above the board.
    for ent in client.entities.values():
        piece_type = _ent_to_piece_type(client, ent)
        if piece_type is not None:
            z_hi = ent.origin[2]
            break
    else:
        raise Exception("No pieces on board")

    # Trace down from that point.
    board_height = _trace_down(bsp.models[0], 0., 0., z_hi) + 1

    logger.info('board height %s', board_height)

    return board_height


async def _play_game(client, bot, fs):
    color = await _find_color(client)
    logger.info('playing as %s', _color_name(color))

    # Wait until we have any pieces at all.
    while _get_board(client).king(color) is None:
        await client.wait_for_update()

    # Find the floor height.
    board_height = _find_board_height(client, fs)

    # Wait until it is our turn.
    await _wait_until_first_turn(client, color, board_height)

    # Quess sometimes has black move first.  Handle this by mirroring the board
    # and moves passed into and received from stockfish.
    board = chess.Board()
    board_after = _get_board(client)
    black_first = (board_after != board) == (color == chess.WHITE)
    if black_first:
        logger.info('playing black first variant')
        board = board.mirror()  # Make initial board but with black to move.

    # If the other player moved first, update the board
    if board_after != board:
        _update_board_after_other_move(client, board, board_after)

    # Play until the game is over.
    while not board.is_game_over():
        await _play_bot_move(client, bot, board, black_first, board_height)
        if not board.is_game_over():
            await _wait_for_other_move(client, board)

    # Declare a winner.
    logger.info('outcome: %s\n%s', board.outcome(),
                board.unicode(empty_square='-', invert_color=True))
    _log_pgn(board)


async def do_client():
    parser = argparse.ArgumentParser(description="quess-stockfish")
    parser.add_argument("--depth", type=int, default=None,
                        help="Stockfish search depth")
    parser.add_argument("--pgn", type=str, default=None,
                        help="Replay game from a pgn string")
    parser.add_argument("--ip", type=str, default="localhost",
                        help="Server to connect to")
    parser.add_argument("--port", type=int, default=26000,
                        help="Port to connect to")
    parser.add_argument("--basedir", type=str, default='.',
                        help="Directory containing id1 directory")
    parser.add_argument("--game", type=str, default='quess134',
                        help="Name of the game directory inside the base "
                             "directory")
    parser.add_argument('--joequake', action='store_true',
                        help="Pass when connecting to a JQ server for "
                             "high-res angles.  Not needed for quakespasm and "
                             "derivatives.")
    args = parser.parse_args()

    fs = pyquake.pak.Filesystem(args.basedir, args.game)

    if args.joequake:
        # Try connecting with MOD_JOEQUAKE.  JoeQuake only handles 16-bit
        # inputs with MOD_JOEQUAKE, regardless of the negotiated protocol
        # (NETQUAKE, FITZQUAKE, etc).
        try:
            client = await pyquake.client.AsyncClient.connect(
                args.ip, args.port,
                joequake_version=35,  # 16-bit input precision
            )
        except pyquake.aiodgram.BadJoeQuakeVersion:
            raise Exception("Could not set MOD_JOEQUAKE.  Are you connecting "
                            "to a JoeQuake server?  If not do not pass "
                            "--joequake")
    else:
        # Connect without any protocol mod, which should work with other modern
        # Quake ports or original Quake.  We will use whatever protocol the
        # server is using --- anything but NETQUAKE should give us 16-bit
        # angles.
        client = await pyquake.client.AsyncClient.connect(args.ip, args.port)

    if args.pgn is None:
        bot = _AsyncStockfish(args.depth)
    else:
        bot = _PgnPlayer(args.pgn)

    try:
        demo = client.record_demo()
        await client.wait_until_spawn()
        if client.high_res_inputs:
            logger.info('using high res inputs')
        else:
            logger.warning('using low resolution inputs, bot may misclick')
        await _play_game(client, bot, fs)
    finally:
        await client.disconnect()
        demo.stop_recording()
        demo_fname = (
            'pyquess'
            f'-{datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}'
            f'-{os.getpid()}.dem'
        )
        with open(demo_fname, 'wb') as f:
            demo.dump(f)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(do_client())
