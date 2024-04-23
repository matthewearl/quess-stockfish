import asyncio
import concurrent.futures
import logging

import chess
import numpy as np
import stockfish

import pyquake.client
import pyquake.proto


logger = logging.getLogger(__name__)


class _Impulse:
    UNSELECT = 20
    SELECT = 100
    PASS = 60


# Initial player origins for each side.
_player_origins = {
    chess.WHITE: (1.0, -411.0, 143.0),
    chess.BLACK: (4.0, 422.0, 137.0),
}


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


class _AsyncStockfish:
    def __init__(self):
        self._stockfish = stockfish.Stockfish()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def get_best_move(self, board: chess.Board) -> chess.Move:
        logger.info('thinking...')
        loop = asyncio.get_event_loop()
        move = await loop.run_in_executor(self._executor, self._get_best_move,
                                          board)
        return move

    def _get_best_move(self, board):
        self._stockfish.set_fen_position(board.fen())
        return chess.Move.from_uci(self._stockfish.get_best_move())


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


def _get_move_from_diff(board_before: chess.BaseBoard,
                        board_after: chess.BaseBoard,
                        color: chess.Color) -> chess.Move:
    """Find the move that transitions between two given boards."""

    map_before = {square: piece
                  for square, piece in board_before.piece_map().items()
                  if piece.color == color}
    map_after = {square: piece
                 for square, piece in board_after.piece_map().items()
                 if piece.color == color}

    squares_before = map_before.keys() - map_after.keys()
    squares_after = map_after.keys() - map_before.keys()

    if len(squares_before) == 1 and len(squares_after) == 1:
        square_before, = squares_before
        square_after, = squares_after

        if map_before[square_before] == map_after[square_after]:
            # This is a normal move
            move = chess.Move(square_before, square_after)
        elif map_before[square_before].piece_type == chess.PAWN:
            # This is a pawn promotion
            move = chess.Move(square_before, square_after,
                              map_after[square_after].piece_type)
        else:
            raise Exception('invalid move')
    elif len(squares_before) == 2 and len(squares_after) == 2:
        # This is a castling move.
        square_before, = (square
                          for square in squares_before
                          if map_before[square].piece_type == chess.KING)
        square_after, = (square
                         for square in squares_after
                         if map_after[square].piece_type == chess.KING)
        move = chess.Move(square_before, square_after)
    else:
        raise Exception('invalid move')

    return move


def _square_to_angles(square, client):
    x = square % 8
    y = square // 8
    view_origin = np.array(client.player_entity.origin)
    target = (np.array([x, y, -15]) - [3.5, 3.5, 0]) * [64, 64, 1]
    dir_ = target - view_origin
    yaw = np.arctan2(dir_[1], dir_[0])
    pitch = np.arctan2(-dir_[2], np.linalg.norm(dir_[:2]))

    return pitch, yaw


def _square_to_highlight_model_num(square):
    x = square % 8
    y = square // 8
    return 9 - x + y * 8


async def _wait_until_turn(client, color: chess.Color):
    # Wait until we have any pieces at all.
    board = None
    while board is None or board.king(color) is None:
        board = _get_board(client)
        await client.wait_for_update()

    # Look at the square below the king, until it is highlighted.
    king_square = board.king(color)
    while all(ent.origin[2] == 0
              for ent in client.entities.values()
              if ent.model_num == _square_to_highlight_model_num(king_square)):
        pitch, yaw = _square_to_angles(king_square, client)
        client.move(pitch, yaw, 0, 0, 0, 0, 0, 20)
        await client.wait_for_update()
        king_square = board.king(color)


async def _find_color(client):
    # Work out which color we are.
    while client.view_entity not in client.entities:
        await client.wait_for_update()
    player_origin = client.player_entity.origin
    for color, origin in _player_origins.items():
        if origin == player_origin:
            out = color
            break
    else:
        raise Exception(f'Invalid player origin {player_origin}')
    return out


async def _wait_for_promotion_anim(client):
    done = False
    while not done:
        await client.wait_for_update()
        for ent in client.entities.values():
            if _ent_to_piece_type(client, ent) == chess.PAWN:
                if ent.frame in _promotion_frames:
                    done = True
                    break


def _mirror_move(move: chess.Move):
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        move.promotion
    )


async def _play_game(client):
    sf = _AsyncStockfish()
    color = await _find_color(client)
    if color != chess.WHITE:
        raise Exception("Only bot as white is supported")
    other_color = not color

    # Wait until it is our turn.
    await _wait_until_turn(client, color)

    # Quess sometimes has black move first.  Handle this by mirroring the board
    # and moves passed into and received from stockfish.
    board = chess.Board()
    board_after = _get_board(client)
    black_first = board_after != board
    if black_first:
        board.turn = other_color
        move = _get_move_from_diff(board, board_after, other_color)
        logger.info('black moved first: %s', move)
        board.push(move)
        assert board == board_after

    # Play until either the other player checkmates us, or we take their king.
    while not board.is_checkmate():
        logger.info('bot to move:\n%s', board)

        # Get the move we should play, according to Stockfish.
        move = await sf.get_best_move(board.mirror() if black_first else board)
        if black_first:
            move = _mirror_move(move)
        logger.info('playing move %s', move)

        # Send commands to apply this move.
        pitch, yaw = _square_to_angles(move.from_square, client)
        client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.UNSELECT)
        await client.wait_for_update()

        client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.SELECT)
        await client.wait_for_update()

        pitch, yaw = _square_to_angles(move.to_square, client)
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

        if not board.is_checkmate():
            logger.info('other player to move:\n%s', board)
            # Wait for other player to make their turn
            await _wait_until_turn(client, color)

            board_after = _get_board(client)
            move = _get_move_from_diff(board, board_after, other_color)
            logger.info('other player moved: %s', move)
            board.push(move)
            assert board == board_after

    # Declare a winner.
    logger.info('%s wins:\n%s', _color_name(not board.turn), board)


async def do_client():
    client = await pyquake.client.AsyncClient.connect(
        "localhost", 26000,
        pyquake.proto.Protocol(
            pyquake.proto.ProtocolVersion.NETQUAKE,
            pyquake.proto.ProtocolFlags(0)
        )
    )

    try:
        demo = client.record_demo()
        await client.wait_until_spawn()
        await _play_game(client)
    finally:
        await client.disconnect()
        demo.stop_recording()
        with open('pyquess.dem', 'wb') as f:
            demo.dump(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(do_client())
