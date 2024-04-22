import asyncio
import concurrent.futures
import logging

import chess
import numpy as np
import stockfish

import pyquake.client
import pyquake.proto


class _Impulse:
    UNSELECT = 20
    SELECT = 100
    PASS = 60


_look_forward_yaw = {
    chess.WHITE:  np.pi / 2,
    chess.BLACK:  -np.pi / 2,
}


_player_origins = {
    chess.WHITE: (1.0, -411.0, 143.0),
    chess.BLACK: (4.0, 422.0, 137.0),
}


_idle_frames = {
    chess.PAWN: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    chess.ROOK: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    chess.KNIGHT: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    chess.BISHOP: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    chess.QUEEN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    chess.KING: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
}


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
        loop = asyncio.get_event_loop()
        move = await loop.run_in_executor(self._executor, self._get_best_move,
                                          board)
        return move

    def _get_best_move(self, board):
        self._stockfish.set_fen_position(board.fen)
        return chess.Move.from_uci(self._stockfish.get_best_move())


def _get_board(client) -> chess.BaseBoard:
    """Get a base board from current entity positions."""
    board = chess.BaseBoard.empty()

    for k, ent in client.entities.items():
        model = client.models[ent.model_num - 1]
        if model in _model_to_piece_type:
            piece_type = _model_to_piece_type[model]
            coords = np.round((np.array(ent.origin[:2]) + 224) / 64).astype(int)
            side = chess.WHITE if ent.skin != 1 else chess.BLACK
            if (np.all(coords >= 0) and np.all(coords < 8)
                    and ent.frame in _idle_frames[piece_type]):
                board.set_piece_at(
                    coords[0] + coords[1] * 8,
                    chess.Piece(piece_type, side)
                )
    return board


def _get_move_from_diff(board_before: chess.BaseBoard,
                        board_after: chess.baseBoard) -> chess.Move:
    """Find the move that transitions between two given boards."""

    map_before = board_before.piece_map()
    map_after = board_before.piece_map()

    squares_before = map_before.keys() - map_after.keys()
    squares_after = map_after.keys() - map_before.keys()

    if len(squares_before) == 1 and len(squares_after) == 1:
        square_before, = squares_before
        square_after, = squares_after

        if map_before[square_before] == map_after[square_after]:
            # This is a normal move
            move = chess.Move(square_before, square_after)
        elif map_before[square_before] == chess.PAWN:
            # This is a pawn promotion
            move = chess.Move(square_before, square_after,
                              map_after[square_after].piece_type)
        else:
            raise Exception(f'invalid move {move}')
    elif len(squares_before) == 2 and len(squares_after) == 2:
        # This is a castling move.
        square_before, = (square
                          for square in squares_before
                          if map_before[square].piece_type == chess.KING)
        square_after, = (square
                         for square in squares_after
                         if coords_to[square].piece_type == chess.KING)
        move = chess.Move(square_before, square_after)
    else:
        raise Exception(f'invalid move {move}')

    return move


def _move_to_coords(move: chess.Move, board_before: chess.Board):
    """Turn a UCI move into a pair of coords, and an optional promotion choice.

    The coordinates represents squares where the bot needs to click.  The
    promotion choice is generated for pawn promotion moves.
    """

    coords_from = (move.from_square % 8), (move.from_square // 8)
    coords_to = (move.to_square % 8), (move.to_square // 8)

    king_square = board_before.king()

    if king_square == move.from_square and coords_to[0] > coords_from[0] + 1:
        # Kingside castling.
        assert coords_from[1] == coords_to[1]
        assert coords_to[1] in (0, 7)
        assert coords_to[0] == 6
        assert coords_from[0] == 4
        out = coords_from, (7, coords_to[1]), None
    elif king_square == move.from_square and coords_to[0] < coords_from[0] - 1:
        # Queenside castling.
        assert coords_from[1] == coords_to[1]
        assert coords_to[1] in (0, 7)
        assert coords_to[0] == 2
        assert coords_from[0] == 4
        out = coords_from, (0, coords_to[1]), None
    else:
        # Normal move or pawn promotion.
        out = coords_from, coords_to, move.promotion

    return out


def _coords_to_angles(x, y, client):
    view_origin = np.array(client.player_entity.origin)
    target = (np.array([x, y, -15]) - [3.5, 3.5, 0]) * [64, 64, 1]
    dir_ = target - view_origin
    yaw = np.arctan2(dir_[1], dir_[0])
    pitch = np.arctan2(-dir_[2], np.linalg.norm(dir_[:2]))

    return pitch, yaw


def _coords_to_highlight_model_num(x, y):
    return 1 + (8 - x) + y * 8


async def _wait_until_turn(client, side: chess.Color):
    # Wait until we have any pieces at all.
    board = None
    while board is None or board.king(side) is None:
        board = _get_board(client)
        await client.wait_for_update()

    # Look at the square below the king, until it is highlighted.
    king_square = board.king(side)
    king_coords = (king_square % 8), (king_square // 8)
    hl_model_num = _coords_to_highlight_model_num(*king_coords)
    while all(ent.origin[2] == 0
              for ent in client.entities.values()
              if ent.model_num == hl_model_num):
        pitch, yaw = _coords_to_angles(*king_coords, client)
        client.move(pitch, yaw, 0, 0, 0, 0, 0, 20)
        await client.wait_for_update()

    # Look away until the square is not highlighted.
    while any(ent.origin[2] != 0
              for ent in client.entities.values()
              if ent.model_num == hl_model_num):
        client.move(0, _look_forward_yaw[side], 0, 0, 0, 0, 0, 20)
        await client.wait_for_update()


async def _find_side(client):
    # Work out which side we are.
    while client.view_entity not in client.entities:
        await client.wait_for_update()
    player_origin = client.player_origin
    for side, origin in _player_origins.items():
        if origin == player_origin:
            out = side
            break
    else:
        raise Exception(f'Invalid player origin {player_origin}')
    return out


async def _wait(client, duration):
    start_time = client.time
    while client.time < start_time + duration:
        await client.wait_for_update()


async def _play_game(client):
    sf = _AsyncStockfish()
    side = _find_side(client)
    if side != chess.WHITE:
        raise Exception("Only bot as white is supported")
    other_side = not side

    await _wait_until_turn(client, side)

    if _get_board(client) != chess.BaseBoard():
        raise Exception("White must move first")

    board = chess.Board()

    # Play until either the other player checkmates us, or we take their king.
    done = False
    while not done:
        # Get the move we should play, according to Stockfish.
        move = await sf.get_best_move(board)

        # Send commands to apply this move.
        from_coords, to_coords, promotion = _move_to_coords(move, board)
        pitch, yaw = _coords_to_angles(*from_coords, client)
        client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.UNSELECT)
        async _wait(client, 0.1)

        client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.SELECT)
        async _wait(client, 0.1)

        pitch, yaw = _coords_to_angles(*to_coords, client)
        client.move(pitch, yaw, 0, 0, 0, 0, 0, 0)
        async _wait(client, 0.1)

        client.move(pitch, yaw, 0, 0, 0, 0, 0, _Impulse.SELECT)
        async _wait(client, 0.1)

        if promotion is not None:
            raise Exception("promotion not yet supported")

        # Update our board state.
        board.push(move)

        if board.king(other_side) is None:
            # We just won.
            done = True

        if not done:
            # Wait for other player to make their turn
            await _wait_until_turn(client, side)

            was_checkmate = board.is_checkmate()
            board_after = _get_board(client)
            move = _get_move_from_diff(board, board_after)
            board.push(move)
            assert board == board_after

            if not was_checkmate and board.is_checkmate():
                # We just lost.
                done = True


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
