import asyncio
import concurrent.futures
import enum
import logging

import numpy as np
import stockfish

import pyquake.client
import pyquake.proto


class PieceType(enum.Enum):
    PAWN = enum.auto()
    ROOK = enum.auto()
    KNIGHT = enum.auto()
    BISHOP = enum.auto()
    QUEEN = enum.auto()
    KING = enum.auto()


class Side(enum.Enum):
    WHITE = enum.auto()
    BLACK = enum.auto()


_idle_frames = {
	PieceType.PAWN: [0, 1, 2, 3, 4, 5, 6, 7, 8],
	PieceType.ROOK: [0, 1, 2, 3, 4, 5, 6, 7, 8],
	PieceType.KNIGHT: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
	PieceType.BISHOP: [0, 1, 2, 3, 4, 5, 6, 7, 8],
	PieceType.QUEEN: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
	PieceType.KING: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
}


_piece_to_notation = {
    PieceType.PAWN: 'P',
    PieceType.ROOK: 'R',
    PieceType.KNIGHT: 'N',
    PieceType.BISHOP: 'B',
    PieceType.QUEEN: 'Q',
    PieceType.KING: 'K',
}


_model_to_piece = {
    "progs/knight.mdl": PieceType.PAWN,
    "progs/ogre.mdl": PieceType.ROOK,
    "progs/demon.mdl": PieceType.KNIGHT,
    "progs/hknight.mdl": PieceType.BISHOP,
    "progs/shambler.mdl": PieceType.QUEEN,
    "progs/zombie.mdl": PieceType.KING,
}


_model_re = r"progs/([a-z]*).mdl"


class _AsyncStockfish:
    def __init__(self):
        self._stockfish = stockfish.Stockfish()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def get_best_move(self, fen):
        loop = asyncio.get_event_loop()
        move = await loop.run_in_executor(self._executor, self._get_best_move,
                                          fen)
        return move

    def _get_best_move(self, fen):
        if not self._stockfish.is_fen_valid(fen):
            return None
        self._stockfish.set_fen_position(fen)
        return self._stockfish.get_best_move()


def _get_board_state(client):
    board_state = {}
    for k, ent in client.entities.items():
        model = client.models[ent.model_num - 1]
        if model in _model_to_piece:
            piece = _model_to_piece[model]
            coords = np.round((np.array(ent.origin[:2]) + 224) / 64).astype(int)
            side = Side.WHITE if ent.skin != 1 else Side.BLACK
            if (np.all(coords >= 0) and np.all(coords < 8)
                    and ent.frame in _idle_frames[piece]):
                board_state[ent.entity_num] = (tuple(coords), side, piece)

    return board_state


def _board_state_to_fen(board_state):
    chars = [['.' for _ in range(8)] for _ in range(8)]

    for (x, y), side, piece in board_state.values():
        char = _piece_to_notation[piece]
        if side == Side.BLACK:
            char = char.lower()
        chars[y][x] = char

    field1 = '/'.join(
        ''.join(
            chars[y][x]
            for x in range(8)
        )
        for y in reversed(range(8))
    )
    for n in reversed(range(1, 9)):
        field1 = field1.replace('.' * n, str(n))

    return f'{field1} w - - 0 1'


def _diff_board_state(bs1, bs2):
    moves = []
    for k in bs1.keys() & bs2.keys():
        (coords1, side1, piece1) = bs1[k]
        (coords2, side2, piece2) = bs2[k]
        assert side1 == side2
        assert piece1 == piece2
        if coords1 != coords2:
            moves.append((coords1, coords2))
    return moves


def _print_board_state(board_state):
    chars = [['.' for _ in range(8)] for _ in range(8)]

    for (x, y), side, piece in board_state.values():
        char = _piece_to_notation[piece]
        if side == Side.BLACK:
            char = char.lower()
        chars[y][x] = char

    print('\n'.join(' '.join(c for c in row) for row in reversed(chars)))


def _square_location(x, y):
    return (np.array([x, y, -16]) - [3.5, 3.5, 0]) * [64, 64, 1]


def _decode_move(move):
    if move is None:
        return None, None
    return ((ord(move[0]) - ord('a'), ord(move[1]) - ord('1')),
            (ord(move[2]) - ord('a'), ord(move[3]) - ord('1')))


async def do_client():
    client = await pyquake.client.AsyncClient.connect(
        "localhost", 26000,
        pyquake.proto.Protocol(
            pyquake.proto.ProtocolVersion.NETQUAKE,
            pyquake.proto.ProtocolFlags(0)
        )
    )
    sf = _AsyncStockfish()
    try:
        demo = client.record_demo()
        await client.wait_until_spawn()
        start_time = client.time

        last_board_state = None
        from_coords = None
        to_coords = None
        prev_click = False
        click = True
        while True:
            if from_coords is None:
                yaw = 0
                pitch = 0
                impulse = 0
            else:
                view_origin = np.array(client.player_entity.origin)
                view_origin[2] += client.view_height
                phase = int(client.time * 10) % 9
                if phase < 6:
                    coords = from_coords
                else:
                    coords = to_coords
                target = _square_location(*coords)
                dir_ = target - view_origin
                yaw = np.arctan2(dir_[1], dir_[0])
                pitch = np.arctan2(-dir_[2], np.linalg.norm(dir_[:2]))
                prev_click = click
                click = (phase % 3 == 1)
                if click and not prev_click:
                    if phase < 3:
                        impulse = 20
                    else:
                        impulse = 100
                else:
                    impulse = 0

            client.move(pitch, yaw, 0, 0, 0, 0, 0, impulse)

            await client.wait_for_movement(client.view_entity)

            board_state = _get_board_state(client)
            if board_state is not None:
                if last_board_state is not None:
                    moves = _diff_board_state(last_board_state, board_state)

                if last_board_state is None or moves:
                    print()
                    _print_board_state(board_state)
                    fen_state = _board_state_to_fen(board_state)
                    print(fen_state)

                    encoded_move = await sf.get_best_move(fen_state)
                    from_coords, to_coords = _decode_move(encoded_move)
                    print(f'playing {encoded_move} ({from_coords}-{to_coords})')

                    last_board_state = board_state

    finally:
        await client.disconnect()
        demo.stop_recording()
        with open('pyquess.dem', 'wb') as f:
            demo.dump(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(do_client())
