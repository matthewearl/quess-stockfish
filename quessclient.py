import asyncio
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


def _get_board_state(client):
    board_state = {}
    for k, ent in client.entities.items():
        model = client.models[ent.model_num - 1]
        if model in _model_to_piece:
            piece = _model_to_piece[model]
            unrounded_coords = (np.array(ent.origin[:2]) + 224) / 64
            coords = np.round((np.array(ent.origin[:2]) + 224) / 64).astype(int)
            if np.any(np.abs(unrounded_coords - coords) > 0.2):
                return None
            side = Side.WHITE if ent.skin == 1 else Side.BLACK
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
        for y in range(8)
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
    sf = stockfish.Stockfish()
    try:
        demo = client.record_demo()
        await client.wait_until_spawn()
        start_time = client.time

        last_board_state = None
        from_coords = None
        to_coords = None
        while True:
            if from_coords is None:
                yaw = 0
                pitch = 0
                impulse = 0
            else:
                view_origin = np.array(client.player_entity.origin)
                view_origin[2] += client.view_height
                phase = int(client.time * 10) % 6
                if phase < 3:
                    coords = from_coords
                else:
                    coords = to_coords
                target = _square_location(*coords)
                dir_ = target - view_origin
                yaw = np.arctan2(dir_[1], dir_[0])
                pitch = np.arctan2(-dir_[2], np.linalg.norm(dir_[:2]))
                impulse = 100 if phase % 3 == 0 else 0

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

                    sf.set_fen_position(fen_state)
                    encoded_move = sf.get_best_move()
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
