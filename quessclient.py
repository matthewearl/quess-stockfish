import asyncio
import enum
import logging

import numpy as np

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
            coords = np.round((np.array(ent.origin[:2]) + 224) / 64).astype(int)
            side = Side.WHITE if ent.skin == 1 else Side.BLACK
            board_state[tuple(coords)] = (side, piece)

    return board_state


def _print_board_state(board_state):
    chars = [['.' for _ in range(8)] for _ in range(8)]

    for (x, y), (side, piece) in board_state.items():
        char = _piece_to_notation[piece]
        if side == Side.WHITE:
            char = char.lower()
        chars[y][x] = char

    print('\n'.join(' '.join(c for c in row) for row in chars))


def _square_location(x, y):
    return (np.array([x, y, -16]) - [3.5, 3.5, 0]) * [64, 64, 1]


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
        start_time = client.time

        while True:
            while not client.center_print_queue.empty():
                print('center print!', client.center_print_queue.get_nowait())
            view_origin = np.array(client.player_entity.origin)
            view_origin[2] += client.view_height
            target = _square_location(int(client.time) & 7,
                                      (int(client.time) >> 3) & 0x7)
            dir_ = target - view_origin
            yaw = np.arctan2(dir_[1], dir_[0])
            pitch = np.arctan2(-dir_[2], np.linalg.norm(dir_[:2]))

            client.move(pitch, yaw, 0, 0, 0, 0, 0, 0)

            await client.wait_for_movement(client.view_entity)

            print('-----')
            _print_board_state(_get_board_state(client))

    finally:
        await client.disconnect()
        demo.stop_recording()
        with open('pyquess.dem', 'wb') as f:
            demo.dump(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(do_client())
