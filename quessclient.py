import asyncio
import logging
import math

import numpy as np

import pyquake.client
import pyquake.proto


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

        prev_origins = None
        while True:
            while not client.center_print_queue.empty():
                print('center print!', client.center_print_queue.get_nowait())
            view_origin = np.array(client.player_origin)
            view_origin[2] += client.view_height
            target = _square_location(int(client.time) & 7, (int(client.time) >> 3) & 0x7)
            dir_ = target - view_origin
            yaw = np.arctan2(dir_[1], dir_[0])
            pitch = np.arctan2(-dir_[2], np.linalg.norm(dir_[:2]))

            #print(dir_, yaw * 180 / np.pi, pitch * 180 / np.pi)
            client.move(pitch, yaw, 0, 0, 0, 0, 0, 0)

            await client.wait_for_movement(client.view_entity)

            if prev_origins is not None:
                for k, origin in client.origins.items():
                    if origin != prev_origins[k]:
                        print(client.time, k, origin)

            prev_origins = dict(client.origins)

    finally:
        await client.disconnect()
        demo.stop_recording()
        with open('pyquess.dem', 'wb') as f:
            demo.dump(f)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(do_client())
