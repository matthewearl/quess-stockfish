import argparse
import asyncio
import logging
import random
import signal


_DEPTHS = range(2, 10)


async def _make_proc(args):
    return await asyncio.create_subprocess_exec(
        'stdbuf', '-eL', '-oL',
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )


async def _read_output(proc, stream, prefix, queue):
    while True:
        line = await stream.readline()
        if line:
            await queue.put((prefix, line.decode().strip()))
        else:
            await queue.put((prefix, None))
            break


async def run_game():
    parser = argparse.ArgumentParser(description="have the bot play itself")
    parser.add_argument("--exe", type=str, required=True,
                        help="Quake executable path")
    parser.add_argument("--basedir", type=str, default='.',
                        help="Directory containing id1 directory")
    parser.add_argument("--game", type=str, default='quess134',
                        help="Name of the game directory inside the base"
                             "directory")
    parser.add_argument("--pgn", type=str, default=None,
                        help="Replay game from a pgn string")
    parser.add_argument('--joequake', action='store_true',
                        help="Pass when connecting to a JQ server for "
                             "high-res angles.  Not needed for quakespasm and "
                             "derivatives.")
    args = parser.parse_args()

    server_args = [
        args.exe,
        '-basedir', args.basedir,
        '-mem', '512',
        '-game', args.game,
        '-dedicated', '3',
        '+teamplay', '2',
        '+coop', '1',
        '+timelimit', '10800',
        '+map', 'quess1',
    ]

    client_args = [
        'python',
        'quessclient.py',
        '--basedir', args.basedir,
    ]
    if args.joequake:
        client_args.append('--joequake')

    if args.pgn is not None:
        client_args.extend(['--pgn', args.pgn])

    server_proc = await _make_proc(server_args)
    client_proc1 = await _make_proc(client_args
                                    + ['--depth', str(random.choice(_DEPTHS))])
    client_proc2 = await _make_proc(client_args
                                    + ['--depth', str(random.choice(_DEPTHS))])

    all_procs = {
        'server': server_proc, 'client1': client_proc1, 'client2': client_proc2
    }

    labels = {
        'server': "S",
        'client1': "C1",
        'client2': "C2",
    }

    queue = asyncio.Queue()
    for proc_name, proc in all_procs.items():
        asyncio.create_task(_read_output(proc, proc.stdout,
                            (proc_name, proc.pid, 'stdout'), queue))
    for proc_name, proc in all_procs.items():
        asyncio.create_task(_read_output(proc, proc.stderr,
                            (proc_name, proc.pid, 'stderr'), queue))

    while True:
        (proc_name, pid, stream_name), line = await queue.get()
        print(f'[{labels[proc_name]} {pid} {stream_name}] {line}')
        if proc_name != 'server' and 'outcome: Outcome' in line:
            print(f'game over:  {line}')
            break

    for proc in all_procs.values():
        proc.send_signal(signal.SIGINT)

    await asyncio.gather(*(p.wait() for p in all_procs.values()))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(run_game())
