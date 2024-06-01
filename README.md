# Quess-stockfish

This is the code behind my Quess bot, which I made a video about here:

[![I made an AI for a classic Quake mod](https://img.youtube.com/vi/q1OlXfFsRX4/maxresdefault.jpg)](https://www.youtube.com/watch?v=q1OlXfFsRX4) 

This has only been tested on Linux, so YMMV.

## Installation

Install a stockfish binary, for example on Ubuntu run:

```bash
sudo apt install stockfish
```

Make and activate a new Python virtualenv (tested with Python 3.10.12).
Checkout and then cd into this repo.  Install dependencies:

```bash
pip install -r requirements.txt
```

Install a Quake source port, I've tested most with
[ironwail](https://github.com/andrei-drexler/ironwail) but I've also had it
working with [joequake](https://github.com/j0zzz/JoeQuake) and the non-remaster
version from Steam.  A port supporting high resolution mouse inputs is
recommended (pretty much any port supporting protocol 666).  For the rest of
this guide I'll assume ironwail.  You'll need the original game's assets and
pak files too, which you can buy from
[Steam](https://store.steampowered.com/app/2310/Quake/) or elsewhere.

Install Quess 1.34.  Check your sha256 sums and file names match:

```
3337368c6258862efbda01553d433e4b62c35fa085bc852117eee79f2fdbfc37  quess134/pak0.pak
29a8b4953b5a1e298f0fcc633138d67299a35aad27c0057a5306e1c9bd56071d  quess134/progs.dat
```

## Running

Start a game in listen mode, with your source port of choice:

```bash
cd <quake-install-dir>
path/to/ironwail -game quess134 -listen 2
```

In another window run the bot:
```bash
cd <quake-install-dir>
python path/to/quessclient.py
```

## Options

```bash
usage: quessclient.py [-h] [--depth DEPTH] [--pgn PGN] [--ip IP]
                      [--port PORT] [--basedir BASEDIR] [--game GAME]
                      [--joequake]

quess-stockfish

options:
  -h, --help         show this help message and exit
  --depth DEPTH      Stockfish search depth
  --pgn PGN          Replay game from a pgn string
  --ip IP            Server to connect to
  --port PORT        Port to connect to
  --basedir BASEDIR  Directory containing id1 directory
  --game GAME        Name of the game directory inside the base directory
  --joequake         Pass when connecting to a JQ server for high-res angles.
                     Not needed for quakespasm and derivatives.
```

## Bugs / limitations

- I *highly* recommend running with a source port that supports high resolution
  inputs.  Without this the bot can misclick and get stuck. If the bot does not
  have high resolution inputs it will emit a warning to the console when
  connecting.
- There is a Quess bug where the grabber (the fiend that removes corpses) gets
  stuck on things.  Occasionally this can block pieces from moving and stall
  the game.
- There is a rare Quess bug where a rook (ogre) attacking a queen (shambler)
  never ends, since the grenades pass straight through the shambler.  In this
  case the game gets stuck.
- The bot is not very tolerant to packet loss, so expect instability if running
  over a real network.