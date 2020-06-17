import random

from bot_fast import FastBot
from engine import Game

NUM_TRAIN = 1000  # number of full validation games
SPARSE_FREQ = 0.02
DENSE_FREQ = 0.02
EPS = 0.05
NUM_VAL = 50
TRAIN_RATIO = 0.2  # proportion of training turns saved (number of games ran adjusted so total turns saved is same)
VAL_RATIO = 0.1  # proportion of validation turns saved
for i in range(3100, int(NUM_TRAIN / TRAIN_RATIO)):
    x = random.random()
    if x < SPARSE_FREQ:
        print(f'Creating sparse training game {i + 1:03}/{int(NUM_TRAIN / TRAIN_RATIO)}')
        Game.run_game(
            [FastBot(eps=EPS), FastBot(eps=EPS)],
            map_width=32, map_gen='sparse',
            save_name=f'data/game{i}',
            save_split=TRAIN_RATIO,
            verbosity=0)
    elif x < SPARSE_FREQ + DENSE_FREQ:
        print(f'Creating dense training game {i + 1:03}/{int(NUM_TRAIN / TRAIN_RATIO)}')
        Game.run_game(
            [FastBot(eps=EPS), FastBot(eps=EPS)],
            map_width=32, map_gen='dense',
            save_name=f'data/game{i}',
            save_split=TRAIN_RATIO,
            verbosity=0)
    else:
        print(f'Creating perlin training game {i+1:03}/{int(NUM_TRAIN / TRAIN_RATIO)}')
        Game.run_game(
            [FastBot(eps=EPS), FastBot(eps=EPS)],
            map_width=32,
            save_name=f'data/game{i}',
            save_split=TRAIN_RATIO,
            verbosity=0
        )
for i in range(NUM_VAL):
    print(f'Running validation game {i+1:03}/{NUM_VAL}')
    Game.run_game(
        [FastBot(), FastBot()],
        map_width=32,
        save_name=f'val/game{i}',
        save_split=VAL_RATIO,
        verbosity=0)
