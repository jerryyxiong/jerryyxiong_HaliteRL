from bot_fast import FastBot
from engine import Game

NUM_TRAIN = 300
SPARSE_FREQ = 20
EPS = 0.05
NUM_VAL = 30
for i in range(NUM_TRAIN):
    if i % SPARSE_FREQ == 0:
        print(f'Creating sparse training game {i+1:03}/{NUM_TRAIN}')
        Game.run_game([FastBot(eps=EPS), FastBot(eps=EPS)], map_width=32, map_gen='sparse', save_name=f'data/game{i}', verbosity=0)
    else:
        print(f'Creating training game {i+1:03}/{NUM_TRAIN}')
        Game.run_game([FastBot(eps=EPS), FastBot(eps=EPS)], map_width=32, save_name=f'data/game{i}', verbosity=0)
for i in range(NUM_VAL):
    print(f'Creating validation game {i+1:03}/{NUM_VAL}')
    Game.run_game([FastBot(), FastBot()], map_width=32, save_name=f'val/game{i}', verbosity=0)
