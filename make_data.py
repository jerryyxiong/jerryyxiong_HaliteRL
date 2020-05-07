from bot_fast import FastBot
from engine import Game


for i in range(20):
    Game.run_game([FastBot(), FastBot()], map_width=32, save_name=f'data/game{i}')
for i in range(5):
    Game.run_game([FastBot(), FastBot()], map_width=32, save_name=f'val/game{i}')
