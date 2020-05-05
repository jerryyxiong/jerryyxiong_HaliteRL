from fast_bot import FastBot
from engine import Game
from viewer import Replayer
import time
import datetime


map_sizes = (32, 40, 48, 56, 64)
totals = {size: 0 for size in map_sizes}
for size in map_sizes:
    num_turns = size * 3.125 + 300
    for _ in range(10):
        start_time = time.time_ns()
        Game.run_game([FastBot(), FastBot()], map_width=size, save_name='data/test_game', verbosity=0)
        totals[size] += (time.time_ns() - start_time) / 10 ** 9
    print(f'Map size {size} average turns/sec: {num_turns / totals[size] * 10:.2f}')
    print(f'Map size {size} average time required for 1 mil turns: {datetime.timedelta(seconds=1000000 * totals[size] / 10 / num_turns)}')
print(f'Overall average turns/sec: {450 / sum(totals.values()) * 50:.2f}')
print(f'Overall average time required for 1 mil turns: {datetime.timedelta(seconds=1000000 * sum(totals.values()) / 50 / 450)}')
