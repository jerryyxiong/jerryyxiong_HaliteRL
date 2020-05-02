import math
import numpy as np
from entity import Position


def roll_to_zero(cells, pos: Position):
    return np.roll(cells, (-pos.y, -pos.x), (0, 1))


def pad_cells(cells, shipyard_pos: Position):
    rolled = roll_to_zero(cells, shipyard_pos)
    tiled = np.tile(rolled, (math.ceil(128 / rolled.shape[0]), math.ceil(128 / rolled.shape[1]), 1))
    return np.roll(tiled, (64, 64), (0, 1))[:128, :128]

