import time
import random

import numpy as np

from collections import defaultdict
from engine import Player, Game
from entity import Position, MoveCommand, SpawnShipCommand
from viewer import Replayer


class FastBot(Player):
    def __init__(self, eps=0.0):
        self.id = None
        self.shipyard = None
        self.halite = 0
        self.returning = defaultdict(bool)  # ship id: bool
        self.map_starting_halite = None
        self.eps = eps

    def start(self, id_, map_width, map_height, game):
        self.id = id_
        for shipyard in game.constructs.values():
            if shipyard.owner_id == self.id:
                self.shipyard = shipyard
                break
        self.map_starting_halite = np.sum(game.cells[:, :, 0])

    def step(self, game):
        def dist(a, b):
            return min(abs(a.x - b.x), game.width - abs(a.x - b.x)) + min(abs(a.y - b.y), game.height - abs(a.y - b.y))
        commands = {}
        next_pos = {}  # ship id: next position
        next_ships = []  # next_ships[y][x] = list of ships that will end want to move to y x
        rem = game.max_turns - game.turn
        for y in range(game.height):
            next_ships.append([])
            for x in range(game.width):
                next_ships[y].append([])

        self.halite = game.bank[self.id]
        for ship in game.ships.values():
            if ship.owner_id == self.id:
                if ship.halite > 800 or game.turn + dist(ship, self.shipyard) + 20 > game.max_turns:
                    self.returning[ship.id] = True
                elif ship.x == self.shipyard.x and ship.y == self.shipyard.y:
                    self.returning[ship.id] = False
                if ship.halite < game.cells[ship.y][ship.x][0] // 10:
                    next_pos[ship.id] = Position(ship.x, ship.y)
                else:
                    if random.random() < self.eps:
                        t = Position(random.randint(0, game.width), random.randint(0, game.height))
                    elif self.returning[ship.id]:
                        t = Position(self.shipyard.x, self.shipyard.y)
                    else:
                        t = Position(ship.x, ship.y)
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                p = Position((ship.x + dx) % game.width, (ship.y + dy) % game.height)
                                if (game.cells[p.y][p.x][0]) / (dist(ship, p) + 1) > game.cells[t.y][t.x][0] / (dist(t, ship) + 1):
                                    t = p
                    xdl = (ship.x - t.x) % game.width
                    xdr = (t.x - ship.x) % game.width
                    ydd = (ship.y - t.y) % game.height
                    ydu = (t.y - ship.y) % game.height

                    if xdl == xdr == 0:
                        x_dir = 0
                    elif xdl <= xdr:
                        x_dir = -1
                    else:
                        x_dir = 1

                    if ydd == ydu == 0:
                        y_dir = 0
                    elif ydd <= ydu:
                        y_dir = -1
                    else:
                        y_dir = 1

                    if x_dir != 0 and y_dir != 0:
                        x_pen = game.cells[ship.y][(ship.x + x_dir) % game.width][0]
                        y_pen = game.cells[(ship.y + y_dir) % game.height][ship.x][0]
                        if len(next_ships[ship.y][(ship.x + x_dir) % game.width]) > 0:
                            x_pen += 3000
                        elif game.cells[ship.y][(ship.x + x_dir) % game.width][2] != -1:
                            x_pen += 300
                        if len(next_ships[(ship.y + y_dir) % game.height][ship.x]) > 0:
                            y_pen += 3000
                        elif game.cells[(ship.y + y_dir) % game.height][ship.x][2] != -1:
                            y_pen += 300
                        if x_pen < y_pen:
                            next_pos[ship.id] = Position((ship.x + x_dir) % game.width, ship.y)
                            if x_dir == -1:
                                commands[ship.id] = MoveCommand(self.id, ship.id, 'W')
                            else:
                                commands[ship.id] = MoveCommand(self.id, ship.id, 'E')
                        else:
                            next_pos[ship.id] = Position(ship.x, (ship.y + y_dir) % game.height)
                            if y_dir == -1:
                                commands[ship.id] = MoveCommand(self.id, ship.id, 'S')
                            else:
                                commands[ship.id] = MoveCommand(self.id, ship.id, 'N')
                    elif x_dir != 0:
                        next_pos[ship.id] = Position((ship.x + x_dir) % game.width, ship.y)
                        if x_dir == -1:
                            commands[ship.id] = MoveCommand(self.id, ship.id, 'W')
                        else:
                            commands[ship.id] = MoveCommand(self.id, ship.id, 'E')
                    elif y_dir != 0:
                        next_pos[ship.id] = Position(ship.x, (ship.y + y_dir) % game.height)
                        if y_dir == -1:
                            commands[ship.id] = MoveCommand(self.id, ship.id, 'S')
                        else:
                            commands[ship.id] = MoveCommand(self.id, ship.id, 'N')
                    else:
                        next_pos[ship.id] = Position(ship.x, ship.y)
                next_ships[next_pos[ship.id].y][next_pos[ship.id].x].append(ship)

        q = [ship for ship in game.ships.values() if ship.owner_id == self.id and ship.id in next_pos and (next_pos[ship.id].x != ship.x or next_pos[ship.id].y != ship.y)]
        while q:
            ship = q.pop()
            nx = next_pos[ship.id].x
            ny = next_pos[ship.id].y
            if len(next_ships[ny][nx]) > 1 and not (rem <= 50 and nx == self.shipyard.x and ny == self.shipyard.y):
                cur = Position(ship.x, ship.y)
                done = False
                visited = set()
                while not done:
                    cur = next_pos[game.cells[cur.y][cur.x][2]]
                    # if hits empty or enemy, then not a cycle
                    if game.cells[cur.y][cur.x][2] == -1 or game.ships[game.cells[cur.y][cur.x][2]].owner_id != self.id:
                        break
                    # if ship stops, then not a cycle
                    if cur == next_pos[game.cells[cur.y][cur.x][2]]:
                        break
                    if cur == Position(ship.x, ship.y):
                        done = True
                        continue
                    elif game.cells[cur.y][cur.x][2] in visited:
                        break
                    visited.add(game.cells[cur.y][cur.x][2])
                else:
                    continue
                next_ships[next_pos[ship.id].y][next_pos[ship.id].x].remove(ship)
                next_pos[ship.id].x = ship.x
                next_pos[ship.id].y = ship.y
                commands[ship.id] = MoveCommand(self.id, ship.id, 'O')
                q.extend(next_ships[ship.y][ship.x])
                next_ships[ship.y][ship.x].append(ship)

        ret = list(commands.values())
        if (len(next_ships[self.shipyard.y][self.shipyard.x]) == 0 and self.halite >= 1000 and rem > 100
                and np.sum(game.cells[:, :, 0]) * 3 > self.map_starting_halite):
            ret.append(SpawnShipCommand(self.id, None))
        return ret

    def __repr__(self):
        return f'FastBot(id={self.id}, eps={self.eps})'


if __name__ == '__main__':
    Replayer.from_data(*Game.run_game([FastBot(), FastBot()], map_gen='dense', return_replay=True)).run()
