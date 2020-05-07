from collections import defaultdict

import numpy as np

from engine import Player, Game
from entity import Position, MoveCommand, SpawnShipCommand, ConstructDropoffCommand
from viewer import Replayer
from fast_bot import FastBot


class StandardBot(Player):
    def __init__(self):
        self.id = None
        self.shipyard = None
        self.dropoffs = []
        self.halite = 0
        self.returning = defaultdict(bool)  # ship id: bool
        self.new_dropoffs = []
        self.pd = None  # planned dropoff

    def start(self, id_, map_width, map_height, game):
        self.id = id_
        for shipyard in game.constructs.values():
            if shipyard.owner_id == self.id:
                self.shipyard = shipyard
                self.dropoffs.append(shipyard)
                break

    def step(self, game):
        def dist(a, b):
            return min(abs(a.x - b.x), game.width - abs(a.x - b.x)) + min(abs(a.y - b.y), game.height - abs(a.y - b.y))
        commands = {}
        next_pos = {}  # ship id: next position
        next_ships = []  # next_ships[y][x] = list of ships that will end want to move to y x
        for y in range(game.height):
            next_ships.append([])
            for x in range(game.width):
                next_ships[y].append([])

        for dropoff in self.new_dropoffs:
            self.dropoffs.append(game.constructs[game.cells[dropoff.y][dropoff.x][1]])

        # attraction = np.zeros((game.height, game.width))
        dominance = np.zeros((game.height, game.width), dtype=int)
        mined = np.zeros((game.height, game.width), dtype=bool)
        for x in range(game.width):
            for y in range(game.height):
                for dx in range(-5, 6):
                    for dy in range(-5 + abs(dx), 6 - abs(dx)):
                        wx = (x + dx) % game.width
                        wy = (y + dy) % game.height
                        if game.cells[wy][wx][2] != -1:
                            if game.ships[game.cells[wy][wx][2]].owner_id == self.id:
                                dominance[y][x] += 1
                            else:
                                dominance[y][x] -= 1
                        # attraction[y][x] += game.cells[wy][wx][0] / (2 ** (abs(dx) + abs(dy)))

        self.halite = game.bank[self.id]
        if game.max_turns - game.turn > 200:
            for x in range(game.width):
                for y in range(game.height):
                    for dropoff in self.dropoffs:
                        if abs(x - dropoff.x) + abs(y - dropoff.y) < 25:
                            break
                    else:
                        if game.cells[y][x][1] != -1 or dominance[y][x] < 5 or game.max_turns - game.turn <= 100:
                            continue
                        surrounding_halite = 0
                        for dx in range(-10, 11):
                            for dy in range(-10 + abs(dx), 11 - abs(dx)):
                                surrounding_halite += game.cells[(y + dy) % game.height][(x + dx) % game.width][0]
                        if surrounding_halite > 2000:
                            self.pd = Position(x, y)

        dropoff_ship_id = None
        if self.pd is not None:
            mine = [s for s in game.ships.values()
                    if s.owner_id == self.id]
            rich = [s for s in game.ships.values()
                    if s.owner_id == self.id and s.halite + self.halite + game.cells[self.pd.y][self.pd.x][0] >= 4000]

            if len(mine) > 0:
                cm = min(mine, key=lambda s: dist(s, self.pd))
                dropoff_ship_id = cm.id
                if len(rich) > 0:
                    cr = min(rich, key=lambda s: dist(s, self.pd))
                    if dist(cr, self.pd) - 3 < dist(cm, self.pd):
                        dropoff_ship_id = cr.id

        for ship in sorted(game.ships.values(), key=lambda s: s.id):
            if ship.owner_id != self.id:
                continue

            if dropoff_ship_id is not None and ship.id == dropoff_ship_id:
                if ship.x != self.pd.x or ship.y != self.pd.y:
                    t = self.pd
                    # print(f'Ship {dropoff_ship_id} targeting {t} for dropoff construction')
                elif self.halite + game.cells[self.pd.y][self.pd.x][0] + game.ships[dropoff_ship_id].halite >= 4000:
                    commands[ship.id] = ConstructDropoffCommand(self.id, ship.id)
                    self.pd = None
                    self.halite -= 4000
                    self.new_dropoffs.append(Position(ship.x, ship.y))
                    continue
                else:
                    # print(f'Ship {dropoff_ship_id} waiting for dropoff construction halite')
                    next_pos[ship.id] = Position(ship.x, ship.y)
                    continue
            else:
                nd = min(self.dropoffs, key=lambda d: dist(ship, d))  # nearest dropoff

                if ship.halite > 950 or game.turn + dist(ship, nd) + 20 > game.max_turns:
                    self.returning[ship.id] = True
                elif ship.halite == 0:
                    self.returning[ship.id] = False
                if ship.halite < game.cells[ship.y][ship.x][0] // 10:
                    next_pos[ship.id] = Position(ship.x, ship.y)
                    next_ships[next_pos[ship.id].y][next_pos[ship.id].x].append(ship)
                    continue
                if self.returning[ship.id]:
                    t = Position(nd.x, nd.y)
                    # print(f'Ship {ship.id} targeting {t} for a return')
                else:
                    # local mining
                    t = Position(ship.x, ship.y)
                    for dx in range(-3, 4):
                        for dy in range(-3 + abs(dx), 4 - abs(dx)):
                            wx = (ship.x + dx) % game.width
                            wy = (ship.y + dy) % game.height
                            if not mined[wy][wx] and game.cells[wy][wx][0] - 50 * (abs(dx) + abs(dy) + dist(Position(wx, wy), nd)) > game.cells[t.y][t.x][0] - 50 * (dist(t, ship) + dist(t, nd)):
                                t.x = wx
                                t.y = wy

                    # long distance mining
                    if game.cells[t.y][t.x][0] - 50 * (dist(t, ship) + dist(t, nd) - dist(ship, nd)) <= 100:
                        v = game.cells[t.y][t.x][0] / (dist(t, ship) + dist(t, nd) + 1)
                        for dx in range(-10, 11):
                            for dy in range(-10 + abs(dx), 11 - abs(dx)):
                                pos = Position((ship.x + dx) % game.width, (ship.y + dy) % game.height)
                                pnd = min(self.dropoffs, key=lambda d: dist(pos, d))
                                if not mined[pos.y][pos.x] and game.cells[pos.y][pos.x][0] / (dist(pos, ship) + dist(pos, pnd) + 1) > v:
                                    t = pos
                                    v = game.cells[pos.y][pos.x][0] / (abs(dx) + abs(dy) + dist(pos, pnd) + 1)
                        # print(f'Ship {ship.id} targeting {t} for long distance mining')
                    else:
                        pass
                        # print(f'Ship {ship.id} targeting {t} for local mining')
                    mined[t.y][t.x] = True

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
                if len(next_ships[ship.y + y_dir][ship.x]) > 0:
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

        # print(next_pos)
        # print(game.ships)
        # print(next_ships)
        q = [ship for ship in game.ships.values() if ship.owner_id == self.id and ship.id in next_pos and (
                next_pos[ship.id].x != ship.x or next_pos[ship.id].y != ship.y)]
        while q:
            ship = q.pop()
            nx = next_pos[ship.id].x
            ny = next_pos[ship.id].y
            if len(next_ships[ny][nx]) > 1 and not (game.max_turns - game.turn <= 50 and any(d.x == nx and d.y == ny for d in self.dropoffs)):
                # is ship part of cycle, let the cycle do its thing no matter what
                current = next_pos[ship.id]
                while current != Position(ship.x, ship.y):
                    if game.cells[next_pos[ship.id].y][next_pos[ship.id].x][2] == -1 or game.ships[game.cells[next_pos[ship.id].y][next_pos[ship.id].x][2]].owner_id != self.id:
                        break
                    if current == next_pos[game.cells[next_pos[ship.id].y][next_pos[ship.id].x][2]]:
                        break
                    current = next_pos[game.cells[next_pos[ship.id].y][next_pos[ship.id].x][2]]
                else:
                    continue

                # print(f'Stopped ship id {ship.id} to prevent collision')
                next_ships[next_pos[ship.id].y][next_pos[ship.id].x].remove(ship)
                next_pos[ship.id].x = ship.x
                next_pos[ship.id].y = ship.y
                commands[ship.id] = MoveCommand(self.id, ship.id, 'O')
                q.extend(next_ships[ship.y][ship.x])
                next_ships[ship.y][ship.x].append(ship)

        ret = list(commands.values())
        if (len(next_ships[self.shipyard.y][self.shipyard.x]) == 0
                and self.halite >= (1000 if self.pd is None else 5000)
                and game.max_turns - game.turn > 100):
            ret.append(SpawnShipCommand(self.id, None))
        return ret


if __name__ == '__main__':
    bot1 = FastBot()
    bot2 = StandardBot()
    players, cell_data, bank_data, owner_data, collisions = Game.run_game([bot1, bot2], return_replay=True, map_gen='perlin')

    my_replay = Replayer.from_data(players, cell_data, bank_data, owner_data, collisions)
    my_replay.run()
