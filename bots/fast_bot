from collections import defaultdict
from engine import Player, Game
from entity import Position, MoveCommand, SpawnShipCommand
from viewer import Replayer
from data_utils import pad_cells


class FastBot(Player):
    def __init__(self):
        self.id = None
        self.shipyard = None
        self.halite = 0
        self.returning = defaultdict(bool)  # ship id: bool

    def start(self, id_, map_width, map_height, game):
        self.id = id_
        for shipyard in game.constructs.values():
            if shipyard.owner_id == self.id:
                self.shipyard = shipyard
                break

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
                    if self.returning[ship.id]:
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

                    if xdl <= xdr != 0:
                        next_pos[ship.id] = Position((ship.x - 1) % game.width, ship.y)
                        commands[ship.id] = MoveCommand(self.id, ship.id, 'W')
                    elif xdr < xdl:
                        next_pos[ship.id] = Position((ship.x + 1) % game.width, ship.y)
                        commands[ship.id] = MoveCommand(self.id, ship.id, 'E')
                    elif ydd <= ydu != 0:
                        next_pos[ship.id] = Position(ship.x, (ship.y - 1) % game.height)
                        commands[ship.id] = MoveCommand(self.id, ship.id, 'S')
                    elif ydu < ydd:
                        next_pos[ship.id] = Position(ship.x, (ship.y + 1) % game.height)
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
                current = next_pos[ship.id]
                while current != Position(ship.x, ship.y):
                    if game.cells[next_pos[ship.id].y][next_pos[ship.id].x][2] == -1 or game.ships[game.cells[next_pos[ship.id].y][next_pos[ship.id].x][2]].owner_id != self.id:
                        break
                    if current == next_pos[game.cells[next_pos[ship.id].y][next_pos[ship.id].x][2]]:
                        break
                    current = next_pos[game.cells[next_pos[ship.id].y][next_pos[ship.id].x][2]]
                else:
                    continue
                next_ships[next_pos[ship.id].y][next_pos[ship.id].x].remove(ship)
                next_pos[ship.id].x = ship.x
                next_pos[ship.id].y = ship.y
                commands[ship.id] = MoveCommand(self.id, ship.id, 'O')
                q.extend(next_ships[ship.y][ship.x])
                next_ships[ship.y][ship.x].append(ship)

        ret = list(commands.values())
        if len(next_ships[self.shipyard.y][self.shipyard.x]) == 0 and self.halite >= 1000 and rem > 100:
            ret.append(SpawnShipCommand(self.id, None))
        return ret


if __name__ == '__main__':
    bot1 = FastBot()
    bot2 = FastBot()
    players, cell_data, bank_data, owner_data, collisions = Game.run_game([bot1, bot2], return_replay=True, map_gen='perlin')

    my_replay = Replayer.from_data(players, cell_data, bank_data, owner_data, collisions)
    my_replay.run()
