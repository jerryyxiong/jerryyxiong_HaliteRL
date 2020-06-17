# import os

import numpy as np
import tensorflow as tf

from train_supervised import create_unet
from engine import Player, Game, pad_frame, center
from entity import Position, MoveCommand, SpawnShipCommand, ConstructDropoffCommand
from viewer import Replayer
from bot_fast import FastBot


# val_folder = 'val/'
# files = [val_folder + f for f in os.listdir(val_folder)]
# val_dataset = Dataset(files, 16)
# 
# model.evaluate(val_dataset)


class SupervisedBot(Player):
    def __init__(self, model, override_collisions=False, verbose=0):
        self.override_collisions = override_collisions
        self.verbose = verbose

        self.id = None
        self.shipyard = None
        self.halite = 0
        self.map_starting_halite = 0
        self.model = model

    def start(self, id_, map_width, map_height, game):
        self.id = id_
        for shipyard in game.constructs.values():
            if shipyard.owner_id == self.id:
                self.shipyard = shipyard
                break
        self.map_starting_halite = np.sum(game.cells[:, :, 0])

    def step(self, game):
        frame = game.cells.copy()
        no_ship = frame[:, :, 2] == -1
        for ship in game.ships.values():
            frame[:, :, 2][frame[:, :, 2] == ship.id] = (1 if ship.owner_id == self.id else -1)
        frame[:, :, 2][no_ship] = 0
        frame = pad_frame(frame, game.shipyard_pos[self.id])
        frame = np.divide(frame, (1000, 1, 1, 1000))
        meta = (game.width,
                game.max_turns - game.turn,
                game.bank[self.id],
                max(game.bank[p.id] for p in game.players if p.id != self.id))
        meta = np.divide(meta, (64, 500, 10000, 10000))

        moves = self.model.predict([np.array(frame).reshape((1, 128, 128, 4)), np.array(meta).reshape(1, -1)])[0]

        self.halite = game.bank[self.id]

        commands = {}
        next_pos = {}
        next_ships = []  # next_ships[y][x] = list of ships that will end want to move to y x
        rem = game.max_turns - game.turn
        for y in range(game.height):
            next_ships.append([])
            for x in range(game.width):
                next_ships[y].append([])

        for x in range(game.width):
            for y in range(game.height):
                p = center(Position(x, y), self.shipyard, game.width, game.height)
                if game.cells[y][x][2] == -1 or game.ships[game.cells[y][x][2]].owner_id != self.id:
                    continue
                move = np.argmax(moves[p.y][p.x])
                ship = game.ships[game.cells[y][x][2]]
                if move == 0:
                    if self.override_collisions:
                        next_pos[ship.id] = Position(x, y)
                elif game.cells[y][x][3] < game.cells[y][x][0] // 10:
                    if self.verbose == 1:
                        print(f'{game.ships[game.cells[y][x][2]]} attempted to move without sufficient halite')
                    if self.override_collisions:
                        next_pos[ship.id] = Position(x, y)
                elif move == 1:
                    commands[ship.id] = MoveCommand(self.id, ship.id, 'N')
                    if self.override_collisions:
                        next_pos[ship.id] = Position(x, (y + 1) % game.height)
                elif move == 2:
                    commands[ship.id] = MoveCommand(self.id, ship.id, 'E')
                    if self.override_collisions:
                        next_pos[ship.id] = Position((x + 1) % game.width, y)
                elif move == 3:
                    commands[ship.id] = MoveCommand(self.id, ship.id, 'S')
                    if self.override_collisions:
                        next_pos[ship.id] = Position(x, (y - 1) % game.height)
                elif move == 4:
                    commands[ship.id] = MoveCommand(self.id, ship.id, 'W')
                    if self.override_collisions:
                        next_pos[ship.id] = Position((x - 1) % game.width, y)
                elif move == 5:
                    if self.halite + game.cells[y][x][0] + game.cells[y][x][3] >= 4000:
                        commands[ship.id] = ConstructDropoffCommand(self.id, ship.id)
                    elif self.override_collisions:
                        next_pos[ship.id] = Position(x, y)
                if self.override_collisions:
                    next_ships[next_pos[ship.id].y][next_pos[ship.id].x].append(ship)
        if self.override_collisions:
            q = [ship for ship in game.ships.values()
                 if ship.owner_id == self.id and (next_pos[ship.id].x != ship.x or next_pos[ship.id].y != ship.y)]
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
                    if self.verbose == 1:
                        print(f'Overrode collision for {ship}')
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


if __name__ == '__main__':
    my_model = create_unet()
    my_model.load_weights(tf.train.latest_checkpoint('checkpoints/'))
    # Replayer.from_data(*Game.run_game([FastBot(), SupervisedBot(my_model, True)], map_width=32, return_replay=True, verbosity=2)).run()

    NUM_GAMES = 100
    print(f'Win rate: {sum(Game.run_game([FastBot(), SupervisedBot(my_model, True)], map_width=32, verbosity=1) for _ in range(NUM_GAMES)) / NUM_GAMES }')
