import random
from collections import defaultdict
import math
import numpy as np
from entity import Ship, Shipyard, Dropoff, MoveCommand, SpawnShipCommand, ConstructDropoffCommand


class Player:
    def start(self, id_, map_width, map_height, cells):
        pass

    def step(self, game):
        pass


class Game:
    def __init__(self, players, halite_stores, width, height, cells, ships, constructs, shipyard_pos):
        self.players = players
        self.halite_stores = halite_stores  # {player id: player halite stores}
        self.width = width
        self.height = height
        self.cells = cells
        self.ships = ships
        self.constructs = constructs  # {construct id: construct}
        self.shipyard_pos = shipyard_pos  # {player id: player's shipyard position tuple}
        self._next_id = len(players)

    def __getitem__(self, x, y):
        return self.cells[y][x]

    @staticmethod
    def generate_map(players, width, height):
        """Initializes a game"""
        if not (width % 4 == height % 2 == 0):
            raise ValueError(f'Invalid Map Dimensions (width={width}, height={height})')
        if not (len(players) == 2 or (len(players) == 4 and height % 4 == 0)):
            raise ValueError(f'Invalid Map Dimensions (width={width}, height={height})')

        for id_, player in enumerate(players):
            player.id = id_

        halite_stores = {player.id: 5000 for player in players}

        if len(players) == 2:
            left_half = np.random.randint(10, 1000, size=(height, width // 2))
        else:
            upper_left = np.random.randint(10, 1000, size=(height // 2, width // 2))
            left_half = np.concatenate((upper_left, np.flip(upper_left, 0)), 0)
        halite_counts = np.concatenate((left_half, np.flip(left_half, 1)), 1)

        ships = {}

        constructs = {}
        shipyard_pos = {}
        if len(players) == 2:
            constructs[0] = Shipyard(players[0].id, 0, width / 4, height / 2)
            shipyard_pos[players[0].id] = (width / 4, height / 2)
            constructs[1] = Shipyard(players[1].id, 1, width * 3 / 4, height / 2)
            shipyard_pos[players[1].id] = (width * 3 / 4, height / 2)
        else:
            constructs[0] = Shipyard(players[0].id, 0, width / 4, height / 4)
            shipyard_pos[players[0].id] = (width / 4, height / 4)
            constructs[1] = Shipyard(players[1].id, 1, width * 3 / 4, height / 4)
            shipyard_pos[players[1].id] = (width * 3 / 4, height / 4)
            constructs[2] = Shipyard(players[2].id, 2, width / 4, height * 3 / 4)
            shipyard_pos[players[2].id] = (width / 4, height * 3 / 4)
            constructs[3] = Shipyard(players[3].id, 3, width * 3 / 4, height * 3 / 4)
            shipyard_pos[players[3].id] = (width * 3 / 4, height * 3 / 4)

        cells = np.zeros((height, width, 4), int)
        for y in range(height):
            for x in range(width):
                cells[y][x] = (halite_counts[y][x], -1, -1, 0)
        for shipyard in constructs.values():
            cells[shipyard.y][shipyard.x][0] = 0
            cells[shipyard.y][shipyard.x][1] = shipyard.id

        return Game(players, halite_stores, width, height, cells, ships, constructs, shipyard_pos)

    @staticmethod
    def create_game(players, map_width=None, map_height=None, return_replay=False):
        if not (len(players) == 2 or len(players) == 4):
            raise ValueError(f'Invalid number of players: {len(players)}')

        if map_width is None:
            map_width = random.choice((32, 40, 48, 56, 64))
        elif not map_width % 4 == 0:
            raise ValueError(f'Invalid Map Dimensions (width={map_width}, height={map_height})')
        if map_height is None:
            map_height = map_width
        elif not ((len(players) == 2 and map_height % 2 == 0) or (len(players) == 4 and map_height % 4 == 0)):
            raise ValueError(f'Invalid Map Dimensions (width={map_width}, height={map_height})')

        game = Game.generate_map(players, map_width, map_height)
        print(game.constructs.values())

        # Send players initial data
        for player in players:
            player.start(player.id, map_width, map_height, game)

        num_turns = round(max(map_width, map_height) * 3.125) + 300
        if return_replay:
            cell_data = np.empty((num_turns, map_height, map_width, 4), int)
            bank_data = []
            owner_data = {}
        else:
            cell_data = None
            bank_data = None
            owner_data = None

        for turn_num in range(num_turns):
            # Get commands from players
            commands = []
            moved = defaultdict(bool)
            for player in players:
                new_commands = player.step(game)
                for command in new_commands:
                    if command.owner_id != player.id:
                        raise ValueError(f'Cannot issue commands for enemy units: {command}')
                commands.extend(player.step(game))

            # Processes commands.
            for command in commands:
                if isinstance(command, MoveCommand):
                    if command.target_id not in game.ships:
                        raise ValueError(f'Invalid target for {command}')
                    ship = game.ships[command.target_id]
                    if ship.halite < game.cells[ship.y][ship.x][0] // 10 or command.direction == "O":
                        continue
                    ship.halite -= game.cells[ship.y][ship.x][0] // 10
                    ship.x = (ship.x + command.direction_vector.x) % map_width
                    ship.y = (ship.y + command.direction_vector.y) % map_height
                    moved[ship.id] = True

                elif isinstance(command, SpawnShipCommand):
                    if game.halite_stores[command.owner_id] < 1000:
                        raise ValueError(f'Not enough halite for {command}')
                    game.halite_stores[command.owner_id] -= 1000
                    new_ship = Ship(command.owner_id, game._next_id, game.shipyard_pos[command.owner_id][0],
                                    game.shipyard_pos[command.owner_id][1], 0, False)
                    game.ships[new_ship.id] = new_ship
                    owner_data[new_ship.id] = new_ship.owner_id
                    game._next_id += 1

                elif isinstance(command, ConstructDropoffCommand):
                    if command.target_id not in game.ships:
                        raise ValueError(f'Invalid target for {command}')
                    if game.halite_stores[command.owner_id] < 4000:
                        raise ValueError(f'Not enough halite for {command}')
                    new_dropoff = Dropoff(command.owner_id, game._next_id, game.ships[command.target_id].x,
                                          game.ships[command.target_id].y)
                    if game.cells[game.ships[command.target_id].y][game.ships[command.target_id].x][1] != -1:
                        raise ValueError(f'Invalid location for {command}')
                    game.constructs[new_dropoff.id] = new_dropoff
                    owner_data[new_dropoff.id] = new_dropoff.owner_id
                    game.cells[new_dropoff.y][new_dropoff.x][1] = new_dropoff.id
                    game.halite_stores[command.owner_id] -= 4000 - game.ships[command.target_id].halite
                    del game.ships[command.target_id]
                    game._next_id += 1

                else:
                    raise ValueError(f'Invalid command: {command}')

            # Handles collisions
            ship_counts = np.zeros((map_height, map_width), int)
            for ship in game.ships.values():
                ship_counts[ship.y][ship.x] += 1
            delete = []
            for ship in game.ships.values():
                if ship_counts[ship.y][ship.x] > 1:
                    if game.cells[ship.y][ship.x][1] != -1:
                        game.halite_stores[game.constructs[game.cells[ship.y][ship.x][1]].owner_id] += ship.halite
                    else:
                        game.cells[ship.y][ship.x][0] += ship.halite
                    delete.append(ship.id)
                    # print(f'Turn {turn_num}: {ship} had a collision at ({ship.x}, {ship.y})')
            for ship_id in delete:
                del game.ships[ship_id]

            game.cells[:, :, 2:4] = -1
            for ship in game.ships.values():
                game.cells[ship.y][ship.x][2] = ship.id

            # Handles mining, inspiration and dropping of halite
            for ship in game.ships.values():
                ship_cell = game.cells[ship.y][ship.x]

                # Mining
                if not moved[ship.id]:
                    for dx in range(-4, 5):
                        if not 0 <= ship.x + dx < map_width:
                            continue
                        for dy in range(-4 + abs(dx), 5 - abs(dx)):
                            if not 0 <= ship.y + dy < map_height:
                                continue
                            # print(turn_num)
                            if (game.cells[ship.y + dy][ship.x + dx][2] != -1
                                    and game.ships[game.cells[ship.y + dy][ship.x + dx][2]].owner_id != ship.owner_id):
                                ship.inspired = True
                                break
                        if ship.inspired:
                            break
                    else:
                        ship.inspired = False
                    amt_mined = (math.ceil(ship_cell[0] / 4)
                                 if ship.halite + math.ceil(ship_cell[0] / 4) <= 1000
                                 else 1000 - ship.halite)
                    ship_cell[0] -= amt_mined
                    if ship.inspired:
                        ship.halite = min(ship.halite + 3 * amt_mined, 1000)
                    else:
                        ship.halite += amt_mined

                # Dropping off halite
                elif ship_cell[1] != -1 and game.constructs[ship_cell[1]].owner_id == ship.owner_id:
                    game.halite_stores[ship.owner_id] += ship.halite
                    ship.halite = 0

            for ship in game.ships.values():
                game.cells[ship.y][ship.x][3] = ship.halite

            if return_replay:
                np.copyto(cell_data[turn_num], game.cells)
                bank_data.append(game.halite_stores.copy())

        # if replay_file is not None:
        #     np.save(replay_file, cell_data)

        if return_replay:
            return players, cell_data, bank_data, owner_data
