import random
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import numpy as np
from entity import Position, Ship, Shipyard, Dropoff, MoveCommand, SpawnShipCommand, ConstructDropoffCommand


class Player(ABC):
    @abstractmethod
    def start(self, id_, map_width, map_height, game):
        pass

    @abstractmethod
    def step(self, game):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id})' if hasattr(self, 'id') else f'{self.__class__.__name__}()'


def _fade(t):
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


def _generate_perlin_noise_2d(shape, res):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = _fade(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def _generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * _generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def roll_to_zero(cells, pos: Position):
    return np.roll(cells, (-pos.y, -pos.x), (0, 1))


def pad_frame(cells, shipyard_pos: Position):
    """
    Returns the cells padded to 128 by 128 and rolled so that shipyard_pos ends up at (64, 64)
    """
    rolled = roll_to_zero(cells, shipyard_pos)
    tiled = np.tile(rolled, (math.ceil(128 / rolled.shape[0]), math.ceil(128 / rolled.shape[1]), 1))
    return np.roll(tiled, (64, 64), (0, 1))[:128, :128]


def center(pos, cent, w, h):
    """
    Given a position in-game, returns the corresponding position when padded to 128 by 128, around cent, the center.
    """
    if (pos.x - cent.x) % w < (cent.x - pos.x) % w:
        x_adj = 64 + (pos.x - cent.x) % w
    else:
        x_adj = 64 - (cent.x - pos.x) % w
    if (pos.y - cent.y) % h < (cent.y - pos.y) % h:
        y_adj = 64 + (pos.y - cent.y) % h
    else:
        y_adj = 64 - (cent.y - pos.y) % h
    return Position(x_adj, y_adj)


class Game:
    def __init__(self, players, bank, width, height, cells, ships, constructs, shipyard_pos, turn, max_turns):
        self.players = players
        self.bank = bank  # player id: player halite bank
        self.width = width
        self.height = height
        self.cells = cells  # cells[y][x] = [cell halite, construction id, ship id, ship halite] at that position
        self.ships = ships
        self.constructs = constructs  # {construct id: construct}
        self.shipyard_pos = shipyard_pos  # {player id: player's shipyard position tuple}
        self._next_id = len(players)
        self.turn = turn
        self.max_turns = max_turns

    def __repr__(self):
        return (f'Game(players={self.players}, width={self.width}, '
                f'height={self.height}, max_turns={self.max_turns})')

    @staticmethod
    def initialize_custom_position(players, bank, cell_halite, ships, constructs, turn, max_turns):
        w = len(cell_halite)
        h = len(cell_halite[0])
        cells = np.zeros((w, h, 4))
        cells[:, :, 0] = cell_halite
        cells[:, :, 1:3] = -1
        for ship in ships:
            cells[ship.y][ship.x][2] = ship.id
            cells[ship.y][ship.x][3] = ship.halite
        shipyard_pos = {}
        for cons in constructs:
            cells[cons.y][cons.x][1] = cons.id
            if isinstance(cons, Shipyard):
                if cons.owner_id in shipyard_pos:
                    raise ValueError('One player cannot have multiple shipyards')
                shipyard_pos[cons.owner_id] = Position(cons.x, cons.y)
        for player in players:
            if player.id not in shipyard_pos:
                raise ValueError('All players must have a shipyard')

        return Game(players, bank, w, h, cells, ships, constructs, shipyard_pos, turn, max_turns)
    
    def step(self):
        commands = []
        moved = defaultdict(bool)

        # spawns = []
        for player in self.players:
            new_commands = player.step(self)
            for command in new_commands:
                if command.owner_id != player.id:
                    raise ValueError(f'Cannot issue commands for enemy units: {command}')
            commands.extend(new_commands)

        # Processes commands.
        for command in commands:
            if isinstance(command, MoveCommand):
                if command.target_id not in self.ships:
                    raise ValueError(f'Invalid target for {command}')
                ship = self.ships[command.target_id]
                if ship.halite < self.cells[ship.y][ship.x][0] // 10 or command.direction == "O":
                    continue
                ship.halite -= self.cells[ship.y][ship.x][0] // 10
                ship.x = (ship.x + command.direction_vector.x) % self.width
                ship.y = (ship.y + command.direction_vector.y) % self.height
                moved[ship.id] = True

            elif isinstance(command, SpawnShipCommand):
                if self.bank[command.owner_id] < 1000:
                    raise ValueError(f'Not enough halite for {command}')
                self.bank[command.owner_id] -= 1000
                new_ship = Ship(owner_id=command.owner_id,
                                id_=self._next_id,
                                x=self.shipyard_pos[command.owner_id].x,
                                y=self.shipyard_pos[command.owner_id].y,
                                halite=0,
                                inspired=False)
                self.ships[new_ship.id] = new_ship
                self._next_id += 1

            elif isinstance(command, ConstructDropoffCommand):
                if command.target_id not in self.ships:
                    raise ValueError(f'Invalid target for {command}')
                ship = self.ships[command.target_id]
                cost = 4000 - ship.halite - self.cells[ship.y][ship.x][0]
                if self.bank[command.owner_id] < cost:
                    raise ValueError(f'Not enough halite for {command}')
                new_dropoff = Dropoff(command.owner_id, self._next_id, ship.x, ship.y)
                if self.cells[ship.y][ship.x][1] != -1:
                    raise ValueError(f'Invalid location for {command}')
                self.bank[command.owner_id] -= cost
                self.constructs[new_dropoff.id] = new_dropoff
                self.cells[new_dropoff.y][new_dropoff.x][1] = new_dropoff.id
                self.cells[new_dropoff.y][new_dropoff.x][0] = 0
                del self.ships[command.target_id]
                self._next_id += 1

            else:
                raise ValueError(f'Invalid command: {command}')

        # Handles collisions
        ship_counts = np.zeros((self.height, self.width), int)
        for ship in self.ships.values():
            ship_counts[ship.y][ship.x] += 1
        delete = []
        for ship in self.ships.values():
            if ship_counts[ship.y][ship.x] > 1:
                if self.cells[ship.y][ship.x][1] != -1:
                    self.bank[self.constructs[self.cells[ship.y][ship.x][1]].owner_id] += ship.halite
                else:
                    self.cells[ship.y][ship.x][0] += ship.halite
                delete.append(ship.id)
        for ship_id in delete:
            del self.ships[ship_id]

        self.cells[:, :, 2:4] = -1
        for ship in self.ships.values():
            self.cells[ship.y][ship.x][2] = ship.id
            ship.inspired = False

        # Handles mining, inspiration, and and dropping of halite
        for ship in self.ships.values():
            ship_cell = self.cells[ship.y][ship.x]

            # Mining
            if not moved[ship.id]:
                # Inspiration
                if not ship.inspired:
                    for dx in range(-4, 5):
                        for dy in range(-4 + abs(dx), 5 - abs(dx)):
                            ship_id = self.cells[(ship.y + dy) % self.height][(ship.x + dx) % self.width][2]
                            if ship_id != -1 and self.ships[ship_id].owner_id != ship.owner_id:
                                ship.inspired = True
                                self.ships[ship_id].inspired = True
                                break
                        if ship.inspired:
                            break
                amt_mined = (math.ceil(ship_cell[0] / 4)
                             if ship.halite + math.ceil(ship_cell[0] / 4) <= 1000
                             else 1000 - ship.halite)
                ship_cell[0] -= amt_mined
                if ship.inspired:
                    ship.halite = min(ship.halite + 3 * amt_mined, 1000)
                else:
                    ship.halite += amt_mined

            # Dropping off halite
            elif ship_cell[1] != -1 and self.constructs[ship_cell[1]].owner_id == ship.owner_id:
                self.bank[ship.owner_id] += ship.halite
                ship.halite = 0

        for ship in self.ships.values():
            self.cells[ship.y][ship.x][3] = ship.halite

        self.turn += 1

    @staticmethod
    def initialize_game(players, width, height):
        """Initializes a game object with a map generated based on the selected mode."""
        if not (width % 4 == height % 2 == 0):
            raise ValueError(f'Invalid Map Dimensions (width={width}, height={height})')
        if not (len(players) == 2 or (len(players) == 4 and height % 4 == 0)):
            raise ValueError(f'Invalid Map Dimensions (width={width}, height={height})')

        for id_, player in enumerate(players):
            player.id = id_

        bank = {player.id: 5000 for player in players}

        if len(players) == 2:
            perlin = np.square(_generate_perlin_noise_2d((height, width // 2), (4, 2)))
            noise = np.clip(np.random.normal(1, 0.5, size=(height, width // 2)), 0.5, 10)
            max_halite = np.amax(perlin * noise)
            actual_max = random.randint(800, 1000)
            left_half = np.clip(perlin * noise * (actual_max / max_halite), 0, 1000).astype(int)
        else:
            perlin = np.square(_generate_perlin_noise_2d((height // 2, width // 2), (2, 2)))
            noise = np.clip(np.random.normal(1, 0.5, size=(height // 2, width // 2)), 0.5, 10)
            max_halite = np.amax(perlin * noise)
            actual_max = random.randint(800, 1000)
            upper_left = np.clip(perlin * noise * (actual_max / max_halite), 0, 1000).astype(int)
            left_half = np.concatenate((upper_left, np.flip(upper_left, 0)), 0)
        halite_counts = np.concatenate((left_half, np.flip(left_half, 1)), 1)

        ships = {}

        constructs = {}
        shipyard_pos = {}
        if len(players) == 2:
            constructs[0] = Shipyard(players[0].id, 0, width / 4, height / 2)
            shipyard_pos[players[0].id] = Position(width / 4, height / 2)
            constructs[1] = Shipyard(players[1].id, 1, width * 3 / 4 - 1, height / 2)
            shipyard_pos[players[1].id] = Position(width * 3 / 4 - 1, height / 2)
        else:
            constructs[0] = Shipyard(players[0].id, 0, width / 4, height / 4)
            shipyard_pos[players[0].id] = Position(width / 4, height / 4)
            constructs[1] = Shipyard(players[1].id, 1, width * 3 / 4 - 1, height / 4)
            shipyard_pos[players[1].id] = Position(width * 3 / 4 - 1, height / 4)
            constructs[2] = Shipyard(players[2].id, 2, width / 4, height * 3 / 4 - 1)
            shipyard_pos[players[2].id] = Position(width / 4, height * 3 / 4 - 1)
            constructs[3] = Shipyard(players[3].id, 3, width * 3 / 4 - 1, height * 3 / 4 - 1)
            shipyard_pos[players[3].id] = Position(width * 3 / 4 - 1, height * 3 / 4 - 1)

        cells = np.zeros((height, width, 4), int)
        for y in range(height):
            for x in range(width):
                cells[y][x] = (halite_counts[y][x], -1, -1, 0)
        for shipyard in constructs.values():
            cells[shipyard.y][shipyard.x][0] = 0
            cells[shipyard.y][shipyard.x][1] = shipyard.id

        max_turns = round(max(width, height) * 3.125) + 300

        return Game(players, bank, width, height, cells, ships, constructs, shipyard_pos, 0, max_turns)

    @staticmethod
    def run_game(players, map_width: int = None, map_height: int = None, return_replay=False,
                 save_name=None, save_split=1.0, verbosity=1):
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

        game = Game.initialize_game(players, map_width, map_height)

        if verbosity == 1 or verbosity == 2:
            print(f'Now Running: {game}')

        # Send players initial data
        for player in players:
            player.start(player.id, map_width, map_height, game)

        num_turns = round(max(map_width, map_height) * 3.125) + 300
        if return_replay:
            cell_data = np.zeros((num_turns, map_height, map_width, 4), dtype=int)
            bank_data = []
            owner_data = {}
            for construct in game.constructs.values():
                owner_data[construct.id] = construct.owner_id
            collisions = np.zeros((num_turns, map_height, map_width), dtype=bool)
        else:
            cell_data = None
            bank_data = None
            owner_data = None
            collisions = None

        while game.turn < game.max_turns:
            # Get commands from players
            commands = []
            moved = defaultdict(bool)
            frames = []
            moves = []

            save_current_turn = random.random() < save_split

            for player in players:
                if save_name is not None and save_current_turn:
                    frame = game.cells.copy()
                    frames.append(frame)
                    moves.append(np.zeros((128, 128)))

            # Processes commands.
            for command in commands:
                if isinstance(command, MoveCommand):
                    if save_name is not None and save_current_turn:
                        c = center(ship, game.shipyard_pos[command.owner_id], game.width, game.height)
                        moves[command.owner_id][c.y][c.x] = int(command)
                    ship.halite -= game.cells[ship.y][ship.x][0] // 10
                    ship.x = (ship.x + command.direction_vector.x) % map_width
                    ship.y = (ship.y + command.direction_vector.y) % map_height
                    moved[ship.id] = True

                elif isinstance(command, SpawnShipCommand):
                    if game.bank[command.owner_id] < 1000:
                        raise ValueError(f'Not enough halite for {command}')
                    game.bank[command.owner_id] -= 1000
                    new_ship = Ship(owner_id=command.owner_id,
                                    id_=game._next_id,
                                    x=game.shipyard_pos[command.owner_id].x,
                                    y=game.shipyard_pos[command.owner_id].y,
                                    halite=0,
                                    inspired=False)
                    game.ships[new_ship.id] = new_ship
                    # if save_name is not None and save_current_turn:
                    #     spawns[command.owner_id] = 1
                    if return_replay:
                        owner_data[new_ship.id] = new_ship.owner_id
                    game._next_id += 1

                elif isinstance(command, ConstructDropoffCommand):
                    if command.target_id not in game.ships:
                        raise ValueError(f'Invalid target for {command}')
                    ship = game.ships[command.target_id]
                    cost = 4000 - ship.halite - game.cells[ship.y][ship.x][0]
                    if game.bank[command.owner_id] < cost:
                        raise ValueError(f'Not enough halite for {command}')
                    new_dropoff = Dropoff(command.owner_id, game._next_id, ship.x, ship.y)
                    if game.cells[ship.y][ship.x][1] != -1:
                        raise ValueError(f'Invalid location for {command}')
                    game.bank[command.owner_id] -= cost
                    game.constructs[new_dropoff.id] = new_dropoff
                    game.cells[new_dropoff.y][new_dropoff.x][1] = new_dropoff.id
                    game.cells[new_dropoff.y][new_dropoff.x][0] = 0
                    if save_name is not None and save_current_turn:
                        c = center(ship, game.shipyard_pos[command.owner_id], game.width, game.height)
                        moves[command.owner_id][c.y][c.x] = 5
                    if return_replay:
                        owner_data[new_dropoff.id] = new_dropoff.owner_id
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
                        game.bank[game.constructs[game.cells[ship.y][ship.x][1]].owner_id] += ship.halite
                    else:
                        game.cells[ship.y][ship.x][0] += ship.halite
                    delete.append(ship.id)
                    if return_replay:
                        collisions[game.turn][ship.y][ship.x] = True
                    # print(f'Turn {turn_num}: {ship} had a collision at ({ship.x}, {ship.y})')
            for ship_id in delete:
                del game.ships[ship_id]

            game.cells[:, :, 2:4] = -1
            for ship in game.ships.values():
                game.cells[ship.y][ship.x][2] = ship.id
                ship.inspired = False

            # Handles mining, inspiration, and and dropping of halite
            for ship in game.ships.values():
                ship_cell = game.cells[ship.y][ship.x]

                # Mining
                if not moved[ship.id]:
                    # Inspiration
                    if not ship.inspired:
                        for dx in range(-4, 5):
                            for dy in range(-4 + abs(dx), 5 - abs(dx)):
                                ship_id = game.cells[(ship.y + dy) % map_height][(ship.x + dx) % map_width][2]
                                if ship_id != -1 and game.ships[ship_id].owner_id != ship.owner_id:
                                    ship.inspired = True
                                    game.ships[ship_id].inspired = True
                                    break
                            if ship.inspired:
                                break
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
                    game.bank[ship.owner_id] += ship.halite
                    ship.halite = 0

            for ship in game.ships.values():
                game.cells[ship.y][ship.x][3] = ship.halite

            if return_replay:
                np.copyto(cell_data[game.turn], game.cells)
                bank_data.append(game.bank.copy())

        if verbosity == 1 or verbosity == 2:
            print(f'Winner: {max((p for p in game.players), key=lambda p: game.bank[p.id])}, '
                  f'Banks: { {p: game.bank[p.id] for p in players}}')

        if return_replay:
            return players, cell_data, bank_data, owner_data, collisions
        else:
            # returns the winner otherwise
            return max((p.id for p in game.players), key=lambda pid: game.bank[pid])
