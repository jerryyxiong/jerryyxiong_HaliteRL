import random
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import numpy as np
from entity import Position, Ship, Shipyard, Dropoff, MoveCommand, SpawnShipCommand, ConstructDropoffCommand


class Player(ABC):
    @abstractmethod
    def start(self, id_, game):
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
    def __init__(self, players, size):
        self.players = players
        self.size = size
        self.turn = 0
        self.max_turns = self.banks = self.cells = None
        self.constructs = {}
        self.ships = {}
        self.shipyard_pos = {}
        self._next_id = len(players)

    def __repr__(self):
        return f'Game(players={self.players}, size={self.size}, max_turns={self.max_turns})'

    def set_custom_start(self, max_turns, banks, hlt: np.ndarray, constructs: list, ships: list):
        for i, player in enumerate(self.players):
            player.id = i

        self.size = len(hlt)
        if len(hlt[0]) != self.size:
            raise ValueError(f'hlt is not a square, instead it has shape: {hlt.shape}')
        self.max_turns = max_turns
        # Halite bank information: {player id: banked halite}
        self.banks = banks
        # self.cells is a size by size by 4 array.
        # Each cell stores: cell hlt, construct id, ship id, ship hlt
        self.cells = np.zeros(shape=(self.size, self.size, 4), dtype=int)
        self.cells[:, :, 0] = hlt
        # No construct/ship means that the id is set to -1.
        self.cells[:, :, 1:3] = -1
        # Converts list arguments into their {entity id: entity} form, and also updates construct and ship information
        # in self.cells
        self.constructs = {c.id: c for c in constructs}
        self.ships = {s.id: s for s in ships}
        for cons in constructs:
            self.cells[cons.y][cons.x][0] = 0
            self.cells[cons.y][cons.x][1] = cons.id
        for ship in ships:
            self.cells[ship.y][ship.x][2] = ship.id
            self.cells[ship.y][ship.x][3] = ship.halite
        # Sets self.shipyard_pos so that it contains {player id: player's shipyard pos} pairs.
        # In addition, checks that each player has exactly one shipyard.
        for construct in constructs:
            if isinstance(construct, Shipyard):
                if construct.owner_id in self.shipyard_pos:
                    raise ValueError(f'Player id={construct.owner_id} has multiple shipyards')
                self.shipyard_pos[construct.owner_id] = construct.pos
        for player in self.players:
            if player.id not in self.shipyard_pos:
                raise ValueError(f'Player id={player.id} has no shipyard')

    def generate_start(self):
        """
        Sets self to a standard initial game state, generating a map using perlin noise. 
        """
        for i, player in enumerate(self.players):
            player.id = i

        self.max_turns = round(self.size * 3.125) + 300

        self.banks = {player.id: 5000 for player in self.players}

        if len(self.players) == 2:
            perlin = np.square(_generate_perlin_noise_2d((self.size, self.size // 2), (4, 2)))
            noise = np.clip(np.random.normal(1, 0.5, size=(self.size, self.size // 2)), 0.5, 10)
            max_halite = np.amax(perlin * noise)
            actual_max = random.randint(800, 1000)
            left_half = np.clip(perlin * noise * (actual_max / max_halite), 0, 1000).astype(int)
        else:
            perlin = np.square(_generate_perlin_noise_2d((self.size // 2, self.size // 2), (2, 2)))
            noise = np.clip(np.random.normal(1, 0.5, size=(self.size // 2, self.size // 2)), 0.5, 10)
            max_halite = np.amax(perlin * noise)
            actual_max = random.randint(800, 1000)
            upper_left = np.clip(perlin * noise * (actual_max / max_halite), 0, 1000).astype(int)
            left_half = np.concatenate((upper_left, np.flip(upper_left, 0)), 0)
        hlt = np.concatenate((left_half, np.flip(left_half, 1)), 1)

        if len(self.players) == 2:
            self.constructs[0] = Shipyard(self.players[0].id, 0, self.size / 4, self.size / 2)
            self.shipyard_pos[self.players[0].id] = Position(self.size / 4, self.size / 2)
            self.constructs[1] = Shipyard(self.players[1].id, 1, self.size * 3 / 4 - 1, self.size / 2)
            self.shipyard_pos[self.players[1].id] = Position(self.size * 3 / 4 - 1, self.size / 2)
        else:
            self.constructs[0] = Shipyard(self.players[0].id, 0, self.size / 4, self.size / 4)
            self.shipyard_pos[self.players[0].id] = Position(self.size / 4, self.size / 4)
            self.constructs[1] = Shipyard(self.players[1].id, 1, self.size * 3 / 4 - 1, self.size / 4)
            self.shipyard_pos[self.players[1].id] = Position(self.size * 3 / 4 - 1, self.size / 4)
            self.constructs[2] = Shipyard(self.players[2].id, 2, self.size / 4, self.size * 3 / 4 - 1)
            self.shipyard_pos[self.players[2].id] = Position(self.size / 4, self.size * 3 / 4 - 1)
            self.constructs[3] = Shipyard(self.players[3].id, 3, self.size * 3 / 4 - 1, self.size * 3 / 4 - 1)
            self.shipyard_pos[self.players[3].id] = Position(self.size * 3 / 4 - 1, self.size * 3 / 4 - 1)

        self.cells = np.zeros(shape=(self.size, self.size, 4), dtype=int)
        self.cells[:, :, 0] = hlt
        self.cells[:, :, 1:3] = -1
        for shipyard in self.constructs.values():
            self.cells[shipyard.y][shipyard.x][0] = 0
            self.cells[shipyard.y][shipyard.x][1] = shipyard.id

    def run(self, replay=False):
        if replay:
            from viewer import Replayer
            cell_data = np.zeros(shape=(self.max_turns, self.size, self.size, 4), dtype=int)
            bank_data = []
            owner_data = {}
            for construct in self.constructs.values():
                owner_data[construct.id] = construct.owner_id
            collisions = np.zeros((self.max_turns, self.size, self.size), dtype=bool)

        for player in self.players:
            player.start(player.id, self)

        while self.turn < self.max_turns:
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
                    ship.x = (ship.x + command.direction_vector.x) % self.size
                    ship.y = (ship.y + command.direction_vector.y) % self.size
                    moved[ship.id] = True

                elif isinstance(command, SpawnShipCommand):
                    if self.banks[command.owner_id] < 1000:
                        raise ValueError(f'Not enough halite for {command}')
                    self.banks[command.owner_id] -= 1000
                    new_ship = Ship(owner_id=command.owner_id,
                                    id_=self._next_id,
                                    x=self.shipyard_pos[command.owner_id].x,
                                    y=self.shipyard_pos[command.owner_id].y,
                                    halite=0,
                                    inspired=False)
                    self.ships[new_ship.id] = new_ship
                    if replay:
                        owner_data[new_ship.id] = new_ship.owner_id
                    self._next_id += 1

                elif isinstance(command, ConstructDropoffCommand):
                    if command.target_id not in self.ships:
                        raise ValueError(f'Invalid target for {command}')
                    ship = self.ships[command.target_id]
                    cost = 4000 - ship.halite - self.cells[ship.y][ship.x][0]
                    if self.banks[command.owner_id] < cost:
                        raise ValueError(f'Not enough halite for {command}')
                    new_dropoff = Dropoff(command.owner_id, self._next_id, ship.x, ship.y)
                    if self.cells[ship.y][ship.x][1] != -1:
                        raise ValueError(f'Invalid location for {command}')
                    self.banks[command.owner_id] -= cost
                    self.constructs[new_dropoff.id] = new_dropoff
                    self.cells[new_dropoff.y][new_dropoff.x][1] = new_dropoff.id
                    self.cells[new_dropoff.y][new_dropoff.x][0] = 0
                    del self.ships[command.target_id]
                    if replay:
                        owner_data[new_dropoff.id] = new_dropoff.owner_id
                    self._next_id += 1

                else:
                    raise ValueError(f'Invalid command: {command}')

            # Handles collisions
            ship_counts = np.zeros((self.size, self.size), int)
            for ship in self.ships.values():
                ship_counts[ship.y][ship.x] += 1
            delete = []
            for ship in self.ships.values():
                if ship_counts[ship.y][ship.x] > 1:
                    if self.cells[ship.y][ship.x][1] != -1:
                        self.banks[self.constructs[self.cells[ship.y][ship.x][1]].owner_id] += ship.halite
                    else:
                        self.cells[ship.y][ship.x][0] += ship.halite
                    delete.append(ship.id)
                    if replay:
                        collisions[self.turn][ship.y][ship.x] = True
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
                                ship_id = self.cells[(ship.y + dy) % self.size][(ship.x + dx) % self.size][2]
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
                    self.banks[ship.owner_id] += ship.halite
                    ship.halite = 0

            for ship in self.ships.values():
                self.cells[ship.y][ship.x][3] = ship.halite

            if replay:
                np.copyto(cell_data[self.turn], self.cells)
                bank_data.append(self.banks.copy())

            self.turn += 1

        if replay:
            r = Replayer.from_data(self.players, cell_data, bank_data, owner_data, collisions)
            r.run()
