import random
from engine import Player, Game
from entity import MoveCommand, SpawnShipCommand
from viewer import Replayer


class RandomBot(Player):
    def __init__(self):
        self.id = None
        self.shipyard = None
        self.halite = 0

    def start(self, id_, map_width, map_height, game):
        self.id = id_
        for y in range(map_width):
            for x in range(map_height):
                if game.cells[y][x][1] != -1 and game.constructs[game.cells[y][x][1]].owner_id == self.id:
                    self.shipyard = game.constructs[game.cells[y][x][1]]

    def step(self, game):
        commands = []

        self.halite = game.bank[self.id]
        for ship in game.ships.values():
            if ship.owner_id == self.id:
                new_move = MoveCommand(self.id, ship.id, random.choice(['N', 'S', 'E', 'W', 'O']))
                commands.append(new_move)
        if game.cells[self.shipyard.y][self.shipyard.x][2] == -1 and self.halite >= 1000:
            commands.append(SpawnShipCommand(self.id, None))
        return commands


if __name__ == '__main__':
    bot1 = RandomBot()
    bot2 = RandomBot()
    players, cell_data, bank_data, owner_data = Game.create_game([bot1, bot2], return_replay=True)
    cells = cell_data[0]
    # for y in range(len(cells)):
    #     for x in range(len(cells[0])):
    #         if cells[y][x].halite != states[-1][y][x].halite:
    #             print('WOO')
    #         print(cells[y][x].halite, end=' ')
    #     print()

    my_replay = Replayer.from_data(players, cell_data, bank_data, owner_data)
    my_replay.run()
