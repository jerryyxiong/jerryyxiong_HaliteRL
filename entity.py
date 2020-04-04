class Position:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.x}, {self.y})'


class Entity(Position):
    def __init__(self, owner_id, id_, x, y):
        self.owner_id = owner_id
        self.id = id_
        super().__init__(x, y)

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id}, owner_id={self.owner_id}, Position({self.x}, {self.y}))'

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id


class Shipyard(Entity):
    pass


class Dropoff(Entity):
    pass


class Ship(Entity):
    def __init__(self, owner_id, id_, x, y, halite, inspired):
        super().__init__(owner_id, id_, x, y)
        self.halite = halite
        self.inspired = inspired

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id}, Position({self.x}, {self.y}), Halite={self.halite})'


class Command:
    def __init__(self, owner_id, target_id):
        self.owner_id = owner_id
        self.target_id = target_id

    def __repr__(self):
        return f'{self.__class__.__name__}(owner_id={self.owner_id}, target_id={self.target_id})'


class MoveCommand(Command):
    def __init__(self, owner_id, target_id, direction: str):
        super().__init__(owner_id, target_id)
        self.direction = direction  # N S E W O

    @property
    def direction_vector(self):
        if self.direction.upper() == 'O':
            return Position(0, 0)
        if self.direction.upper() == 'N':
            return Position(0, 1)
        if self.direction.upper() == 'S':
            return Position(0, -1)
        if self.direction.upper() == 'E':
            return Position(1, 0)
        if self.direction.upper() == 'W':
            return Position(-1, 0)
        else:
            raise ValueError(f'Invalid Direction: {self.direction}')

    def __repr__(self):
        return f'{self.__class__.__name__}(owner_id={self.owner_id}, target_id={self.target_id}, direction={self.direction})'


class SpawnShipCommand(Command):
    def __repr__(self):
        return f'{self.__class__.__name__}(owner_id={self.owner_id})'


class ConstructDropoffCommand(Command):
    pass
