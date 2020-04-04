import time
from collections import defaultdict
import pyglet
from pyglet.gl import *

key = pyglet.window.key
RAD2DEG = 57.29577951308232
COLORS = (255, 0, 0), (0, 255, 0), (6, 152, 253), (255, 125, 0)
SHIPYARD_COLORS = (200, 0, 0), (0, 200, 0), (0, 125, 200), (200, 100, 0)
DROPOFF_COLORS = (150, 0, 0), (0, 150, 0), (0, 90, 150), (150, 75, 0)
SHIP_MARGIN = 5
STORAGE_MARGIN = 6
SQUARE_SIZE = 20
CELL_MAX_BRIGHTNESS_HALITE = 1500


class Replayer(pyglet.window.Window):
    def __init__(self, width, height, players, cell_data, bank_data, owner_data):
        super().__init__(width, height, fullscreen=False)
        self.players = players
        self.cell_data = cell_data
        self.bank_data = bank_data
        self.owner_data = owner_data
        self.current_turn = 0
        self.max_turns = len(cell_data)
        self.alive = 1
        self.keys = defaultdict(bool)
        self.key_press_time = {}
        self.vertices = []

        self.colors = {}
        for i, player in enumerate(players):
            self.colors[player.id] = COLORS[i]

    def on_close(self):
        self.alive = 0

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:  # [ESC]
            self.alive = 0
        elif symbol == key.RIGHT:
            self.keys['RIGHT'] = True
            self.current_turn = min(self.current_turn + 1, len(self.cell_data) - 1)
            self.key_press_time['RIGHT'] = time.time_ns()
        elif symbol == key.LEFT:
            self.keys['LEFT'] = True
            self.current_turn = max(self.current_turn - 1, 0)
            self.key_press_time['LEFT'] = time.time_ns()

    def on_key_release(self, symbol, modifiers):
        if symbol == key.RIGHT:
            self.keys['RIGHT'] = False
        elif symbol == key.LEFT:
            self.keys['LEFT'] = False

    def render(self):
        self.clear()

        if self.keys['RIGHT']:
            if time.time_ns() - self.key_press_time['RIGHT'] >= 5e8:
                self.current_turn = min(self.current_turn + 1, len(self.cell_data) - 1)
        if self.keys['LEFT']:
            if time.time_ns() - self.key_press_time['LEFT'] >= 5e8:
                self.current_turn = max(self.current_turn - 1, 0)

        cells = self.cell_data[self.current_turn]

        batch = pyglet.graphics.Batch()

        ship_count = defaultdict(int)
        held_halite = defaultdict(int)

        # Draws the grid.
        for y in range(len(cells)):
            for x in range(len(cells[0])):
                if cells[y][x][1] == -1:
                    batch.add(4, pyglet.gl.GL_QUADS, None,
                              ('v2i', (x*SQUARE_SIZE, y*SQUARE_SIZE, (x+1)*SQUARE_SIZE, y*SQUARE_SIZE, (x+1)*SQUARE_SIZE,
                                       (y+1)*SQUARE_SIZE, x*SQUARE_SIZE, (y+1)*SQUARE_SIZE)),
                              ('c3B', (cells[y][x][0] * 255 // CELL_MAX_BRIGHTNESS_HALITE, ) * 12))
                elif cells[y][x][1] < len(self.players):
                    batch.add(4, pyglet.gl.GL_QUADS, None,
                              ('v2i', (x*SQUARE_SIZE, y*SQUARE_SIZE, (x+1)*SQUARE_SIZE, y*SQUARE_SIZE, (x+1)*SQUARE_SIZE,
                                       (y+1)*SQUARE_SIZE, x*SQUARE_SIZE, (y+1)*SQUARE_SIZE)),
                              ('c3B', SHIPYARD_COLORS[cells[y][x][1]] * 4))
                else:
                    batch.add(4, pyglet.gl.GL_QUADS, None,
                              ('v2i', (x * SQUARE_SIZE, y * SQUARE_SIZE, (x + 1) * SQUARE_SIZE, y * SQUARE_SIZE,
                                       (x + 1) * SQUARE_SIZE,
                                       (y + 1) * SQUARE_SIZE, x * SQUARE_SIZE, (y + 1) * SQUARE_SIZE)),
                              ('c3B', DROPOFF_COLORS[self.owner_data[cells[y][x][1]]] * 4))
                if cells[y][x][2] != -1:
                    ship_count[self.owner_data[cells[y][x][2]]] += 1
                    held_halite[self.owner_data[cells[y][x][2]]] += cells[y][x][3]
                    batch.add(4, pyglet.gl.GL_QUADS, None,
                              ('v2i', (x * SQUARE_SIZE + SHIP_MARGIN, y * SQUARE_SIZE + SHIP_MARGIN,
                                       (x + 1) * SQUARE_SIZE - SHIP_MARGIN, y * SQUARE_SIZE + SHIP_MARGIN,
                                       (x + 1) * SQUARE_SIZE - SHIP_MARGIN, (y + 1) * SQUARE_SIZE - SHIP_MARGIN,
                                       x * SQUARE_SIZE + SHIP_MARGIN, (y + 1) * SQUARE_SIZE - SHIP_MARGIN)),
                              ('c3B', COLORS[self.owner_data[cells[y][x][2]]] * 4))
                    batch.add(4, pyglet.gl.GL_QUADS, None,
                              ('v2i', (x * SQUARE_SIZE + STORAGE_MARGIN, y * SQUARE_SIZE + STORAGE_MARGIN,
                                       (x + 1) * SQUARE_SIZE - STORAGE_MARGIN, y * SQUARE_SIZE + STORAGE_MARGIN,
                                       (x + 1) * SQUARE_SIZE - STORAGE_MARGIN, (y + 1) * SQUARE_SIZE - STORAGE_MARGIN,
                                       x * SQUARE_SIZE + STORAGE_MARGIN, (y + 1) * SQUARE_SIZE - STORAGE_MARGIN)),
                              ('c3B', [a * cells[y][x][3] // 1000 for a in COLORS[self.owner_data[cells[y][x][2]]]] * 4))

        # Text
        pyglet.text.Label(text=f'Turn: {self.current_turn + 1}/{self.max_turns}', x=self.width - 270,
                          y=self.height - 25, batch=batch)
        for i in range(len(self.players)):
            pyglet.text.Label(text=f'Banked Halite: {self.bank_data[self.current_turn][i]}',
                              x=self.width-270,
                              y=self.height-50-100*i,
                              color=COLORS[i]+(255,),
                              batch=batch)
            pyglet.text.Label(text=f'Held Halite: {held_halite[i]}',
                              x=self.width-270,
                              y=self.height-75-100*i,
                              color=COLORS[i]+(255,),
                              batch=batch)
            pyglet.text.Label(text=f'Ships: {ship_count[i]}',
                              x=self.width-270,
                              y=self.height-100-100*i,
                              color=COLORS[i]+(255,),
                              batch=batch)

        batch.draw()
        self.flip()

    @staticmethod
    def from_data(players, cell_data, stat_data, owner_data):
        return Replayer(20 * len(cell_data[0][0]) + 300, 20 * len(cell_data[0]), players, cell_data, stat_data,
                        owner_data)

    def run(self):
        while self.alive == 1:
            self.render()
            self.dispatch_events()
