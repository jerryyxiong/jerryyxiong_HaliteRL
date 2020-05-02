from time import time_ns
from math import sqrt
from collections import defaultdict
import pyglet
from pyglet.gl import *

key = pyglet.window.key

CELL_SIZE = 20
SHIP_MARGIN = 5
STORAGE_MARGIN = 7
COLORS = (255, 0, 0), (0, 255, 0), (6, 152, 253), (255, 125, 0)
SHIPYARD_CR = 0.6  # CR: Color Ratio
DROPOFF_CR = 0.8
BOOM_COLOR = (255, 255, 0)


def _hue(halite_amt: int):
    return min(int(sqrt(halite_amt / 1024) * 255), 255)


def _ship_vertices(x, y, direction, storage=False):
    margin = STORAGE_MARGIN if storage else SHIP_MARGIN
    if direction == (0, 0):
        return (CELL_SIZE * x + margin, CELL_SIZE * y + margin,
                CELL_SIZE * x + margin, CELL_SIZE * (y + 1) - margin,
                CELL_SIZE * (x + 1) - margin, CELL_SIZE * (y + 1) - margin,
                CELL_SIZE * (x + 1) - margin, CELL_SIZE * y + margin)
    elif direction == (-1, 0):
        return (CELL_SIZE * (x + 1) - margin, CELL_SIZE * y + CELL_SIZE // 2,
                CELL_SIZE * x + margin, CELL_SIZE * y + margin,
                CELL_SIZE * x + margin, CELL_SIZE * (y + 1) - margin,
                CELL_SIZE * x + margin, CELL_SIZE * (y + 1) - margin)
    elif direction == (0, -1):
        return (CELL_SIZE * x + CELL_SIZE // 2, CELL_SIZE * (y + 1) - margin,
                CELL_SIZE * x + margin, CELL_SIZE * y + margin,
                CELL_SIZE * (x + 1) - margin, CELL_SIZE * y + margin,
                CELL_SIZE * (x + 1) - margin, CELL_SIZE * y + margin)
    elif direction == (1, 0):
        return (CELL_SIZE * x + margin, CELL_SIZE * y + CELL_SIZE // 2,
                CELL_SIZE * (x + 1) - margin, CELL_SIZE * y + margin,
                CELL_SIZE * (x + 1) - margin, CELL_SIZE * (y + 1) - margin,
                CELL_SIZE * (x + 1) - margin, CELL_SIZE * (y + 1) - margin)
    elif direction == (0, 1):
        return (CELL_SIZE * x + CELL_SIZE // 2, CELL_SIZE * y + margin,
                CELL_SIZE * x + margin, CELL_SIZE * (y + 1) - margin,
                CELL_SIZE * (x + 1) - margin, CELL_SIZE * (y + 1) - margin,
                CELL_SIZE * (x + 1) - margin, CELL_SIZE * (y + 1) - margin)
    else:
        raise ValueError(f'Invalid direction: {direction}')


class Replayer(pyglet.window.Window):
    def __init__(self, width, height, players, cell_data, bank_data, owner_data, collisions):
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        template = Config(sample_buffers=1, samples=4)
        try:
            config = screen.get_best_config(template)
        except pyglet.window.NoSuchConfigException:
            template = Config()
            config = screen.get_best_config(template)

        super().__init__(width, height, config=config)
        self.players = players
        self.cell_data = cell_data
        self.bank_data = bank_data
        self.owner_data = owner_data
        self.collisions = collisions

        self.alive = 1
        self.keys = defaultdict(bool)
        self.key_press_time = {}
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)

        self.current_turn = 0
        self.current_rendered = None
        self.max_turns = len(cell_data)
        self.selected_cell = None
        self.turn_label = pyglet.text.Label(text=f'Turn: {self.current_turn + 1}/{self.max_turns}', x=self.width - 270,
                                            y=self.height - 25, bold=True, batch=self.batch)
        self.bank_labels = []
        self.held_labels = []
        self.ship_count_labels = []

        for i in range(len(self.players)):
            self.bank_labels.append(pyglet.text.Label(
                x=self.width - 270,
                y=self.height - 50 - 100 * i,
                color=COLORS[i] + (255,),
                bold=True,
                batch=self.batch))
            self.held_labels.append(pyglet.text.Label(
                x=self.width - 270,
                y=self.height - 75 - 100 * i,
                color=COLORS[i] + (255,),
                bold=True,
                batch=self.batch))
            self.ship_count_labels.append(pyglet.text.Label(
                x=self.width - 270,
                y=self.height - 100 - 100 * i,
                color=COLORS[i] + (255,),
                bold=True,
                batch=self.batch))

        cells = cell_data[0]
        self.cell_quads = []
        for y in range(len(cells)):
            self.cell_quads.append([])
            for x in range(len(cells[0])):
                if cells[y][x][1] != -1:
                    # Renders shipyards
                    self.cell_quads[y].append(self.batch.add(
                        4, GL_QUADS, self.background,
                        ('v2i/static', (x * CELL_SIZE, y * CELL_SIZE,
                                        (x + 1) * CELL_SIZE, y * CELL_SIZE,
                                        (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                        x * CELL_SIZE, (y + 1) * CELL_SIZE)),
                        ('c3B/static', tuple(round(a * SHIPYARD_CR) for a in COLORS[cells[y][x][1]]) * 4)))
                else:
                    # Renders normal cells
                    self.cell_quads[y].append(self.batch.add(
                        4, GL_QUADS, self.background,
                        ('v2i/static', (x * CELL_SIZE, y * CELL_SIZE,
                                        (x + 1) * CELL_SIZE, y * CELL_SIZE,
                                        (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                        x * CELL_SIZE, (y + 1) * CELL_SIZE)),
                        ('c3B/dynamic', (min(int(sqrt(cells[y][x][0] / 1024) * 255), 255),) * 12)))

        self.ship_vertices = {}
        self.storage_vertices = {}
        self.selection_quad = self.batch.add(
            4, GL_QUADS, self.foreground,
            ('v2i/dynamic', (0,) * 8)
        )
        self.selection_quad_visible = False
        self.selection_cell_label = pyglet.text.Label(
                    text='',
                    x=self.width - 270,
                    y=50,
                    bold=True,
                    batch=self.batch)
        self.selection_ship_label = pyglet.text.Label(
                    text='',
                    x=self.width - 270,
                    y=25,
                    bold=True,
                    batch=self.batch)

    def on_close(self):
        self.alive = 0

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.alive = 0
        elif symbol == key.RIGHT:
            self.keys[key.RIGHT] = True
            self.current_turn = min(self.current_turn + 1, len(self.cell_data) - 1)
            self.key_press_time[key.RIGHT] = time_ns()
        elif symbol == key.LEFT:
            self.keys[key.LEFT] = True
            self.current_turn = max(self.current_turn - 1, 0)
            self.key_press_time[key.LEFT] = time_ns()

    def on_key_release(self, symbol, modifiers):
        if symbol == key.RIGHT:
            self.keys[key.RIGHT] = False
        elif symbol == key.LEFT:
            self.keys[key.LEFT] = False

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.current_rendered = None
            self.selected_cell = x // CELL_SIZE, y // CELL_SIZE
            if self.selected_cell[0] >= len(self.cell_data[0][0]):
                self.selected_cell = None
                return

    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.current_rendered = None
            self.selected_cell = x // CELL_SIZE, y // CELL_SIZE
            if self.selected_cell[0] >= len(self.cell_data[0][0]):
                self.selected_cell = None

                return

    def update(self, dt):
        if self.keys[key.RIGHT] and time_ns() - self.key_press_time[key.RIGHT] >= 5e8 and self.current_turn + 1 < self.max_turns:
            self.current_turn += 1
        elif self.keys[key.LEFT] and time_ns() - self.key_press_time[key.LEFT] >= 5e8 and self.current_turn > 0:
            self.current_turn -= 1
        elif self.current_rendered == self.current_turn:
            return

        self.current_rendered = self.current_turn

        cells = self.cell_data[self.current_turn]
        boom = self.collisions[self.current_turn]
        if self.current_turn > 0:
            prev_cells = self.cell_data[self.current_turn - 1]
            prev_boom = self.collisions[self.current_turn - 1]
        else:
            prev_cells = cells
            prev_boom = boom
        if self.current_turn + 1 < len(self.cell_data):
            next_cells = self.cell_data[self.current_turn + 1]
            next_boom = self.collisions[self.current_turn + 1]
        else:
            next_cells = cells
            next_boom = boom

        ship_count = defaultdict(int)
        held_halite = defaultdict(int)

        living_ships = set()

        # Draws the grid.
        for y in range(len(cells)):
            for x in range(len(cells[0])):
                if boom[y][x]:
                    self.cell_quads[y][x].colors = BOOM_COLOR * 4
                elif cells[y][x][1] != prev_cells[y][x][1] or (cells[y][x][1] != -1 and (prev_boom[y][x] or next_boom[y][x])):  # Means that cell was turned into a dropoff
                    self.cell_quads[y][x].colors = tuple(
                        int(DROPOFF_CR * a) for a in COLORS[self.owner_data[cells[y][x][1]]]) * 4
                # draws halite, maybe because going back from creating a dropoff next turn or just boomed
                elif cells[y][x][0] != prev_cells[y][x][0] or cells[y][x][1] != next_cells[y][x][1] or prev_boom[y][x] or next_boom[y][x]:
                    self.cell_quads[y][x].colors = (_hue(cells[y][x][0]),) * 12
                if cells[y][x][2] != -1:  # If there is a ship
                    living_ships.add(cells[y][x][2])
                    ship_count[self.owner_data[cells[y][x][2]]] += 1
                    held_halite[self.owner_data[cells[y][x][2]]] += cells[y][x][3]

                    if cells[y][x][2] in self.ship_vertices:  # If this ship was drawn before
                        d = (0, 0)
                        for delta in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                            if cells[y][x][2] == prev_cells[(y + delta[1]) % len(cells)][(x + delta[0]) % len(cells[0])][2]:
                                d = delta
                                break

                        new_ship_vertices = _ship_vertices(x, y, d)
                        new_storage_vertices = _ship_vertices(x, y, d, storage=True)
                        # if self.ship_vertices[cells[y][x][2]].get_size() != len(new_ship_vertices) // 2:
                        #     self.ship_vertices[cells[y][x][2]].resize(len(new_ship_vertices) // 2)
                        #     self.storage_vertices[cells[y][x][2]].resize(len(new_ship_vertices) // 2)
                        self.ship_vertices[cells[y][x][2]].vertices = new_ship_vertices
                        self.storage_vertices[cells[y][x][2]].vertices = new_storage_vertices
                        self.storage_vertices[cells[y][x][2]].colors = tuple(
                            round(COLORS[self.owner_data[cells[y][x][2]]][i % 3] / 255 * _hue(cells[y][x][3]))
                            for i in range(3 * len(new_ship_vertices) // 2)
                        )
                    else:
                        for delta in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):
                            if cells[y][x][2] == prev_cells[(y + delta[1]) % len(cells)][(x + delta[0]) % len(cells[0])][2]:
                                new_ship_vertices = _ship_vertices(x, y, delta)
                                new_storage_vertices = _ship_vertices(x, y, delta, storage=True)
                                self.ship_vertices[cells[y][x][2]] = self.batch.add(
                                    4, GL_QUADS, self.foreground, ('v2i/stream', new_ship_vertices),
                                    ('c3B/static', COLORS[self.owner_data[cells[y][x][2]]] * 4)
                                )
                                self.storage_vertices[cells[y][x][2]] = self.batch.add(
                                    4, GL_QUADS, self.foreground, ('v2i/stream', new_storage_vertices),
                                    ('c3B/stream', (0,) * 12)
                                )

        delete = []
        for id_, vertices in self.ship_vertices.items():
            if id_ not in living_ships:
                vertices.delete()
                self.storage_vertices[id_].delete()
                delete.append(id_)
        for id_ in delete:
            del self.ship_vertices[id_]
            del self.storage_vertices[id_]

        self.turn_label.text = f'Turn: {self.current_turn + 1}/{self.max_turns}'
        for i in range(len(self.players)):
            self.bank_labels[i].text = f'Bank: {self.bank_data[self.current_turn][i]}'
            self.held_labels[i].text = f'Held: {held_halite[i]}'
            self.ship_count_labels[i].text = f'Ships: {ship_count[i]}'

        if self.selected_cell is not None:
            if not self.selection_quad_visible:
                self.selection_quad_visible = True
            self.selection_quad.vertices = (
                self.selected_cell[0] * CELL_SIZE - 2, self.selected_cell[1] * CELL_SIZE - 2,
                self.selected_cell[0] * CELL_SIZE - 2, self.selected_cell[1] * CELL_SIZE + 2,
                self.selected_cell[0] * CELL_SIZE + 2, self.selected_cell[1] * CELL_SIZE + 2,
                self.selected_cell[0] * CELL_SIZE + 2, self.selected_cell[1] * CELL_SIZE - 2,)
            sel = self.selected_cell
            self.selection_cell_label.text = f'Cell ({sel[0]}, {sel[1]}) Halite: {cells[sel[1]][sel[0]][0]}'
            if cells[sel[1]][sel[0]][2] == -1:
                self.selection_ship_label.text = ''
            else:
                self.selection_ship_label.text = f'Ship ID {cells[sel[1]][sel[0]][2]} Halite: {cells[sel[1]][sel[0]][3]}'
        elif self.selection_quad_visible:
            self.selection_quad.vertices = (0,) * 8
            self.selection_cell_label.text = ''
            self.selection_ship_label.text = ''

    def render(self):
        self.clear()
        self.batch.draw()
        self.flip()

    @staticmethod
    def from_data(players, cell_data, stat_data, owner_data, collisions):
        return Replayer(CELL_SIZE * len(cell_data[0][0]) + 300, CELL_SIZE * len(cell_data[0]), players, cell_data,
                        stat_data, owner_data, collisions)

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1 / 60)
        while self.alive == 1:
            self.render()
            self.dispatch_events()
            pyglet.clock.tick()
            # print(pyglet.clock.get_fps())
