import math
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from engine import Player, Game, pad_frame, center
from entity import Position, MoveCommand, SpawnShipCommand, ConstructDropoffCommand
from viewer import Replayer
from bot_fast import FastBot


GAMMA = 0.99
MAX_EPSILON = 0.2
MIN_EPSILON = 0.01
LAMBDA = 0.05
R_WIN = 1e3
R_LOSE = -1e3
R_HLT_BANK = .01
R_HLT_SHIP = .001
R_SHIP = 1


def create_unet():
    frames_input = tf.keras.Input(shape=(128, 128, 4), name='frame_input')
    d7 = layers.Conv2D(64, kernel_size=1, padding='same', kernel_initializer='he_normal')(frames_input)

    d6 = layers.Activation('relu')(d7)
    d6 = layers.BatchNormalization()(d6)
    d6 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(d6)
    d6 = layers.BatchNormalization()(d6)
    d6 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        d6)
    d6 = layers.BatchNormalization()(d6)
    d6 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d6)

    d5 = layers.Activation('relu')(d6)
    d5 = layers.BatchNormalization()(d5)
    d5 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        d5)
    d5 = layers.BatchNormalization()(d5)
    d5 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d5)

    d4 = layers.Activation('relu')(d5)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(d4)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d4)

    d3 = layers.Activation('relu')(d4)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        d3)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d3)

    d2 = layers.Activation('relu')(d3)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        d2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d2)

    d1 = layers.Activation('relu')(d2)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        d1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(d1)

    lf = layers.Activation('relu')(d1)
    lf = layers.BatchNormalization()(lf)
    lf = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(
        lf)
    lf = layers.BatchNormalization()(lf)
    lf = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(lf)
    lf = layers.Reshape((64,))(lf)  # latent frame

    meta_input = tf.keras.Input(shape=(4,), name='meta_input')  # map size, turns remaining, my bank, enemy bank
    meta = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(meta_input)
    meta = layers.BatchNormalization()(meta)
    meta = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(meta)
    meta = layers.BatchNormalization()(meta)

    # latent game state
    lg = layers.Concatenate()([lf, meta])
    lg = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(lg)
    lg = layers.BatchNormalization()(lg)

    # spawn = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(lg)
    # spawn = layers.BatchNormalization()(spawn)
    # spawn = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(spawn)
    # spawn = layers.BatchNormalization()(spawn)
    # spawn = layers.Dense(1, activation='sigmoid', name='spawn_output')(spawn)  # build predictions

    u1 = layers.Reshape((1, 1, 128))(lg)
    u1 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u1)
    u1 = layers.add([d1, u1])
    u1 = layers.Activation('relu')(u1)
    u1 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u1)
    u1 = layers.BatchNormalization()(u1)

    u2 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u1)
    u2 = layers.add([u2, d2])
    u2 = layers.Activation('relu')(u2)
    u2 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u2)
    u2 = layers.BatchNormalization()(u2)

    u3 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u2)
    u3 = layers.add([u3, d3])
    u3 = layers.Activation('relu')(u3)
    u3 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u3)
    u3 = layers.BatchNormalization()(u3)

    u4 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u3)
    u4 = layers.add([u4, d4])
    u4 = layers.Activation('relu')(u4)
    u4 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u4)
    u4 = layers.BatchNormalization()(u4)

    u5 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u4)
    u5 = layers.add([u5, d5])
    u5 = layers.Activation('relu')(u5)
    u5 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u5)
    u5 = layers.BatchNormalization()(u5)

    u6 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u5)
    u6 = layers.add([u6, d6])
    u6 = layers.Activation('relu')(u6)
    u6 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u6)
    u6 = layers.BatchNormalization()(u6)

    u7 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(u6)
    u7 = layers.add([u7, d7])
    u7 = layers.Activation('relu')(u7)

    moves = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(u7)
    moves = layers.BatchNormalization()(moves)
    moves = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(moves)
    moves = layers.BatchNormalization()(moves)
    moves = layers.Conv2D(6, kernel_size=1, name='moves_output')(moves)  # move logits

    # model = tf.keras.Model(inputs=[frames_input, meta_input], outputs=[moves, spawn])
    model = tf.keras.Model(inputs=[frames_input, meta_input], outputs=moves)
    model.summary()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.Huber())
    return model


class Memory:
    def __init__(self, size):
        self.size = size
        self.frames = np.zeros((size, 128, 128, 4), dtype=int)
        self.metas = np.zeros((size, 4), dtype=int)
        self.actions = np.zeros((size, 128, 128), dtype=int)
        self.rewards = np.zeros(size, dtype=int)
        self.terminals = np.zeros(size, dtype=bool)
        self.frames_n = np.zeros((size, 128, 128, 4), dtype=int)
        self.metas_n = np.zeros((size, 4), dtype=int)
        self.i = 0
        self.num_samples = 0

    def add_sample(self, frame, meta, action, reward, terminal, frame_n, meta_n):
        self.frames[self.i] = frame
        self.metas[self.i] = meta
        self.actions[self.i] = action
        self.rewards[self.i] = reward
        self.terminals[self.i] = terminal
        if not terminal:
            self.frames_n[self.i] = frame_n
            self.metas_n[self.i] = meta_n
        self.i = (self.i + 1) % self.size
        self.num_samples = min(self.num_samples + 1, self.size)

    def sample(self, batch_size=32):
        idxs = np.random.default_rng().integers(self.num_samples, size=batch_size)
        return (self.frames[idxs], self.metas[idxs], self.actions[idxs], self.rewards[idxs], self.terminals[idxs],
                self.frames_n[idxs], self.metas_n[idxs])


def train(network, memory):
    frames, metas, actions, rewards, terminals, frames_n, metas_n = memory.sample(16)
    target_q = network.predict([frames, metas])
    updates = rewards.reshape(-1, 1, 1) + ~terminals.reshape(-1, 1, 1) * np.amax(network.predict([frames_n, metas_n]), axis=3)
    for i in range(len(frames)):
        for x in range(128):
            for y in range(128):
                target_q[i][x][y][actions[i][x][y]] = updates[i][x][y]
    loss = network.train_on_batch([frames, metas], target_q)
    return loss


class RLBot(Player):
    def __init__(self, network, memory=None, eps=0.0, override_collisions=False, verbose=0):
        self.network = network
        self.memory: Memory = memory
        self.eps = eps
        self.override_collisions = override_collisions
        self.verbose = verbose

        self.id = None
        self.shipyard = None
        self.halite = None
        self.map_starting_halite = None
        self.prev_action = None
        self.prev_ship_halite = None
        self.prev_ships = None
        self.prev_frame = None
        self.prev_meta = None
        self.total_loss = 0

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

        if self.memory is not None:
            if self.prev_frame is not None:
                ship_hlt_delta = sum(s.halite for s in game.ships.values() if s.owner_id == self.id) - self.prev_ship_halite
                bank_hlt_delta = game.bank[self.id] - self.halite
                ship_delta = (len([s for s in game.ships.values() if s.owner_id == self.id]) - self.prev_ships)
                reward = R_HLT_SHIP * ship_hlt_delta + R_HLT_BANK * bank_hlt_delta + R_SHIP * ship_delta
                self.memory.add_sample(self.prev_frame, self.prev_meta, self.prev_action, reward, False, frame, meta)
            if game.turn + 1 == game.max_turns:
                reward = R_WIN if meta[2] > meta[3] else R_LOSE
                self.memory.add_sample(frame, meta, self.prev_action, reward, True, None, None)
            if self.memory.num_samples >= min(320, self.memory.size / 10):
                self.total_loss += train(self.network, self.memory)

        self.prev_action = np.zeros((128, 128), dtype=int)
        self.prev_ship_halite = sum(s.halite for s in game.ships.values() if s.owner_id == self.id)
        self.prev_ships = len([s for s in game.ships.values() if s.owner_id == self.id])
        self.prev_frame = frame.copy()
        self.prev_meta = meta.copy()

        moves = self.network.predict([np.array(frame).reshape((1, 128, 128, 4)), np.array(meta).reshape(1, -1)])[0]

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
                move = np.argmax(moves[p.y][p.x]) if random.random() < (1 - self.eps) else random.randint(0, 5)
                self.prev_action[x][y] = move
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
                    if self.halite + game.cells[y][x][0] + game.cells[y][x][3] >= 4000 and game.cells[y][x][1] == -1:
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
                    p = Position(ship.x, ship.y)
                    done = False
                    visited = set()
                    while not done:
                        p = next_pos[game.cells[p.y][p.x][2]]
                        # if hits empty or enemy, then not a cycle
                        if game.cells[p.y][p.x][2] == -1 or game.ships[game.cells[p.y][p.x][2]].owner_id != self.id:
                            break
                        # if ship stops, then not a cycle
                        if p == next_pos[game.cells[p.y][p.x][2]]:
                            break
                        if p == Position(ship.x, ship.y):
                            done = True
                            continue
                        elif game.cells[p.y][p.x][2] in visited:
                            break
                        visited.add(game.cells[p.y][p.x][2])
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

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id}, eps={self.eps}, training={self.memory is not None})'


epsilon = MAX_EPSILON
model = create_unet()
model.load_weights('./checkpoints/cp5000_02.ckpt')
mem = Memory(10000)
for step_num in range(500):
    epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * step_num)
    rl_bot = RLBot(model, memory=mem, eps=epsilon)
    Game.run_game([rl_bot, FastBot()], map_width=32, verbosity=0)
    print(f'Loss: {rl_bot.total_loss}')
    if step_num % 20 == 19:
        model.save_weights(f'./checkpoints/RL_checkpoint_{step_num}')
