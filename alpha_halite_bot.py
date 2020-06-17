import random
import tensorflow as tf
import numpy as np
from engine import Player, Game
from entity import MoveCommand, SpawnShipCommand, ConstructDropoffCommand
from viewer import Replayer


STORE_PATH = r'C:\Users\Jerry Xiong\PycharmProjects\AlphaHalite\checkpoints'
MIN_EPSILON = 0.1
TAU = 0.08
GAMMA = .99
BATCH_SIZE = 32


def deep_q_model():
    pp_cell_input = tf.keras.Input(shape=(64, 64, 4))  # single turn's cell data
    score_input = tf.keras.Input(shape=(6,))  # width, height, my bank, maximum enemy bank, turn, max turns
    conv1_output = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu')(
        pp_cell_input)
    conv2_output = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation='relu')(
        conv1_output)
    flatten_output = tf.keras.layers.Flatten()(conv2_output)
    concat_output = tf.keras.layers.concatenate([flatten_output, score_input])
    dense1_output = tf.keras.layers.Dense(units=64, activation='relu', kernel_initializer='he_normal')(concat_output)
    dense2_output = tf.keras.layers.Dense(units=64, activation='relu', kernel_initializer='he_normal')(dense1_output)
    outputs = tf.keras.layers.Dense(6, kernel_initializer='he_normal')(dense2_output)
    return tf.keras.Model(inputs=[pp_cell_input, score_input], outputs=outputs)


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._i = 0
        self._available_samples = 0
        self._cell_data_buffer = np.zeros((max_memory, 32, 32, 4), dtype=int)
        self._score_data_buffer = np.zeros((max_memory, 6), dtype=int)
        self._next_cell_data_buffer = np.zeros((max_memory, 32, 32, 4), dtype=int)
        self._next_score_data_buffer = np.zeros((max_memory, 6), dtype=int)
        self._actions_buffer = np.zeros(max_memory, dtype=int)
        self._rewards_buffer = np.zeros(max_memory, dtype=int)
        self._terminals_buffer = np.zeros(max_memory, dtype=bool)

    def add_sample(self, cell_data, score_data, next_cell_data, next_score_data, action, reward, terminal):
        self._cell_data_buffer[self._i] = cell_data
        self._score_data_buffer[self._i] = score_data
        self._next_cell_data_buffer[self._i] = next_cell_data
        self._next_score_data_buffer[self._i] = next_score_data
        self._actions_buffer[self._i] = action
        self._rewards_buffer[self._i] = reward
        self._terminals_buffer[self._i] = terminal

        self._i += 1
        self._available_samples = min(self._available_samples + 1, self._max_memory)

    def take_samples(self, num_samples):
        if num_samples < self._available_samples:
            raise ValueError('Not enough samples to take samples from')
        else:
            indices = np.random.randint(self._available_samples, size=num_samples)
            return (self._cell_data_buffer[indices],
                    self._score_data_buffer[indices],
                    self._next_cell_data_buffer[indices],
                    self._next_score_data_buffer[indices],
                    self._actions_buffer[indices],
                    self._rewards_buffer[indices],
                    self._terminals_buffer[indices])


def update_model(primary_model, target_model, tau=TAU):
    for a, b in zip(target_model.trainable_variables, primary_model.trainable_variables):
        a.assign(a * (1 - tau) + b * tau)


def train(primary_network, memory, target_network):
    cell_data, score_data, next_cell_data, next_score_data, actions, rewards, terminals = memory.take_samples(BATCH_SIZE)

    print(cell_data)
    print(score_data)
    print(next_cell_data)
    print(next_score_data)

    primary_q = primary_network.predict((cell_data, score_data))
    primary_q_next = primary_network.predict((next_cell_data, next_score_data))

    print(primary_q)
    print(primary_q_next)

    updates = rewards  # IGNORE THE FACT THAT THIS IS A VIEW AND PRETEND ITS A COPY
    primary_q_target = primary_q  # IGNORE THE FACT THAT THIS IS A VIEW AND PRETEND ITS A COPY REEEEEEEEEEEEEEEEEEEEEE
    primary_next_action = np.argmax(primary_q_next, axis=1)
    target_model_q = target_network.predict((cell_data, score_data))

    # big brain numpy array math
    # What this SHOULD do is add gamma * max future reward (from target network) to updates, but only at indices where
    # there actually is a future reward (not terminal).
    # Then, it SHOULD try and set primary_q_target to updates at the correct places
    updates[np.invert(terminals)] += GAMMA * target_model_q[np.invert(terminals), primary_next_action[np.invert(terminals)]]
    primary_q_target[np.invert(terminals), actions] = updates
    loss = primary_network.train_on_batch((cell_data, score_data), primary_q_target)
    return loss


def get_processed_cell_data(id_, game: Game):
    pp_cells = np.copy(game.cells)
    for y in range(len(pp_cells)):
        for x in range(len(pp_cells[0])):
            if pp_cells[y][x][1] in game.constructs:
                if game.constructs[pp_cells[y][x][1]].owner_id == id_:
                    pp_cells[y][x][1] = 1
                else:
                    pp_cells[y][x][1] = -1
            else:
                pp_cells[y][x][1] = 0

            if pp_cells[y][x][2] != -1:
                if game.ships[pp_cells[y][x][2]].owner_id == id_:
                    pp_cells[y][x][2] = 1
                else:
                    pp_cells[y][x][2] = -1
            else:
                pp_cells[y][x][2] = 0
    return np.tile(pp_cells, (2, 2))[:64, :64]


def center_around_ship(pp_cells, ship_x, ship_y):
    return np.roll(pp_cells, shift=(len(pp_cells) // 2 - ship_y, len(pp_cells[0]) // 2 - ship_x), axis=(0, 1))


def get_score_data(id_, game):
    # width, height, my bank, max enemy bank, turn, max turns
    return (game.width,
            game.height,
            game.bank[id_],
            max(game.bank[i] for i in range(4) if i != id_),
            game.turn,
            game.max_turns)


def calculate_reward(id_, game: Game):
    return game.bank[id_]


class AlphaHaliteBot(Player):
    def __init__(self, primary_model, epsilon):
        self.id = None
        self.primary_model = primary_model
        self.epsilon = epsilon
        self.memories = []

    def start(self, id_, map_width, map_height, cells):
        self.id = id_

    def step(self, game):
        commands = []
        cell_data = get_processed_cell_data(game, self.id)
        score_data = get_score_data(self.id, game)
        for ship in game.ships:
            if ship.owner_id != self.id:
                continue
            if random.random() < self.epsilon:
                action: int = random.randint(6)
            else:
                pp_cells = center_around_ship(cell_data, ship.x, ship.y)
                action: int = np.argmax(self.primary_model((pp_cells, score_data)))[0]
            if action == 0:  # stand still
                pass
            elif action == 1:
                commands.append(MoveCommand(self.id, ship.id, 'N'))
            elif action == 2:
                commands.append(MoveCommand(self.id, ship.id, 'E'))
            elif action == 3:
                commands.append(MoveCommand(self.id, ship.id, 'S'))
            elif action == 4:
                commands.append(MoveCommand(self.id, ship.id, 'W'))
            elif action == 5:
                commands.append(ConstructDropoffCommand(self.id, None))
        # def add_sample(self, cell_data, score_data, next_cell_data, next_score_data, action, reward, terminal)
        self.memories.append((
            cell_data,
            score_data,
            
        ))
        return commands


if __name__ == '__main__':
    my_primary_model = deep_q_model()
    my_target_model = deep_q_model()
    update_model(my_primary_model, my_target_model, tau=1)

    my_primary_model.compile()
