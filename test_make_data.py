import numpy as np
import matplotlib.pyplot as plt


with open('data/test_game_50_0.npy', 'rb') as f:
    cells = np.load(f)
    moves = np.load(f)
    spawns = np.load(f)


fig, axes = plt.subplots(2, 4)

axes[0][0].imshow(cells[:, :, 0].astype(float), origin='lower')
axes[0][1].imshow(cells[:, :, 1].astype(float), origin='lower')
axes[0][2].imshow(cells[:, :, 2].astype(float), origin='lower')
axes[0][3].imshow(cells[:, :, 3].astype(float), origin='lower')
axes[1][0].imshow(moves, cmap='tab10', origin='lower')
print(cells[:, :, 2])
print(moves[60:70, 60:70].astype(int))
plt.show()
