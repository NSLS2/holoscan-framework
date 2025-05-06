import matplotlib.pyplot as plt
import numpy as np
plt.ion()
# data = np.load('/test_data/output.npy')
data = np.load('../eiger_simulation/test_data/output.npy')
plt.figure(1, clear=True)
plt.subplot(211)
plt.imshow(np.flipud(np.angle(data).T), vmin=-0.5, vmax=0.5)
plt.subplot(212)
plt.imshow(np.flipud(np.abs(data).T), vmin=0, vmax=1)
plt.show()