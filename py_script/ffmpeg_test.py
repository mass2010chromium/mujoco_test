import mediapy

import numpy as np

frames = []
for i in range(30):
    frames.append(np.array(np.random.random((256, 256)) * 255, dtype=np.uint8))

mediapy.write_video('test.mp4', frames, fps=30)
