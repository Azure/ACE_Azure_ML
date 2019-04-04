import hickle as hkl
from settings import *
import numpy as np
import matplotlib.pyplot as plt

# Data files
# train_file = os.path.join(KITTI_DATA_DIR, 'X_train.hkl')
# train_sources = os.path.join(KITTI_DATA_DIR, 'sources_train.hkl')

# change this if you decided to skip frames in the video sequences. Note that this is not supported for anomaly detection yet, because we still need to implement how to properly add anomalies to the model output (mse) data frame if video frames are skipped
skip_frames = 0

train_file = os.path.join(DATA_DIR, 'X_test.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

X = hkl.load(train_file)
sources = hkl.load(train_sources)

n_images, height, width, depth = X.shape

for i in range(int(148/(skip_frames + 1)), int(153/(skip_frames + 1))):
    # random_image = np.random.randint(0,n_images)
    
    plt.imshow(X[i])
    plt.title("Frame: %s" % (i + 1))
    # plt.imshow(X[random_image])
    plt.show()

print("Done.")