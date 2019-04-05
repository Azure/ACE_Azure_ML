'''
Build and train PredNet on UCSD sequences. 
'''

# from settings import WEIGHTS_DIR, DATA_DIR, RESULTS_DIR

# standard modules
import os
import numpy as np
import argparse

# keras modules
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger, Callback
from keras.optimizers import Adam
import tensorflow as tf

import matplotlib.pyplot as plt

# custom modules and scripts
from prednet import PredNet
from data_utils import SequenceGenerator

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

# define input arguments that this script accepts
parser = argparse.ArgumentParser(description='Process input arguments')
parser.add_argument('--data-folder', default='./data/', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--learning_rate', default=1e-3, help='learning rate', type=float, required=False)
parser.add_argument('--lr_decay', default=1e-9, help='learning rate decay', type=float, required=False)
parser.add_argument('--stack_sizes', dest="stack_sizes_arg", default="48 96 192", help='Stack sizes of hidden layers', type=str, required=False)
parser.add_argument('--filter_sizes', dest="filter_sizes", default="3 3 3", help='Filter sizes for the three convolutional layers (A, A_hat, R)', type=str, required=False)
parser.add_argument('--layer_loss_weights', dest="layer_loss_weights_arg", default=[1.0, 0.], help='Layer loss weights', type=list, required=False)
parser.add_argument('--compute_target', dest="compute_target", default=None, help='Name of compute target', type=str, required=False)
parser.add_argument('--batch_size', dest="batch_size", default=4, help='Batch size', type=int, required=False)

# process input arguments
args = parser.parse_args()
learning_rate = args.learning_rate
lr_decay = args.lr_decay
stack_sizes_arg = tuple(map(int, args.stack_sizes_arg.split(' ')))
layer_loss_weights_arg = args.layer_loss_weights_arg
filter_sizes = tuple(map(int, args.filter_sizes.split(' ')))
compute_target = args.compute_target
data_folder = args.data_folder
batch_size = args.batch_size

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
output_dir = 'outputs'
os.makedirs(os.path.join(output_dir), exist_ok=True)

print('training dataset is stored here:', data_folder)

if compute_target:
    from azureml.core import Run

    # start an Azure ML run
    run = Run.get_context()

    run.log('learning_rate', learning_rate)
    run.log('lr_decay', lr_decay)
    run.log('filter_sizes', filter_sizes)
    run.log('stack_sizes', stack_sizes_arg)
    run.log('batch_size', batch_size)

# model parameters
A_filt_size, Ahat_filt_size, R_filt_size = filter_sizes

# format of video frames (size of input layer)
n_channels, im_height, im_width = (3, 152, 232) 

# settings for sampling the video data
nt = 10  # number of timesteps used for sequences in training
samples_per_epoch = 75
nb_epoch = 150
N_seq_val = 15  # number of sequences to use for validation

# settings for training and optimization
loss_type='mean_absolute_error'
optimizer_type='adam'
min_delta=1e-1 #1e-4
patience=2 #10
save_model = True  # if weights will be saved


weights_file = os.path.join(output_dir, 'weights.hdf5')  # where weights will be saved
json_file = os.path.join(output_dir, 'model.json')

# Load data and source files 
train_file = os.path.join(data_folder, 'X_train.hkl')
train_sources = os.path.join(data_folder, 'sources_train.hkl')
val_file = os.path.join(data_folder, 'X_val.hkl')
val_sources = os.path.join(data_folder, 'sources_val.hkl')

# Set model characteristics according to above settings
stack_sizes = (n_channels,) + stack_sizes_arg # 4 layer architecture, with 3 input channels (rgb), and 48, 96, 192 units in the deep layers
nb_layers = len(stack_sizes) # number of layers
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
R_stack_sizes = stack_sizes # number of channels in the representation modules
A_filt_sizes = (A_filt_size,) * (nb_layers - 1) # length is 1 - len(stack_sizes), here targets for layers 2-4 are computered by 3x3 convolutions of errors from layer below
Ahat_filt_sizes = (Ahat_filt_size,) * nb_layers # convolutions of the representation layer for computing the predictions in each layer
R_filt_sizes = (R_filt_size,) * nb_layers  # filter sizes for the representation modules

# Set up how much the error in each layer is taken into account during training
layer_loss_weights = np.array([layer_loss_weights_arg[0]] + [layer_loss_weights_arg[1]] * (nb_layers - 1))  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)

# Set up how much the error in each video frame in a sequence is taken into account
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first (done in next line)
time_loss_weights[0] = 0 # we obviously don't blame the model for not being able to predict the first video frame in a sequence

# build the model (see prednet.py)
prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape) # define tf tensor for inputs
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time

optimizer = Adam(lr=learning_rate, decay=lr_decay)

# put it all together
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss=loss_type, optimizer=optimizer)

# object for generating video sequences for training and validation
train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)


# start with lr of learning_rate and then drop by 1e-1 after .5 * nb_epochs
# lr_schedule = lambda epoch: learning_rate if epoch < nb_epoch / 3 else learning_rate * 1e-1    
# callbacks = [LearningRateScheduler(lr_schedule)]

# add early stopping
callbacks = [EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=0, mode='min')]

if save_model:
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

# log training results to a file
callbacks.append(CSVLogger(filename=os.path.join(output_dir, 'train.log'), separator=',', append=False))

if compute_target:
    class LogRunMetrics(Callback):
        # callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list
            run.log('val_loss', log['val_loss'])

    callbacks.append(LogRunMetrics())

# train the model
history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

plt.figure(figsize=(6, 3))
plt.title('({} epochs)'.format(nb_epoch), fontsize=14)
plt.plot(history.history['val_loss'], 'b-', label='Validation Loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)


if compute_target:
    # log an image
    run.log_image('Validation Loss', plot=plt)
else:
    plt.savefig('val_log.png')
plt.close()

# serialize NN architecture to JSON
model_json = model.to_json()

# save model JSON
with open(os.path.join(output_dir, 'model.json'), 'w') as f:
    f.write(model_json)

# # save model weights
# model.save_weights(os.path.join(output_dir, 'model', 'model.h5')
# print("model saved in ./outputs/model folder")

