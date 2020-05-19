"""
implement on plotting callback that inputs 2 generators,
performs inference on each signal in the val and train set
and plots the results and saves the image

"""

from keras.callbacks import Callback

class PlotterCallback(Callback):

    def __init__(self, train_generator, val_generator):
        Callback.__init__(self)
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass