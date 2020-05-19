from pathlib import Path
from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

from image_builder import EmgImageGenerator

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=(8, 8, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss=keras.losses.mean_absolute_error,
                  optimizer=keras.optimizers.Adam()
                  )
    return model

def get_model_checkpoint():
    date_id = datetime.now().strftime('%Y%m%d')
    path = Path(__file__, '..', 'files', 'weights', date_id, '{epoch:02d}.hdf5').resolve()
    if not path.parents[0]:
        path.parents[0].mkdir(parents=True)
    path = str(path)
    model_checkpoint = ModelCheckpoint(path)
    return model_checkpoint
if __name__ == '__main__':
    model = get_model()

    train_path = Path(__file__, '..', 'files', 'train_annotations.csv')
    val_path = Path(__file__, '..', 'files', 'val_annotations.csv')

    train_emg_gen = EmgImageGenerator(train_path, 16)
    val_emg_gen = EmgImageGenerator(val_path, 16)
    model_checkpoint = get_model_checkpoint()

    model.fit_generator(train_emg_gen.generator(),
                        steps_per_epoch=train_emg_gen.num_samples // train_emg_gen.batch_size,
                        epochs=50,
                        verbose=1,
                        validation_data=val_emg_gen.generator(),
                        validation_steps=val_emg_gen.num_samples // val_emg_gen.batch_size)