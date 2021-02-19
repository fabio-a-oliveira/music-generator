# %% DECLARE STUFF

import mido
import numpy as np
import pandas as pd

# %% CHANGE DIRECTORY

from os import getcwd, chdir

chdir('Documents\\GitHub\\music-generator\\')
getcwd()


# %% OPEN FILE

name = 'cs1-1pre.mid'
path = 'original-cello-suites'
filepath = path + '\\' + name
file = mido.MidiFile(filepath)

# %% PARSE FILE

from music_generator_utils import parse_file

df = parse_file(file)

# %% PARSE ALL FILES

files = []

from music_generator_utils import parse_file, make_unique_beats
from os import listdir

unique_num_beats = make_unique_beats()

path = 'original-cello-suites'
for name in listdir(path):
    filepath = path + '\\' + name
    file = mido.MidiFile(filepath)
    df = parse_file(file, unique_num_beats)
    files.append(df)

# %% ENCODE FILE

from music_generator_utils import encode_notes_one_hot

encode_notes_one_hot(files[0])


from music_generator_utils import encode_pitch_one_hot

encode_pitch_one_hot(list(files[0].pitch))

encode_pitch_one_hot([])


for note in list(files[0].pitch):
    print(note)


note_dict[0]


note_dict = {0:'C', 1:'C#/Db', 2:'D', 3:'D#/Eb',
             4:'E', 5:'F', 6:'F#/Gb', 7:'G',
             8:'G#/Ab', 9:'A', 10:'A#/Bb', 11:'B'}

# %% NUMBER OF NOTES OF EACH DURATION PER MOVEMENT

from music_generator_utils import parse_file, make_unique_beats

for i in range(36):

    name = listdir(path)[i]
    path = 'original-cello-suites'
    filepath = path + '\\' + name
    file = mido.MidiFile(filepath)
    
    unique_num_beats = make_unique_beats()
    df = parse_file(file, unique_num_beats)
    
    print('Movement ' + str(i+1))
    print(df.groupby(by='beats').size())
    print('\n')

# %% ONE-HOT ENCODING OF PITCH AND DURATION

from music_generator_utils import  encode_one_hot, parse_file

file = mido.MidiFile('original-cello-suites\\cs1-1pre.mid')
df = parse_file(file, unique_num_beats)

encoding_pitch, encoding_beats, encoding_eof, piece_range = encode_one_hot(df, unique_num_beats)   

# make sure both are one-hot
pitch.sum(axis=1).mean()
beats.sum(axis=1).mean()

# check ocurrences of each note duration
beats.sum(axis=0)

# run consistency check for all movements

for i in range(36):

    name = listdir(path)[i]
    path = 'original-cello-suites'
    filepath = path + '\\' + name
    file = mido.MidiFile(filepath)
    
    unique_num_beats = make_unique_beats()
    df = parse_file(file, unique_num_beats)
    
    print('Movement ' + str(i+1))
    print(pitch.sum(axis=1).mean())
    print(beats.sum(axis=1).mean())
    print('\n')

# %% KERAS MODEL

from models import model_SimpleRNN
model = model_SimpleRNN()

model.summary()

# %% CREATE 3D INPUT TENSOR WITH BATCHES

encoding_pitch.sum(axis=0).nonzero()[0][0] + 36
encoding_pitch.sum(axis=0).nonzero()[0][-1] + 36
piece_range

nz = encoding_pitch.sum(axis=0).nonzero()
nz.shape
type(nz)
dir(nz)
nz[0][0]

from music_generator_utils import create_np_tensor

X = create_np_tensor(encoding_pitch, encoding_beats, encoding_eof, 32)

X.shape
dir(X)
X.size

from sys import getsizeof
getsizeof(X)
getsizeof(df) * 36 * 12 / 1e6

getsizeof(encoding_pitch)
getsizeof(encoding_beats)

# %% CREATE INPUT TENSOR WITH ALL PIECES

from music_generator_utils import create_input_tensor
from sys import getsizeof

X = create_input_tensor('original-cello-suites', 32)
X.shape
getsizeof(X) / 1e6
X.size
(X != 0).sum()

# %% EVALUATE MODEL

import numpy as np
from models import model_SimpleRNN
from tensorflow.keras.optimizers import Adam 

model = model_SimpleRNN()

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


Y = X.reshape((874*32,101))
Y = X.swapaxes(2,0).swapaxes(1,2)
Y = Y.reshape((101, 874*32))
Y = Y[:, 1:]
Y = np.concatenate([Y, np.zeros((101,1))], axis=1)
Y = Y.reshape((101,874,32))
Y = Y.swapaxes(0,1).swapaxes(1,2)



model.fit(x=X, y=Y, epochs=1000)

model.fit(x=X, epochs=1000)


# %% MODEL COPIED FROM COURSERA

# import numpy as np
# from models import model_coursera
# from tensorflow.keras.optimizers import Adam 

# model = model_coursera()
# opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# Y = X.reshape((874*32,101))
# Y = X.swapaxes(2,0).swapaxes(1,2)
# Y = Y.reshape((101, 874*32))
# Y = Y[:, 1:]
# Y = np.concatenate([Y, np.zeros((101,1))], axis=1)
# Y = Y.reshape((101,874,32))
# Y = Y.swapaxes(0,1).swapaxes(1,2)
# Y = Y.swapaxes(0,1)


# model.fit(x=X, y=Y, epochs=1000)