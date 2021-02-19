# %%
def parse_file(file, unique_num_beats):
    """
    Parses object of MidiFile class and returns data frame
    
    Arguments:
    file -- object of MidiFile class
    unique_num_beats -- numpy array of acceptable note durations
                        in number of beats, calculated with
                        make_unique_beats function 
    
    Return:
    df -- data frame with note pitch and duration in seconds
    """
    
    # this should not output a data frame, should have implemented with np for
    # better performance
    
    import numpy as np
    import pandas as pd
    import mido

    assert(isinstance(file, mido.midifiles.midifiles.MidiFile))

    notes = []
    time = 0
    numerator = 0
    denominator = 0
    clocks_per_click = 0
    notated_32nd_notes_per_beat = 0
    tempo = 0
    
    # get all midi messages in the file
    
    for msg in file:
        
        if msg.dict()['type'] == 'time_signature':
            numerator = msg.dict()['numerator']
            denominator = msg.dict()['denominator']
            clocks_per_click = msg.dict()['clocks_per_click']
            notated_32nd_notes_per_beat = msg.dict()['notated_32nd_notes_per_beat']
        
        if msg.dict()['type'] == 'set_tempo':
            tempo = msg.dict()['tempo']
        
        if msg.dict()['type'] == 'note_on':
            channel = msg.dict()['channel']
            pitch = msg.dict()['note']
            velocity = msg.dict()['velocity']
            time += msg.dict()['time']
            notes.append((channel,pitch,velocity,time,tempo))     
          
    # create data frame        
          
    df = pd.DataFrame(notes, columns=['channel','pitch','velocity','time','tempo'])    
    
    # identify individual notes and clean up
    
    df.sort_values(by = ['time', 'velocity'],
                   axis=0, inplace=True, ignore_index=True)
    
    df.insert(loc=df.shape[1],
              column='duration', 
              value=-df.time.diff(periods=-1))
    
    df = df.loc[df.duration != 0]
    
    df.dropna(inplace=True)
    
    df.insert(loc = df.shape[1],
              column = 'beats',
              value = 1e6 * df.duration / df.tempo)
    
    df.drop(labels = ['channel','velocity','time','tempo','duration'],
            axis = 1,
            inplace = True)
    
    df.reset_index(drop=True, inplace=True)
    
    # include columns with note name and octave
    
    note_dict = {0:'C', 1:'C#/Db', 2:'D', 3:'D#/Eb',
                 4:'E', 5:'F', 6:'F#/Gb', 7:'G',
                 8:'G#/Ab', 9:'A', 10:'A#/Bb', 11:'B'}
    
    note_names = [note_dict[pitch%12] for pitch in df.pitch]
    
    note_octaves = [int(pitch/12)-1 for pitch in df.pitch]
    
    df.insert(loc = 1,
              column = 'note',
              value = note_names)
    
    df.insert(loc = 2,
              column = 'octave',
              value = note_octaves)
    
    # include column with standardized number of beats 
    
    beats = []
    
    for beat in df.beats:
        beats.append(unique_num_beats[np.abs(beat - unique_num_beats).argmin()])
    
    df.drop(labels = 'beats',
            axis = 1,
            inplace = True)
    
    df.insert(loc = df.shape[1],
              column = 'beats',
              value = beats)
    
    return df

# %%
def make_unique_beats():
    """
    Creates numpy array of unique acceptable note durations 
    in number of beats
    
    Return:
    unique_beats -- numpy array with unique number of beats
    """
    
    import numpy as np
    
    a = np.array([1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8]).reshape((-1,1))
    b = np.array([1/3,2/3,5/4,3/2,7/4,1]).reshape((1,-1))
    unique_beats = np.unique(np.round(np.dot(a,b), 4)) 
    
    return unique_beats

# %%
def encode_one_hot(df, unique_num_beats):
    """
    Creates one-hot encoding of the pitch and duration of each of the notes
    in a piece, as well as a tuple with the melodic range
    
    Arguments:
    df -- data frame created with the parse_file function,
          containing the pitches and beats for a piece
    unique_num_beats -- numpy array of acceptable note durations
                        in number of beats, calculated with
                        make_unique_beats function
                        
    Return:
    encoding_pitch -- one-hot encoding of the pitch
                      (time along rows + 1, pitch along columns)
    encoding_beats -- one-hot encoding of note duration
                      (time along rows + 1, duration along columns)
    encoding_eof -- one_hot encoding of end-of-file (positive after last note)
                    (time along rows + 1, 1 column)
    piece_range -- tuple with the lowest and highest notes
    """
    
    import numpy as np
    import pandas as pd
    
    piece_length = df.shape[0]
    
    lowest_note = 36
    highest_note = 36 + 12*4
    num_possible_notes = highest_note - lowest_note + 1
    
    num_possible_beats = unique_num_beats.shape[0]
    
    encoding_pitch = np.zeros((piece_length + 1, num_possible_notes)) # +1 accounts for eof
    encoding_beats = np.zeros((piece_length + 1, num_possible_beats)) # +1 accounts for eof
    encoding_eof = np.zeros((piece_length + 1, 1))
    encoding_eof[-1] = 1 # the last entry is positive, indicating the end-of-file
        
    for row in df.itertuples():
        encoding_pitch[row.Index, row.pitch - lowest_note] = 1
        encoding_beats[row.Index] = (unique_num_beats == row.beats).astype(int)
    
    piece_range = (df.pitch.min(), df.pitch.max())
    
    return encoding_pitch, encoding_beats, encoding_eof, piece_range
    
# %%
def create_np_tensor(encoding_pitch, encoding_beats, encoding_eof, batch_size):
    """
    Creates a numpy 3D array from the pitch and beats encoding with a certain batch size
    
    Arguments:
    encoding_pitch -- note pitches encoded via encode_one_hot function
    encoding_beats -- note duration encoded via encode_one_hot function
    batch_size -- number of notes for each batch
    
    Return:
    X -- 3D numpy array with shape (batch_num, batch_size, input_size)
    """
    
    # this should clearly have been implemented with a np.reshape....
    
    import numpy as np
    
    # shorter variable names
    pitch = encoding_pitch
    beats = encoding_beats
    eof = encoding_eof
    
    piece_length = pitch.shape[0]
    
    piece_range = (pitch.sum(axis=0).nonzero()[0][0] + 36,
                   pitch.sum(axis=0).nonzero()[0][-1] + 36)
    
    shape_x = pitch.shape[1] + beats.shape[1] + 1
    T_x = batch_size
    num_batches = int((piece_length-1) / batch_size) + 1
    
    X = np.zeros((num_batches, T_x, shape_x))
    
    # fill X tensor batch by batch
    
    for batch in range(num_batches):
        
        # this bit necessary to accomodate last batch with smaller size
        if batch == num_batches-1:
            batch_notes = piece_length % batch_size
        else:
            batch_notes = batch_size
        
        batch_eof = eof[batch*batch_size:(batch+1)*batch_size, :]
        batch_pitch = pitch[batch*batch_size:(batch+1)*batch_size, :]
        batch_beats = beats[batch*batch_size:(batch+1)*batch_size, :]
        
        X[batch, 0:batch_notes, 0] = batch_eof.flatten()
        X[batch, 0:batch_notes, 1:pitch.shape[1]+1] = batch_pitch
        X[batch, 0:batch_notes, pitch.shape[1]+1:] = batch_beats
        
    return X

# %% 
def transpose_key(X):
    pass

# %%
def create_input_tensor(path, batch_size):
    """
    Prepares 3D input tensor from .midi files in the indicated path
    
    Arguments:
    path -- directory with the .midi files
    batch_size -- number of notes per batch in the tensor
    
    Return:
    X -- 3D tensor of shape(number of batches, batch size, length of encoding)
    """
    
    import numpy as np
    import pandas as pd
    import mido
    #from music_generator_utils import parse_file, make_unique_beats, encode_one_hot, create_np_tensor
    from os import getcwd, chdir, listdir
    
    #batch_size = 32
    
    pitch = []
    beats = []
    eof = []
    
    unique_num_beats = make_unique_beats()
    
    #path = 'original-cello-suites'
    
    for name in listdir(path):
        
        # read file into df
        filepath = path + '\\' + name
        file = mido.MidiFile(filepath)
        df = parse_file(file, unique_num_beats)
        
        # get encodings for pitch, beats and eof
        p, b, e, _ = encode_one_hot(df, unique_num_beats)
        
        # append new encodings to complete lists
        pitch.append(p)
        beats.append(b)
        eof.append(e)
        
    # concatenate lists of encodings
    pitch = np.concatenate(pitch)
    beats = np.concatenate(beats)
    eof = np.concatenate(eof)
        
    # create 3D tensor with shape(batch, batch_duration, input_dim)
    X = create_np_tensor(pitch, beats, eof, batch_size)
    
    return X