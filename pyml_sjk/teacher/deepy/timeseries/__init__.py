import numpy as np
import pandas as pd
from pandas import Series, DataFrame

def time_step_seq(data, look_back, look_forward, fillna=None, dropna=True):
    """
    시계열 데이터를 지도학습의 입력으로 사용하기 위한 형상 형성.

    Params:
        look_back (int): t-2, t-1, t, ...
        look_forward (int): 각 행별 t+1, t+2, ... 
    Returns:
        backward_sequences (ndarray)
        forward_sequences (ndarray)     
    """
    nfeatures = 1 if data.ndim == 1 else data.shape[-1]
    
    data = DataFrame(data)

    # backward shift data
    #   includes x(t)
    backward_frames = _gen_backward_frames(
        data, look_back, fillna, dropna, nfeatures)

    # forward shift data
    forward_frames = _gen_forward_frames(
        data, look_forward, fillna, dropna, nfeatures)

    time_step_frames = pd.concat(backward_frames + forward_frames, axis=1)
    if fillna is not None:
        time_step_frames = time_step_frames.fillna(fillna)
    elif dropna:
        time_step_frames = time_step_frames.dropna()

    # split back backward/forward frames
    split_point = sum(frames.shape[-1] for frames in backward_frames)    
    backward_frames = time_step_frames.iloc[:, :split_point]
    forward_frames = time_step_frames.iloc[:, split_point:]

    # format
    backward_sequences = _format_sequences(backward_frames, look_back+1, nfeatures)    
    forward_sequences = _format_sequences(forward_frames, look_forward, nfeatures)
    
    return backward_sequences, forward_sequences

def _gen_backward_frames(data, look_back, fillna, dropna, nfeatures):
    # backward shift data
    #   includes x(t)
    shift_backward = [data]
    for i in range(look_back):
        shift_backward.insert(0, data.shift(i+1))
    return shift_backward

def _format_sequences(seqs_frame, time_steps, nfeatures):
    # 각 time step별 형상 형성
    sequences = []
    for i in range(0, nfeatures * (time_steps), nfeatures):
        X_t = seqs_frame.iloc[:, i:i+nfeatures].values
        sequences.append(X_t)
    # (time steps, samples, features)
    sequences = np.asarray(sequences)
    # (samples, time step, features)
    sequences = sequences.transpose(1, 0, 2)
    return sequences

def _gen_forward_frames(data, look_forward, fillna, dropna, nfeatures):
    # forward shift data
    shift_forward = []
    for i in range(1, look_forward+1):
        shift_forward.append(data.shift(i * -1))
    return shift_forward
