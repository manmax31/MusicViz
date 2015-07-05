__author__ = 'manabchetia'


import subprocess
import wave
import struct
import numpy
import csv
import pandas as pd
from pydub import AudioSegment
import os
from pprint import pprint as pp
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import warnings
from ggplot import *
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", category=RuntimeWarning)


def read_wav(wav_file):
    """Returns two chunks of sound data from wave file."""
    w = wave.open(wav_file)
    n = 60 * 10000
    if w.getnframes() < n * 2:
        raise ValueError('Wave file too short')
    frames = w.readframes(n)
    wav_data1 = struct.unpack('%dh' % n, frames)
    frames = w.readframes(n)
    wav_data2 = struct.unpack('%dh' % n, frames)

    return wav_data1, wav_data2

def compute_chunk_features(mp3_file):

    """Return feature vectors for two chunks of an MP3 file."""
    out_file = 'temp.wav'
    sound = AudioSegment.from_mp3(mp3_file)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(10000)
    sound.export(out_file, format="wav")
    # Read in chunks of data from WAV file
    wav_data1, wav_data2 = read_wav(out_file)
    return features(wav_data1), features(wav_data2)

def moments(x):
    mean = x.mean()
    std = x.var()**0.5
    skewness = ((x - mean)**3).mean() / std**3
    kurtosis = ((x - mean)**4).mean() / std**4
    return [mean, std, skewness, kurtosis]

def fftfeatures(wavdata):
    f = numpy.fft.fft(wavdata)
    f = f[2:(f.size / 2 + 1)]
    f = abs(f)
    total_power = f.sum()
    f = numpy.array_split(f, 10)
    return [e.sum() / total_power for e in f]

def features(x):
    x = numpy.array(x)
    f = []

    xs = x
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))

    xs = x.reshape(-1, 10).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))

    xs = x.reshape(-1, 100).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))

    xs = x.reshape(-1, 1000).mean(1)
    diff = xs[1:] - xs[:-1]
    f.extend(moments(xs))
    f.extend(moments(diff))

    f.extend(fftfeatures(x))
    return f

def dim_reduction(features):
    pca = PCA(n_components=2)
    red_data =  pca.fit_transform(features)

    return red_data

# Main script starts here
# =======================

if __name__ == '__main__':
    aud_dir = '../babylon/'
    file_names, features1, features2 = [], [], []

    for path, dirs, files in os.walk(aud_dir):
        for f in files:
            file_names.append(f)
            if f.endswith('.mp3'):
                # Skip any non-MP3 files
                # continue
                mp3_file = os.path.join(path, f)

                # Extract the track name (i.e. the file name) plus the names
                # of the two preceding directories. This will be useful
                # later for plotting.
                tail, track = os.path.split(mp3_file)
                tail, dir1 = os.path.split(tail)
                tail, dir2 = os.path.split(tail)

                # Compute features. feature_vec1 and feature_vec2 are lists of floating
                # point numbers representing the statistical features we have extracted
                # from the raw sound data.
                try:
                    feature_vec1, feature_vec2 = compute_chunk_features(mp3_file)
                    features1.append(feature_vec1)
                    features2.append(feature_vec2)
                except:
                    continue
    file_names = filter(lambda x: x.endswith('.mp3'), file_names)
    df = pd.DataFrame(index=file_names, columns={'Features1', 'Features2'})

    print(len(file_names), len(features1), len(features2), '\n')

    df['Features1'] = features1
    df['Features2'] = features2

    features1 /= numpy.max(numpy.abs(features1),axis=0)
    features2 /= numpy.max(numpy.abs(features2),axis=0)


    red_feat1 = numpy.asarray(dim_reduction(features1))
    red_feat2 = numpy.asarray(dim_reduction(features2))

    print(red_feat1)
    print
    print(red_feat2)

    x = red_feat1[:, 0]
    y = red_feat1[:, 1]

    df_plot = pd.DataFrame(red_feat1, index = file_names, columns={'x', 'y'})


    print(df_plot)

    print ggplot(df_plot, aes('x', 'y')) + geom_point(color = 'red') + ggtitle('Features1') + xlab('PC1') + ylab('PC2')

