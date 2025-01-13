import cv2
import numpy as np
import scipy.fftpack as fftpack
from scipy import signal
from imutils import face_utils
import imutils
import dlib

class HeartRateCalci(object):
    def __init__(self,video_frames):
        # self.detector = dlib.get_frontal_face_detector()
        # self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # self.fa = face_utils.FaceAligner(self.predictor, desiredFaceWidth=256)
        self.video=video_frames
    
    def fft_filter(self,video, freq_min, freq_max, fps):
        fft = fftpack.fft(video, axis=0)
        frequencies = fftpack.fftfreq(video.shape[0], d=1.0 / fps)
        bound_low = (np.abs(frequencies - freq_min)).argmin()
        bound_high = (np.abs(frequencies - freq_max)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0
        iff = fftpack.ifft(fft, axis=0)

        return fft, frequencies


    def find_heart_rate(self,fft, freqs, freq_min, freq_max):
        fft_maximums = []

        for i in range(fft.shape[0]):
            if freq_min <= freqs[i] <= freq_max:
                fftMap = abs(fft[i])
                fft_maximums.append(fftMap.max())
            else:
                fft_maximums.append(0)

        peaks, properties = signal.find_peaks(fft_maximums)
        max_peak = -1
        max_freq = 0

        for peak in peaks:
            if fft_maximums[peak] > max_freq:
                max_freq = fft_maximums[peak]
                max_peak = peak

        return freqs[max_peak] * 60
    
    def HeartRateMethod(self):
        freq_min = 1
        freq_max = 1.8
        fps=6
        if len(self.video):
            fft, frequencies = self.fft_filter(self.video, freq_min, freq_max, fps)
            heart_rate = self.find_heart_rate(fft, frequencies, freq_min, freq_max)
            return heart_rate
        else:
            return "No Face Found"
     


     
    
     






