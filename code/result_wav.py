from final import classify_cough
import librosa
from final import cutting_wave
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def final_result(filename):
    # test
    file_name = filename
    #write_name = '../cut_wave/00eead73-a2c2-480d-b2ee-d79b1a526871_exam.wav'
    #cutting_wave(file_name, write_name)
    model = load_model('../model/model_exam.h5')
    image = classify_cough(file_name, model)[0]

    
    return(image)

