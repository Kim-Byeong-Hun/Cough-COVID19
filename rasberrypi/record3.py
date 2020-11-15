import pyaudio

import wave

import RPi.GPIO as GPIO
import time 

FORMAT = pyaudio.paInt16

CHANNELS = 1 #마이크 채널 설정

RATE = 44100 #마이크 RATE값  44.1KHz

CHUNK = 12000 #버퍼링의 프레임 수

RECORD_SECONDS = 5 #녹음 기록시간 

WAVE_OUTPUT_FILENAME = "file.wav" #저장할 파일이름

 

audio = pyaudio.PyAudio()

 

# start Recording

stream = audio.open(format=pyaudio.paInt16, 

                channels=CHANNELS, 

                rate=RATE, 

                input=True, 

                input_device_index=0,

                frames_per_buffer=CHUNK)

print ("recording...")

frames = []

 

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

    data = stream.read(CHUNK)

    frames.append(data)

print ("finished recording")

 

 

# stop Recording

stream.stop_stream()

stream.close()

audio.terminate()

 

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')

waveFile.setnchannels(CHANNELS)

waveFile.setsampwidth(audio.get_sample_size(FORMAT))

waveFile.setframerate(RATE)

waveFile.writeframes(b''.join(frames))

waveFile.close()


from result_wav import final_result

result_f = final_result("file.wav")

print(result_f)


GPIO.setmode(GPIO.BCM) #GPIO 제어로 설정

RUNNING = True
green=27 #GPIO 번호
red=17
blue=22

GPIO.setup(red, GPIO.OUT) #출력모드 설정
GPIO.setup(green, GPIO.OUT)
GPIO.setup(blue, GPIO.OUT)


try:
    
        
        if result_f == 0 :
            
            GPIO.output(red, GPIO.LOW)
            GPIO.output(green, GPIO.HIGH)
            GPIO.output(blue, GPIO.HIGH)

        elif result_f == 1 :
            
            GPIO.output(red, GPIO.HIGH)
            GPIO.output(green, GPIO.LOW)
            GPIO.output(blue, GPIO.HIGH)
        
        elif result_f ==2 :

            GPIO.output(red, GPIO.LOW)
            GPIO.output(green, GPIO.LOW)
            GPIO.output(blue, GPIO.HIGH)
   
            
except KeyboardInterrupt:
    
    GPIO.cleanup()
