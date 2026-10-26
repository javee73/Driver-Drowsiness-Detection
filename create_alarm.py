
import wave
import math
import struct

sample_rate = 44100
duration = 3
frequency = 1000

with wave.open("alarm.wav", "w") as alarm:
    alarm.setnchannels(1)
    alarm.setsampwidth(2)
    alarm.setframerate(sample_rate)

    for i in range(sample_rate * duration):
        # Repeating beep pattern
        t = i / sample_rate
        beep = (t % 1.0) < 0.5

        if beep:
            value = int(12000 * math.sin(2 * math.pi * frequency * t))
        else:
            value = 0

        alarm.writeframesraw(struct.pack("<h", value))

print("alarm.wav created successfully!")
