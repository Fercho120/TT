"""Real time plotting of Microphone level using kivy
"""

from kivy.lang import Builder
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.garden.graph import MeshLinePlot
from kivy.clock import Clock
from threading import Thread
from kivy.config import Config
from kivy.properties import NumericProperty
import audioop
import pyaudio
import wave
Config.set('graphics', 'fullscreen', '0')  # Deactivates fullscreen


def get_microphone_level():

    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    global levels, s
    s = p.open(format=FORMAT,
               channels=CHANNELS,
               rate=RATE,
               input=True,
               frames_per_buffer=chunk)

    while True:
        data = s.read(chunk)
        mx = audioop.rms(data, 2)
        if len(levels) >= 100:
            levels = []
        levels.append(mx)


class Logic(BoxLayout):

    counter = NumericProperty(0)

    def __init__(self,):
        super(Logic, self).__init__()
        self.plot = MeshLinePlot(color=[1, 0, 0, 1])

    def start(self):
#        frames = []
#        for i in range(0, int(44100 / 1024 * 5)):
        self.ids.graph.add_plot(self.plot)
#            data = s.read(get_microphone_level.chunk)
#            frames.append(data)
        Clock.schedule_interval(self.get_value, 0.001)
#        return frames

    def stop(self):
        Clock.unschedule(self.get_value)
        print(self.counter)
        self.counter = 0
#        wf = wave.open("prueba.wav", 'wb')
#        wf.setnchannels(1)
#        wf.setampwidth(16)
#        wf.setframerate(44100)
#        wf.writeframes(b''.join(self.frames))
#        wf.close()

    def get_value(self, dt):
        self.plot.points = [(i, j/5) for i, j in enumerate(levels)]
        self.counter = self.counter + 1


class RealTimeMicrophone(App):
    def build(self):
        return Builder.load_file("look.kv")


if __name__ == "__main__":
    levels = []  # store levels of microphone
    get_level_thread = Thread(target=get_microphone_level)
    get_level_thread.daemon = True
    get_level_thread.start()
    RealTimeMicrophone().run()
