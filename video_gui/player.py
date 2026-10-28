"""PTS-paced PyAV player: avoids platform-specific Qt multimedia codecs."""
import time
from enum import Enum
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage

class VideoPlayer(QObject):
    class PlaybackState(Enum):
        StoppedState = 0
        PlayingState = 1
        PausedState = 2
    class Error(Enum):
        NoError = 0
        ResourceError = 1

    positionChanged = Signal(int)
    durationChanged = Signal(int)
    errorOccurred = Signal()
    imageReady = Signal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.tick)
        self.container = None
        self.pending = None
        self.position_ms = 0
        self.duration_ms = 0
        self.state = self.PlaybackState.StoppedState
        self.error_text = ""

    def setVideoOutput(self, widget):
        pass

    def setSource(self, url):
        self.stop()
        if self.container:
            self.container.close()
        self.container = None
        self.pending = None
        self.error_text = ""
        try:
            import av
            self.container = av.open(url.toLocalFile())
            self.stream = self.container.streams.video[0]
            self.frames = iter(self.container.decode(self.stream))
            self.pending = next(self.frames)
            self.origin = self.timestamp(self.pending)
            seconds = (float(self.stream.duration*self.stream.time_base) if self.stream.duration
                       else float(self.container.duration or 0)/av.time_base)
            self.duration_ms = max(0,round(seconds*1000))
            self.durationChanged.emit(self.duration_ms)
            self.position_ms = 0
            self.present(self.pending)
        except Exception as e:
            self.fail(e)

    @staticmethod
    def timestamp(frame):
        if frame.pts is None or frame.time_base is None:
            raise ValueError("Video has no presentation timestamps")
        return float(frame.pts*frame.time_base)

    def present(self, frame):
        import numpy as np
        # Match the orientation used for analysis and export.
        rgb = frame.to_ndarray(format="rgb24")
        angle = getattr(frame,"rotation",None)
        if angle is None:
            angle = -float(self.stream.metadata.get("rotate",0))
        rgb = np.ascontiguousarray(np.rot90(rgb,round(float(angle)/90)%4))
        h,w = rgb.shape[:2]
        self.imageReady.emit(QImage(rgb.data,w,h,rgb.strides[0],QImage.Format.Format_RGB888).copy())
        self.position_ms = max(0,round((self.timestamp(frame)-self.origin)*1000))
        self.positionChanged.emit(self.position_ms)

    def play(self):
        if not self.container or self.error_text:
            return
        if self.pending is None:
            self.setPosition(0)
        self.state = self.PlaybackState.PlayingState
        self.anchor = time.monotonic()-self.position_ms/1000
        self.timer.start(0)

    def pause(self):
        self.timer.stop()
        self.state = self.PlaybackState.PausedState

    def stop(self):
        self.timer.stop()
        self.state = self.PlaybackState.StoppedState

    def tick(self):
        if self.state != self.PlaybackState.PlayingState or self.pending is None:
            return
        try:
            self.present(self.pending)
            self.pending = next(self.frames,None)
            if self.pending is None:
                self.pause()
                return
            due = self.timestamp(self.pending)-self.origin
            self.timer.start(max(1,round((due-(time.monotonic()-self.anchor))*1000)))
        except Exception as e:
            self.fail(e)

    def setPosition(self, milliseconds):
        if not self.container:
            return
        was_playing = self.state == self.PlaybackState.PlayingState
        self.pause()
        try:
            seconds = max(0,milliseconds/1000)
            target = seconds+self.origin
            self.container.seek(int(target/self.stream.time_base),stream=self.stream,backward=True)
            self.frames = iter(self.container.decode(self.stream))
            self.pending = None
            for frame in self.frames:
                self.pending = frame
                if self.timestamp(frame) >= target-1e-7:
                    break
            if self.pending is not None:
                self.present(self.pending)
            if was_playing:
                self.play()
        except Exception as e:
            self.fail(e)

    def fail(self,error):
        self.stop()
        self.error_text = str(error)
        self.errorOccurred.emit()

    def error(self):
        return self.Error.ResourceError if self.error_text else self.Error.NoError

    def errorString(self):
        return self.error_text

    def duration(self):
        return self.duration_ms

    def playbackState(self):
        return self.state
