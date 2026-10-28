#!/usr/bin/env python3
"""macOS desktop UI. Run: python app.py"""
from pathlib import Path
import multiprocessing as mp
import sys
import traceback
from queue import Empty
from PySide6.QtCore import Qt, QObject, QTimer, Signal, QUrl
from PySide6.QtGui import QImage, QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QProgressBar, QMessageBox, QSlider,
    QSplitter, QStackedWidget, QScrollArea)
from player import VideoPlayer as QMediaPlayer


def _run_analysis_process(video, output, cancel_event, result_queue):
    """รันใน multiprocessing.Process แยกต่างหาก (ไม่ใช่ QThread) เพราะ MediaPipe บน
    macOS ยังคงเรียก Metal (GPU) ในบางส่วนของกราฟเสมอ แม้ตั้ง delegate=CPU แล้วก็ตาม
    และ Metal ต้องถูกเริ่มต้นจาก "main thread ของโปรเซส" เท่านั้น การรันใน QThread
    (ซึ่งเป็นแค่ background thread ของโปรเซสเดิม) ทำให้ทั้งแอป crash แบบไม่มี exception
    ให้ดักจับได้เลย การใช้ process แยกทำให้ได้ main thread ใหม่ของมันเอง แก้ปัญหานี้ได้จริง
    """
    try:
        from backend import analyze, Cancelled

        def progress(value, text):
            result_queue.put(("progress", value, text))

        def preview(rgb):
            result_queue.put(("preview", rgb))

        folder, report = analyze(video, output, progress=progress, preview=preview,
                                 cancel=cancel_event.is_set)
        result_queue.put(("completed", str(folder), report))
    except Exception as e:
        if cancel_event.is_set():
            result_queue.put(("cancelled",))
        else:
            result_queue.put(("failed", str(e) + "\n\n" + traceback.format_exc()))


class AnalysisWorker(QObject):
    progress = Signal(int, str)
    preview = Signal(QImage)
    completed = Signal(str, object)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, video, output, parent=None):
        super().__init__(parent)
        self.video, self.output = video, output
        self.cancel_event = mp.Event()
        self.process = None
        self.result_queue = None
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._poll)

    def start(self):
        self.result_queue = mp.Queue()
        self.process = mp.Process(
            target=_run_analysis_process,
            args=(self.video, self.output, self.cancel_event, self.result_queue),
            daemon=True,
        )
        self.process.start()
        self.timer.start()

    def isRunning(self):
        return self.process is not None and self.process.is_alive()

    def _poll(self):
        try:
            while True:
                message = self.result_queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    _, value, text = message
                    self.progress.emit(value, text)
                elif kind == "preview":
                    _, rgb = message
                    h, w = rgb.shape[:2]
                    image = QImage(rgb.data, w, h, rgb.strides[0],
                                   QImage.Format.Format_RGB888).copy()
                    self.preview.emit(image)
                elif kind == "completed":
                    _, folder, report = message
                    self.completed.emit(folder, report)
                    self._finish()
                    return
                elif kind == "failed":
                    _, error = message
                    self.failed.emit(error)
                    self._finish()
                    return
                elif kind == "cancelled":
                    self.cancelled.emit()
                    self._finish()
                    return
        except Empty:
            pass
        if self.process is not None and not self.process.is_alive():
            # โปรเซสย่อยหยุดทำงานกะทันหันโดยไม่ส่งผลลัพธ์กลับมาเลย (native crash)
            self.failed.emit(
                "การประมวลผลล้มเหลว (โปรเซสย่อยหยุดทำงานกะทันหัน) กรุณาลองใหม่อีกครั้ง"
            )
            self._finish()

    def _finish(self):
        self.timer.stop()
        if self.process is not None:
            self.process.join(timeout=2)
        self.finished.emit()

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PD Gait — TCN-F3 Video Test")
        self.resize(1200, 850)
        self.video = None
        self.output_root = Path.home() / "Movies" / "PD_Gait_Results"
        self.last_output = None
        self.worker = None
        self.closing = False
        self.last_preview = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("ทดสอบวิดีโอการเดิน • TCN–F3")
        title.setStyleSheet("font-size: 23px; font-weight: bold;")
        layout.addWidget(title)
        warning = QLabel("การทดลองกับ MediaPipe: skeleton ต่างจาก CARE-PD ที่ใช้ฝึก • คะแนน PD ไม่ใช่โอกาสเป็นโรค • ใช้คลิปที่มีคนเดียว")
        warning.setWordWrap(True)
        warning.setStyleSheet("background:#fff3cd; color:#664d03; padding:10px;")
        layout.addWidget(warning)
        controls = QHBoxLayout()
        self.choose_button = QPushButton("เลือกวิดีโอ")
        self.run_button = QPushButton("เริ่มวิเคราะห์และบันทึก")
        self.run_button.setEnabled(False)
        self.stop_button = QPushButton("หยุดวิเคราะห์")
        self.stop_button.setEnabled(False)
        self.folder_button = QPushButton("เลือกโฟลเดอร์บันทึก")
        self.open_button = QPushButton("เปิดผลลัพธ์")
        self.open_button.setEnabled(False)
        for button in [self.choose_button,self.run_button,self.stop_button,self.folder_button,self.open_button]:
            controls.addWidget(button)
        layout.addLayout(controls)
        self.file_label = QLabel("ยังไม่ได้เลือกวิดีโอ")
        self.file_label.setWordWrap(True)
        self.output_label = QLabel(f"บันทึกอัตโนมัติ: {self.output_root}")
        self.output_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        layout.addWidget(self.output_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.stack = QStackedWidget()
        self.video_widget = QLabel()
        self.video_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_widget.setMinimumSize(400,300)
        self.video_widget.setStyleSheet("background:#111827;")
        self.preview_label = QLabel("เลือกวิดีโอเพื่อเริ่ม")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(400,300)
        self.preview_label.setStyleSheet("background:#111827;color:white;")
        self.stack.addWidget(self.video_widget)
        self.stack.addWidget(self.preview_label)
        self.stack.setCurrentWidget(self.preview_label)
        left_layout.addWidget(self.stack,1)
        player_controls = QHBoxLayout()
        self.play_button = QPushButton("เล่น / พัก")
        self.play_button.setEnabled(False)
        self.position = QSlider(Qt.Orientation.Horizontal)
        self.position.setRange(0,0)
        self.time_label = QLabel("0:00 / 0:00")
        player_controls.addWidget(self.play_button)
        player_controls.addWidget(self.position,1)
        player_controls.addWidget(self.time_label)
        left_layout.addLayout(player_controls)
        splitter.addWidget(left)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        self.result_label = QLabel("ผลการทำนาย: —")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-size:20px;font-weight:bold;padding:12px;")
        self.quality_label = QLabel("โมเดล: TCN–F3 (โหลดอัตโนมัติ)\nไม่มีการกรอก label ปกติ/PD")
        self.quality_label.setWordWrap(True)
        self.plot_label = QLabel("กราฟคะแนนจะแสดงหลังวิเคราะห์")
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_label.setMinimumSize(280,180)
        self.note_label = QLabel("เมื่อวิเคราะห์เสร็จ โปรแกรมจะบันทึก MP4 พร้อม skeleton และผลทำนาย รวมทั้ง CSV, JSON และกราฟโดยอัตโนมัติ\n\nวิดีโอผลลัพธ์ไม่มีเสียง และแสดงผลรวมจากทั้งคลิป จึงไม่ใช่การทำนายแบบ real-time")
        self.note_label.setWordWrap(True)
        for widget in [self.result_label,self.quality_label,self.plot_label,self.note_label]:
            side_layout.addWidget(widget)
        side_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        splitter.addWidget(scroll)
        splitter.setSizes([800,400])
        layout.addWidget(splitter,1)
        self.progress = QProgressBar()
        self.status_label = QLabel("พร้อมใช้งาน")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)

        self.player = QMediaPlayer(self)
        self.player.imageReady.connect(self.on_playback_image)
        self.player.positionChanged.connect(self.update_position)
        self.player.durationChanged.connect(lambda duration: self.position.setRange(0,duration))
        self.player.errorOccurred.connect(self.player_error)
        self.choose_button.clicked.connect(self.choose_video)
        self.folder_button.clicked.connect(self.choose_folder)
        self.run_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.play_button.clicked.connect(self.toggle_play)
        self.position.sliderReleased.connect(lambda: self.player.setPosition(self.position.value()))
        self.open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output))))

    def on_playback_image(self, image):
        self.video_widget.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_widget.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))

    def choose_video(self):
        filename, _ = QFileDialog.getOpenFileName(self,"เลือกวิดีโอ",str(Path.home()),
            "Video (*.mp4 *.mov *.m4v *.avi *.mkv *.webm);;All files (*)")
        if not filename:
            return
        self.video = Path(filename)
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(filename))
        self.stack.setCurrentWidget(self.video_widget)
        self.play_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.file_label.setText(str(self.video))
        self.result_label.setText("ผลการทำนาย: ยังไม่ได้วิเคราะห์คลิปนี้")
        self.quality_label.setText("TCN–F3 • ไม่มีข้อมูลคำตอบจริง")
        self.plot_label.clear()
        self.plot_label.setText("กราฟคะแนนจะแสดงหลังวิเคราะห์")
        self.last_output = None
        self.open_button.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("พร้อมวิเคราะห์")
        self.player.play()

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self,"โฟลเดอร์บันทึก",str(self.output_root.parent))
        if folder:
            self.output_root = Path(folder)
            self.output_label.setText(f"บันทึกอัตโนมัติ: {folder}")

    def start(self):
        if not self.video or (self.worker and self.worker.isRunning()):
            return
        self.player.stop()
        self.play_button.setEnabled(False)
        self.position.setEnabled(False)
        self.stack.setCurrentWidget(self.preview_label)
        self.preview_label.setText("กำลังโหลดโมเดล...")
        self.last_preview = None
        self.last_output = None
        self.open_button.setEnabled(False)
        self.plot_label.clear()
        self.result_label.setText("กำลังวิเคราะห์...")
        self.quality_label.setText("กรุณารอจนบันทึกวิดีโอเสร็จ")
        for button in [self.run_button,self.choose_button,self.folder_button]:
            button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setValue(0)
        self.worker = AnalysisWorker(self.video,self.output_root,self)
        self.worker.progress.connect(self.on_progress)
        self.worker.preview.connect(self.on_preview)
        self.worker.completed.connect(self.on_complete)
        self.worker.failed.connect(self.on_error)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel_event.set()
            self.stop_button.setEnabled(False)
            self.status_label.setText("กำลังหยุดและปิดไฟล์ที่กำลังเขียน...")

    def on_progress(self, value, text):
        self.progress.setValue(value)
        self.status_label.setText(text)

    def on_preview(self, image):
        self.last_preview = QPixmap.fromImage(image)
        self.preview_label.setPixmap(self.last_preview.scaled(
            self.preview_label.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.last_preview is not None:
            self.preview_label.setPixmap(self.last_preview.scaled(
                self.preview_label.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))

    def on_complete(self, folder, report):
        self.last_output = Path(folder)
        p,q = report["prediction"],report["quality"]
        if p["status"] == "ok":
            label = "กลุ่ม PD" if p["label"] else "กลุ่มปกติ"
            self.result_label.setText(f"ทำนายเป็น: {label}\nคะแนน PD: {p['score']:.3f}\nจุดตัด: {p['threshold']:.2f}")
        else:
            self.result_label.setText("ข้อมูลไม่เพียงพอสำหรับทำนาย\n" + p["reason"])
        self.quality_label.setText(
            f"พบคนเดียว: {q['detected_single_person_fraction']:.1%} ของเฟรม\n"
            f"ข้อต่อที่จำเป็นผ่านเกณฑ์: {q['required_joints_valid_fraction']:.1%}\n"
            f"ช่วงที่ทำนายได้: {p['windows']} | ไม่ผ่านเกณฑ์: {p['rejected_windows']}\n"
            f"เฟรมหลายคน: {q['multiple_person_frames']}\n"
            f"ความยาว: {q['duration_s']:.2f} วินาที\n"
            "Accuracy / F1: ไม่คำนวณ เพราะไม่มีคำตอบจริง")
        self.plot_label.setPixmap(QPixmap(str(self.last_output/"scores.png")).scaledToWidth(360,Qt.TransformationMode.SmoothTransformation))
        self.status_label.setText(f"บันทึกวิดีโอและผลแล้ว: {folder}")
        self.open_button.setEnabled(True)
        self.stack.setCurrentWidget(self.video_widget)
        self.player.setSource(QUrl.fromLocalFile(str(self.last_output/"annotated.mp4")))
        self.play_button.setEnabled(True)
        self.position.setEnabled(True)
        self.player.play()

    def on_error(self, error):
        self.result_label.setText("วิเคราะห์ไม่สำเร็จ")
        self.status_label.setText(error.splitlines()[0])
        message = QMessageBox(self)
        message.setWindowTitle("วิเคราะห์ไม่สำเร็จ")
        message.setText(error.splitlines()[0])
        message.setDetailedText(error)
        message.exec()

    def on_cancelled(self):
        self.result_label.setText("ยกเลิกการวิเคราะห์")
        self.status_label.setText("หยุดแล้ว — ไฟล์บางส่วนถูกระบุว่า cancelled และไม่ใช่ผลสำเร็จ")

    def on_finished(self):
        self.run_button.setEnabled(self.video is not None)
        self.choose_button.setEnabled(True)
        self.folder_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.closing:
            self.close()

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def update_position(self, value):
        if not self.position.isSliderDown():
            self.position.setValue(value)
        def clock(ms):
            seconds = ms // 1000
            return f"{seconds//60}:{seconds%60:02d}"
        self.time_label.setText(f"{clock(value)} / {clock(self.player.duration())}")

    def player_error(self,*args):
        if self.player.error() != QMediaPlayer.Error.NoError:
            self.status_label.setText("เล่นวิดีโอในหน้าจอไม่ได้: " + self.player.errorString() +
                                      " — ยังสามารถวิเคราะห์ไฟล์ หรือเปิด annotated.mp4 ด้วย QuickTime ได้")

    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self.closing = True
            self.stop()
            event.ignore()
        else:
            self.player.stop()
            event.accept()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PD Gait TCN-F3")
    window = Window()
    window.show()
    return app.exec()

if __name__ == "__main__":
    mp.freeze_support()  # กันปัญหา multiprocessing บน macOS/Windows
    sys.exit(main())
