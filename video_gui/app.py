#!/usr/bin/env python3
"""macOS desktop UI. Run: python app.py"""
from pathlib import Path
import sys
import threading
import traceback
from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QImage, QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QProgressBar, QMessageBox, QSlider,
    QSplitter, QStackedWidget, QScrollArea)
from player import VideoPlayer as QMediaPlayer

class AnalysisWorker(QThread):
    progress = Signal(int, str)
    preview = Signal(QImage)
    completed = Signal(str, object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, source, output, mode="video", parent=None):
        super().__init__(parent)
        self.source, self.output, self.mode = source, output, mode
        self.cancel_event = threading.Event()

    def run(self):
        try:
            from backend import analyze, Cancelled
            def preview(rgb):
                h, w = rgb.shape[:2]
                self.preview.emit(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy())
            if self.mode == "pose":
                from backend import analyze_pose
                folder, report = analyze_pose(self.source, self.output,
                                              progress=self.progress.emit,
                                              cancel=self.cancel_event.is_set)
            else:
                folder, report = analyze(self.source, self.output,
                                         progress=self.progress.emit, preview=preview,
                                         cancel=self.cancel_event.is_set)
            self.completed.emit(str(folder), report)
        except Exception as e:
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(e) + "\n\n" + traceback.format_exc())

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PD Gait — TCN-F3 Video Test")
        self.resize(1200, 850)
        self.video = None
        self.pose = None
        self.input_mode = None
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
        self.pose_button = QPushButton("เลือก pose.npz")
        self.run_button = QPushButton("เริ่มวิเคราะห์และบันทึก")
        self.run_button.setEnabled(False)
        self.stop_button = QPushButton("หยุดวิเคราะห์")
        self.stop_button.setEnabled(False)
        self.folder_button = QPushButton("เลือกโฟลเดอร์บันทึก")
        self.open_button = QPushButton("เปิดผลลัพธ์")
        self.open_button.setEnabled(False)
        for button in [self.choose_button,self.pose_button,self.run_button,self.stop_button,self.folder_button,self.open_button]:
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
        self.pose_button.clicked.connect(self.choose_pose)
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
        self.pose = None
        self.input_mode = "video"
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

    def choose_pose(self):
        filename, _ = QFileDialog.getOpenFileName(self, "เลือก pose.npz", str(Path.home()),
            "Pose NPZ (*.npz);;All files (*)")
        if not filename:
            return
        self.pose = Path(filename)
        self.video = None
        self.input_mode = "pose"
        self.player.stop()
        self.play_button.setEnabled(False)
        self.position.setEnabled(False)
        self.stack.setCurrentWidget(self.preview_label)
        self.preview_label.setText("เลือก pose.npz แล้ว — พร้อมวิเคราะห์")
        self.file_label.setText(f"Pose: {self.pose}")
        self.result_label.setText("ผลการทำนาย: ยังไม่ได้วิเคราะห์ pose นี้")
        self.quality_label.setText("TCN–F3 • ใช้ pose ที่สกัดไว้แล้ว • ไม่รัน MediaPipe ซ้ำ")
        self.plot_label.clear()
        self.plot_label.setText("กราฟคะแนนจะแสดงหลังวิเคราะห์")
        self.last_output = None
        self.open_button.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("พร้อมวิเคราะห์ pose.npz")
        self.run_button.setEnabled(True)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self,"โฟลเดอร์บันทึก",str(self.output_root.parent))
        if folder:
            self.output_root = Path(folder)
            self.output_label.setText(f"บันทึกอัตโนมัติ: {folder}")

    def start(self):
        if not (self.video or self.pose) or (self.worker and self.worker.isRunning()):
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
        for button in [self.run_button,self.choose_button,self.pose_button,self.folder_button]:
            button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setValue(0)
        source = self.pose if self.input_mode == "pose" else self.video
        self.worker = AnalysisWorker(source,self.output_root,self.input_mode,self)
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
        self.status_label.setText(f"บันทึกผลแล้ว: {folder}")
        self.open_button.setEnabled(True)
        if report.get("input_type") == "pose_npz":
            self.stack.setCurrentWidget(self.preview_label)
            self.preview_label.setText("วิเคราะห์จาก pose.npz เสร็จแล้ว\nเปิดโฟลเดอร์ผลลัพธ์เพื่อดู scores.png / CSV / JSON")
            self.play_button.setEnabled(False)
            self.position.setEnabled(False)
        else:
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
        self.run_button.setEnabled(self.video is not None or self.pose is not None)
        self.choose_button.setEnabled(True)
        self.pose_button.setEnabled(True)
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
    sys.exit(main())
