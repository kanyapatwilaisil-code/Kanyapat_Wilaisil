# GUI ทดสอบวิดีโอการเดินด้วย TCN–F3

ใช้โมเดล final TCN–F3 ตัวเดียวอัตโนมัติ เลือกวิดีโอโดยไม่ระบุ label สกัด MediaPipe world pose วาด skeleton และบันทึกวิดีโอพร้อมผลทำนายลงในภาพจริง ใช้ PySide6 สำหรับหน้าจอ และ PyAV สำหรับอ่าน/เล่น/เขียนวิดีโอ ไม่ต้องเลือก model หรือฝึกใหม่

## ติดตั้งบน macOS

คัดลอก **ทั้งโฟลเดอร์ video_gui** ไปที่ Mac หรือแตกไฟล์ PD_Gait_GUI_macOS.zip ต้องมี model/ และ frozen_pdml/ อยู่ข้าง app.py ไม่ต้องนำ datasets หรือ artifacts ทั้งโครงการไปด้วย

แนะนำ Python 3.11 แบบ native ให้ตรงกับสถาปัตยกรรมเครื่อง ตรวจด้วย:

~~~bash
python3.11 -c "import platform; print(platform.python_version(), platform.machine())"
~~~

จาก Terminal เข้าโฟลเดอร์ video_gui แล้วสร้าง environment:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
~~~

**Apple Silicon (M1/M2/M3/M4 และรุ่นที่ใช้ arm64):**

~~~bash
python -m pip install -r requirements.txt
~~~

**Mac Intel (x86_64):**

~~~bash
python -m pip install -r requirements-macos-intel.txt
~~~

ไฟล์ Intel กำหนด PyTorch 2.2.2 และ NumPy ต่ำกว่า 2 เพื่อให้มี wheel ที่ตรงกับ Mac Intel; MediaPipe ใช้ 0.10.21 ตรวจรายการแพ็กเกจจาก [PyTorch 2.2.2](https://pypi.org/project/torch/2.2.2/#files) และ [MediaPipe 0.10.21](https://pypi.org/project/mediapipe/0.10.21/#files) หาก pip แจ้งว่าไม่มี distribution ให้ตรวจ Python/CPU/macOS ของเครื่อง ไม่ควรฝืนติดตั้ง wheel คนละสถาปัตยกรรม

ตรวจโมเดลและเปิดโปรแกรม:

~~~bash
python -c "from inference import FixedPredictor; p=FixedPredictor(); print('Model OK; reference error:', p.reference_max_error)"
python app.py
~~~

หลังติดตั้งครั้งแรก สามารถเปิด launch.command ได้ หาก permission ไม่ได้ติดมาจากไฟล์ ZIP ให้ใช้ chmod +x launch.command แล้วเปิด หรือรัน python app.py ผ่าน Terminal

ไม่ต้องติดตั้ง CUDA ใช้ CPU ทั้งการทำนาย TCN และ MediaPipe รุ่นนี้ไม่ได้สร้าง .app installer

## วิธีใช้

1. กด **เลือกวิดีโอ** ใช้คลิปที่เห็นร่างกายครบและมีคนเดียว ดูภาพต้นฉบับได้ด้วยปุ่มเล่น/พักและแถบเวลา
2. หากต้องการเปลี่ยนปลายทาง กด **เลือกโฟลเดอร์บันทึก** ค่าเริ่มต้นคือ ~/Movies/PD_Gait_Results
3. กด **เริ่มวิเคราะห์และบันทึก** โปรแกรมแสดง skeleton ระหว่างอ่านคลิป แล้วคำนวณผลทั้งคลิป ก่อนเขียนวิดีโอรอบที่สอง
4. เมื่อเสร็จ โปรแกรมเล่นวิดีโอผลลัพธ์และแสดงคะแนน/กราฟ/ข้อมูลคุณภาพ กด **เปิดผลลัพธ์** เพื่อเปิดโฟลเดอร์ใน Finder
5. ปุ่ม **หยุดวิเคราะห์** ยกเลิกงานได้ งานที่ยกเลิกหรือผิดพลาดมี status.json ระบุชัด และไม่นำเสนอว่าเป็นผลสำเร็จ

ทุกครั้งสร้างโฟลเดอร์ใหม่ตามเวลาและรหัสสุ่ม ไม่เขียนทับวิดีโอต้นฉบับ ไม่อ่าน label จากชื่อไฟล์ โฟลเดอร์ หรือ manifest เดิม

## สิ่งที่แสดงในวิดีโอ

- skeleton บนคนจาก image landmarks; features คำนวณจาก **world landmarks** เท่านั้น
- Clip prediction: ผลรวมทั้งคลิป และ PD score เฉลี่ยของหน้าต่างที่ผ่านเกณฑ์
- Current interval: ค่าเฉลี่ยคะแนนหน้าต่างที่ครอบคลุมเวลาปัจจุบัน เพื่อแสดงแนวโน้ม โดยไม่เปลี่ยนวิธีรวมผลทั้งคลิป
- เวลา จุดตัด 0.5 จำนวนหน้าต่างที่ใช้ได้ และข้อความระบุว่าเป็นการทดลองกับ MediaPipe
- หากช่วงใดไม่มีหน้าต่างที่ผ่านเกณฑ์ จะแสดง no valid window ไม่ลากคะแนนจากช่วงก่อนหน้าข้ามช่องว่าง
- หากทำนายทั้งคลิปไม่ได้ จะแสดง UNABLE TO ASSESS พร้อมสาเหตุ และยังบันทึกวิดีโอให้ตรวจ pose ได้

ข้อความในวิดีโอใช้ภาษาอังกฤษเพื่อให้อ่านได้บนเครื่องที่ไม่มีฟอนต์ไทย ส่วน GUI ใช้ภาษาไทย มีแถบข้อความเหนือภาพเพื่อไม่บังคน

วิดีโอผลลัพธ์เป็น **H.264 MP4, 30 FPS, ไม่มีเสียง** การเลือกภาพอิงเวลา PTS ของต้นฉบับ โดยค้างเฟรมล่าสุดจนถึงเฟรมถัดไป ระยะท้ายเฟรมสุดท้ายประมาณจาก median frame interval แล้วปัดระยะเวลารวมขึ้นตามความละเอียด 1/30 วินาที ภาพมีด้านยาวไม่เกิน 1280 px ก่อนเพิ่มแถบข้อความ และขนาดภาพออกเป็นเลขคู่สำหรับ H.264 หาก FPS ต้นฉบับสูงกว่า 30 บางภาพจะไม่ถูกส่งออก แต่การสกัด pose อ่านทุกเฟรม

## ไฟล์ที่บันทึกแต่ละรอบ

| ไฟล์ | เนื้อหา |
|---|---|
| annotated.mp4 | วิดีโอพร้อม skeleton และข้อความผลทำนายที่ฝังในภาพ เปิดด้วย QuickTime ได้ |
| prediction.json | ผลรวม ข้อมูลคุณภาพ preprocessing เวอร์ชันแพ็กเกจ hashes และข้อมูลแหล่งโมเดล |
| window_scores.csv | เวลาเริ่ม/จบ คะแนน คลาส และสัดส่วน interpolation ของหน้าต่างที่ผ่านเกณฑ์ |
| rejected_windows.csv | หน้าต่างที่ไม่ผ่านเกณฑ์และสาเหตุ |
| frame_quality.csv | เวลา จำนวนคนที่พบ และสถานะข้อต่อที่จำเป็นรายเฟรม |
| scores.png | กราฟคะแนนหน้าต่างเทียบเวลาและเส้น threshold |
| pose.npz | world pose ที่จับคู่ข้อต่อแล้ว เวลา image landmarks สำหรับวาด และข้อมูลคุณภาพ |
| features.npz | signals 22 ช่อง หน้าต่างที่ผ่านเกณฑ์ และชื่อ features |
| status.json | running, complete, cancelled หรือ failed |

ค่า NaN ใน NPZ หมายถึงข้อมูล pose/feature ที่ใช้ไม่ได้ ไม่ใช่ศูนย์ บางกรณีที่ตัดทั้งคลิปก่อนทำ features เช่นหลายคน จำนวน rejected windows อาจเป็นศูนย์ ให้ดูเหตุผลระดับคลิปและ frame quality ร่วมด้วย

## วิธีประมวลผลและขอบเขต

- โมเดลต้นทาง: 20260930T032616_final_bundle_02f2dc, TCN–F3, width 64, refit 9 epochs
- เก็บ weights และ scaler เป็น NumPy arrays โดยไม่มี pickle ในเส้นทางใช้งาน GUI ไม่เปลี่ยน weights และไม่ fit scaler ใหม่
- โครงสร้าง TCN ตรงกับโมเดลเดิม ใช้ CPU; แต่ละครั้งตรวจ hashes และทำนาย reference windows เทียบคะแนนต้นฉบับ หากไม่ตรงเกณฑ์จะหยุด
- frozen_pdml/features.py, protocol.py, common.py เป็นสำเนาจาก source snapshot ของ bundle พร้อมตรวจ hash
- ใช้การจับคู่ world landmarks จาก MediaPipe แบบเดียวกับ pdml/video.py เดิม พลิกแกน y และจัด 17 joint indices ให้สูตรคำนวณใช้ได้ **ไม่ใช่การแปลงเป็น SMPL หรือประมาณ h36m ที่เทียบเท่าข้อมูลฝึก**
- ตรวจเวลา PTS ปรับสัญญาณเป็น 30 Hz ใช้ window 45 samples, stride 22 และ smoothing/QC ตามโมเดลที่บันทึก
- MediaPipe กำหนดตรวจได้ถึงสองคน หากตรวจพบมากกว่าหนึ่งคนในเฟรมใด จะงดทำนายทั้งคลิป เพราะไม่มีการยืนยันอัตลักษณ์ข้ามคน; การตรวจไม่พบหลายคนไม่ใช่หลักฐานว่าไม่มีคนอื่นจริง
- ไม่มีคำตอบจริง จึง **ไม่คำนวณ Accuracy, F1, Sensitivity, Specificity หรือ AUC**
- คะแนนไม่ผ่าน calibration จึงไม่ใช่เปอร์เซ็นต์โอกาสเป็นโรค การทำนายผิดแม้คะแนนใกล้ 1 เกิดขึ้นได้ โดยเฉพาะเมื่อ MediaPipe ต่างจาก CARE-PD
- เป็น offline analysis ใช้ smoothing และสเกลจากทั้งคลิป ผลรวมที่แสดงตั้งแต่ต้นวิดีโอเป็นผลหลังอ่านคลิปครบ ไม่ใช่ผล real-time
- การอ่าน landmarks มีหน่วยความจำเพิ่มตามจำนวนเฟรม แนะนำเริ่มจากคลิปสั้น ใช้ Python 64-bit
- แหล่งต้นฉบับ/โมเดลเดิมของโครงการไม่ได้ถูกแก้ไข

## การตรวจสอบที่ทำแล้ว

ทดสอบบน Linux ใน workspace นี้ (ยัง **ไม่ได้ทดสอบบนเครื่อง macOS จริง**):

- เทียบ reference scores ของโมเดลพกพากับต้นฉบับ: maximum absolute error ประมาณ 1.25e-12
- automated tests 7 รายการผ่าน: mapping/visibility, reference scores, gaps, rotation, abstention/export, cancellation และคลิปสั้น
- วิดีโอจริงผ่าน MediaPipe → features → TCN → export MP4; ตรวจได้ 207 output frames ที่ 30 FPS สำหรับคลิปทดสอบประมาณ 6.9 วินาที
- ตรวจภาพ skeleton และข้อความจากเฟรมที่ decode กลับจาก MP4
- เปิด GUI แบบ offscreen เล่นวิดีโอผลลัพธ์และเลื่อนเวลาได้ ใช้ PyAV player เพื่อไม่ขึ้นกับ QtMultimedia backend ของเครื่อง
- การทดสอบเหล่านี้ยืนยันการทำงานซอฟต์แวร์ ไม่ใช่การประเมินความแม่นยำจำแนกโรค

รันทดสอบบน Mac เพิ่มได้:

~~~bash
python -m unittest discover -s tests -v
~~~

## ใช้ผ่าน command line

~~~bash
python backend.py "/path/to/video.mov" --output "~/Movies/PD_Gait_Results"
~~~

prepare_model.py เป็นเครื่องมือสำหรับผู้ดูแลเพื่อสร้างแพ็กจาก bundle ต้นทาง ไม่จำเป็นต้องรันบน Mac และต้องมี joblib/scikit-learn กับโครงการต้นทางจึงจะใช้ได้

## เอกสาร API อ้างอิง

- [MediaPipe Python setup](https://developers.google.com/edge/mediapipe/solutions/setup_python)
- [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [PyAV documentation](https://pyav.org/docs/stable/)
- [PySide6 documentation](https://doc.qt.io/qtforpython-6/)

MediaPipe task file ที่แนบเป็นไฟล์เดียวกับที่มีในโครงการเดิม เก็บ SHA-256 ไว้ใน model/manifest.json
