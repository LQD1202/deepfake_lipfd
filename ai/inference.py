from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model (thay đường dẫn nếu khác)
model = YOLO("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/ai/yolov5-face/weights/yolov5n-0.5.pt")
print(f"✅ Model is running on: {model.device}")

# Hàm xử lý 1 frame (trả về ảnh có bounding box)
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    annotated_frame = results[0].plot()  # Vẽ bounding boxes lên ảnh
    return annotated_frame

# Hàm xử lý video
def process_video(input_path, output_path, model):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {input_path}")
        return

    # Lấy thông tin video gốc
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Khởi tạo VideoWriter để lưu video kết quả
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎥 Processing video: {input_path}")
    print(f"🔄 Total frames: {frame_count}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = process_frame(frame)
        out.write(annotated_frame)

        frame_idx += 1
        print(f"🧠 Processed frame {frame_idx}/{frame_count}", end='\r')

    cap.release()
    out.release()
    print(f"\n✅ Done! Output saved to: {output_path}")

# -----------------------------
# GỌI CHƯƠNG TRÌNH CHÍNH Ở ĐÂY
# -----------------------------
if __name__ == "__main__":
    # Đường dẫn video đầu vào & đầu ra (đổi nếu cần)
    input_video_path = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/static/temp/video/21aed5c69f6447a7a295a9f678f18961.mp4"
    output_video_path = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/static/temp/video/output_annotated.mp4"

    # Gọi xử lý
    process_video(input_video_path, output_video_path, model)
