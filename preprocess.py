import os
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import feature as audio

N_EXTRACT = 100   
WINDOW_LEN = 5  
MAX_SAMPLE = 100 

temp_mel_path = "./temp/mel.png"

def get_spectrogram(audio_file, output_path):
    data, sr = librosa.load(audio_file)
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    plt.imsave(output_path, mel)
    return mel.shape

def process_video(video_path: str, audio_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./temp", exist_ok=True)

    name = os.path.splitext(os.path.basename(video_path))[0]

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{name}] Frame count: {frame_count}")

    frame_idx = np.linspace(
        0,
        frame_count - WINDOW_LEN - 1,
        N_EXTRACT,
        endpoint=True,
        dtype=np.uint8,
    ).tolist()
    frame_idx.sort()
    frame_sequence = [i for num in frame_idx for i in range(num, num + WINDOW_LEN)]

    frame_list = []
    current_frame = 0
    while current_frame <= frame_sequence[-1]:
        ret, frame = video_capture.read()
        if not ret:
            print(f"Error in reading frame {name}: {current_frame}")
            break
        if current_frame in frame_sequence:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = cv2.resize(frame, (500, 500))
            frame_list.append(frame)
        current_frame += 1
    video_capture.release()

    # Spectrogram
    get_spectrogram(audio_path, temp_mel_path)
    mel = plt.imread(temp_mel_path) * 255
    mel = mel.astype(np.uint8)

    mapping = mel.shape[1] / frame_count
    group = 0

    for i in range(len(frame_list)):
        idx = i % WINDOW_LEN
        if idx == 0:
            try:
                begin = int(np.round(frame_sequence[i] * mapping))
                end = int(np.round((frame_sequence[i] + WINDOW_LEN) * mapping))
                begin = max(0, min(begin, mel.shape[1] - 1))
                end = max(begin + 1, min(end, mel.shape[1]))

                if end > begin:
                    sub_mel = mel[:, begin:end]
                    sub_mel = cv2.resize(sub_mel, (500 * WINDOW_LEN, 500))

                    x = np.concatenate(frame_list[i : i + WINDOW_LEN], axis=1)
                    x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)

                    out_path = os.path.join(output_dir, f"{name}_{group}.png")
                    cv2.imwrite(out_path, cv2.cvtColor(x, cv2.COLOR_RGBA2BGR))
                    group += 1
                else:
                    print(f"[{name}] Skipped empty mel slice ({begin}-{end})")
            except Exception as e:
                print(f"[{name}] Exception at frame {i}: {str(e)}")
                continue
