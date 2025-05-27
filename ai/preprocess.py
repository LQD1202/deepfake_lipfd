# import os
# import cv2
# import numpy as np
# import librosa
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from librosa import feature as audio


# N_EXTRACT = 10   # number of extracted images from video
# WINDOW_LEN = 5   # frames of each window
# MAX_SAMPLE = 100 

# audio_root = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/static/temp/wav"
# video_root = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/static/temp/video"
# output_root = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/static/preprocessing"
# root = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/static/temp/video"
# ############################################


# def get_spectrogram(audio_file):
#     data, sr = librosa.load(audio_file)
#     mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
#     plt.imsave("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/ai/temp_model/mel.png", mel)


# def run():
#     video_list = os.listdir(video_root)
#     for j in tqdm(range(len(video_list))):
#         v = video_list[j]
#         # load video
#         video_capture = cv2.VideoCapture(f"{root}/{v}")
#         fps = video_capture.get(cv2.CAP_PROP_FPS)
#         frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
#         print(frame_count)
#         # select 10 starting point from frames
#         frame_idx = np.linspace(
#             0,
#             frame_count - WINDOW_LEN - 1,
#             N_EXTRACT,
#             endpoint=True,
#             dtype=np.uint8,
#         ).tolist()
#         frame_idx.sort()
#         # selected frames
#         frame_sequence = [
#             i for num in frame_idx for i in range(num, num + WINDOW_LEN)
#         ]
#         frame_list = []
#         current_frame = 0
#         while current_frame <= frame_sequence[-1]:
#             ret, frame = video_capture.read()
#             if not ret:
#                 print(f"Error in reading frame {v}: {current_frame}")
#                 break
#             if current_frame in frame_sequence:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#                 frame_list.append(cv2.resize(frame, (500, 500)))  # to floating num
#             current_frame += 1
#         video_capture.release()

#         # load audio
#         name = v.split(".")[0]
#         a = f"{audio_root}/{name}.wav"

#         group = 0
#         get_spectrogram(a)
#         mel = plt.imread("/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/ai/temp_model/mel.png") * 255  # load spectrogram (int)
#         # print(mel)
#         mel = mel.astype(np.uint8)
#         print(mel)
#         mapping = mel.shape[1] / frame_count
#         for i in range(len(frame_list)):
#             idx = i % WINDOW_LEN
#             if idx == 0:
#                 try:
#                     begin = np.round(frame_sequence[i] * mapping)
#                     end = np.round((frame_sequence[i] + WINDOW_LEN) * mapping)
#                     sub_mel = cv2.resize(
#                         (mel[:, int(begin) : int(end)]), (500 * WINDOW_LEN, 500)
#                     )
#                     x = np.concatenate(frame_list[i : i + WINDOW_LEN], axis=1)
#                     # print(x.shape)
#                     # print(sub_mel.shape)
#                     x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)
#                     # print(x.shape)
#                     if output_root is None:
#                         output_root.mkdir(parents=True, exist_ok=True)
#                         plt.imsave(
#                             f"{output_root}/{name}_{group}.png", x
#                         )
#                     group = group + 1
#                 except ValueError:
#                     print(f"ValueError: {name}")
#                     continue
#     i += 1


# if __name__ == "__main__":
#     if not os.path.exists(output_root):
#         os.makedirs(output_root, exist_ok=True)
#     if not os.path.exists("./temp"):
#         os.makedirs("./temp", exist_ok=True)
#     run()
import os
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import feature as audio

N_EXTRACT = 100   # number of extracted images from video
WINDOW_LEN = 5   # frames of each window
MAX_SAMPLE = 100 

audio_root = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/static/temp/wav"
# video_root = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/static/temp/video"
output_root = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/static/preprocessing"
temp_mel_path = "./temp/mel.png"

def get_spectrogram(audio_file, output_path):
    data, sr = librosa.load(audio_file)
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    plt.imsave(output_path, mel)
    return mel.shape

def run(video_root):
    video_list = os.listdir(video_root)
    for j in tqdm(range(len(video_list))):
        v = video_list[j]
        name = os.path.splitext(v)[0]

        # Load video
        video_path = os.path.join(video_root, v)
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[{name}] Frame count: {frame_count}")

        # Frame index selection
        frame_idx = np.linspace(
            0,
            frame_count - WINDOW_LEN - 1,
            N_EXTRACT,
            endpoint=True,
            dtype=np.uint8,
        ).tolist()
        frame_idx.sort()
        frame_sequence = [i for num in frame_idx for i in range(num, num + WINDOW_LEN)]

        # Read selected frames
        frame_list = []
        current_frame = 0
        while current_frame <= frame_sequence[-1]:
            ret, frame = video_capture.read()
            if not ret:
                print(f"Error in reading frame {v}: {current_frame}")
                break
            if current_frame in frame_sequence:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame = cv2.resize(frame, (500, 500))
                frame_list.append(frame)
            current_frame += 1
        video_capture.release()

        # Load audio & mel spectrogram
        audio_name = os.listdir(audio_root)
        audio_path = os.path.join(audio_root, audio_name[0])
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

                        out_path = os.path.join(output_root, f"{name}_{group}.png")
                        cv2.imwrite(out_path, cv2.cvtColor(x, cv2.COLOR_RGBA2BGR))
                        group += 1
                    else:
                        print(f"[{name}] Skipped empty mel slice ({begin}-{end})")
                except Exception as e:
                    print(f"[{name}] Exception at frame {i}: {str(e)}")
                    continue

if __name__ == "__main__":
    os.makedirs(output_root, exist_ok=True)
    os.makedirs("./temp", exist_ok=True)
    run()
