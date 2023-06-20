import cv2
import numpy as np
import os
from tqdm import tqdm


def get_median_background_image():
    
    video_path = os.path.join("train", "context_videos_all_tasks", "01.mp4")

    in_video = cv2.VideoCapture(video_path)

    r_median_bins = np.zeros(shape=(880, 1920, 256))
    g_median_bins = np.zeros(shape=(880, 1920, 256))
    b_median_bins = np.zeros(shape=(880, 1920, 256))

    frames = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(desc="Processing median image", total=100)

    for _ in range(frames) :
        ret, frame = in_video.read()

        if not ret:
            break

        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]

        for i, line in enumerate(b):
            for j, elem in enumerate(line):
                b_median_bins[i, j, elem] += 1

        for i, line in enumerate(g):
            for j, elem in enumerate(line):
                g_median_bins[i, j, elem] += 1
        
        for i, line in enumerate(r):
            for j, elem in enumerate(line):
                g_median_bins[i, j, elem] += 1

        pbar.update(1)

    in_video.release()


    median_image = np.concatenate([np.argmax(b_median_bins, axis=2), np.argmax(g_median_bins, axis=2), np.argmax(r_median_bins, axis=2)], axis=2, dtype=np.uint8)

    cv2.imshow("median_image", median_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

get_median_background_image()



def get_mean_background_image():
    video_path = os.path.join("train", "context_videos_all_tasks", "02.mp4")

    in_video = cv2.VideoCapture(video_path)

    mean_image = np.zeros(shape=(880, 1920, 3))

    frames = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(desc="Processing mean image", total=frames)

    for _ in range(frames) :#True:
        ret, frame = in_video.read()

        if not ret:
            break

        mean_image += frame

        pbar.update(1)

    in_video.release()


    cv2.imshow("b", np.array(mean_image / frames, dtype=np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()



def process_one_image(image):
    pass

