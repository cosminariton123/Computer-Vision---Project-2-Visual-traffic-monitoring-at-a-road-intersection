import numpy as np
import cv2
from tqdm import tqdm


def get_differences_mask(image, background, threshold=50):

    mask = image.astype(np.int32) - background.astype(np.int32)
    mask = np.array([[np.abs(elem) for elem in line] for line in mask], np.uint8)
    
    mask = np.array([[[0 for _ in elem] if np.mean(elem) < threshold else [1 for _ in elem] for elem in line] for line in mask], dtype=np.uint8)

    return mask


def extract_foreground_from_image(image, background, threshold = 50):
    assert image.shape == background.shape, f"Shape missmatch. Image shape is: {image.shape}. Background shape is: {background.shape}"

    mask = get_differences_mask(image, background, threshold)

    result = image * mask
    result = result.astype(np.uint8)
    
    return result




def get_mean_background_image(video_path):

    in_video = cv2.VideoCapture(video_path)

    mean_image = None

    frames = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frames), desc="Processing mean image"):
        ret, frame = in_video.read()

        if not ret:
            print("Hello?")
            break

        if mean_image is None:
            mean_image = np.zeros(shape=frame.shape)

        mean_image += frame

    in_video.release()

    return np.array(mean_image / frames, dtype=np.uint8)