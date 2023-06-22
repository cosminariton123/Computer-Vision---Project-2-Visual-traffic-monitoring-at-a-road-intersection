import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import uuid
import os

from consts import frames_key, rectangles_key
from grab_cut import grab_cut_from_image
from util import roirect_to_rect, rect_to_roirect


def get_hist_loss(hists_1, hists_2):
    
    losses = list()

    for hist_1, hist_2 in zip(hists_1, hists_2):
        losses.append(cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_BHATTACHARYYA))

    return np.mean(np.array(losses))



def search_best_match(frame, histogram, last_rect, search_space):
    
    l_x, l_y, l_width, l_height  = rect_to_roirect(last_rect)

    matches = list()

    for x in range(l_x - search_space, l_x + search_space):
        for y in range(l_y - search_space, l_y + search_space):
            
            loss = get_hist_loss(get_histogram(frame[y:y+l_height, x:x+l_width], None), histogram)
            matches.append([[x, y], loss])

    matches.sort(key=lambda x: x[1])

    return roirect_to_rect(matches[0][0] + [l_width, l_height])




def get_histogram(frame, mask):
    if mask is not None:
        mask = np.reshape(mask, newshape=frame.shape[:2])

    hists = list()

    for channel in range(3):
        hist = cv2.calcHist([frame], [channel], mask, [256], [0, 256])
        hists.append(hist)

    return hists


def process_one_video(video_path, query, visualise=False):


    in_video = cv2.VideoCapture(video_path)

    if visualise:
        id = str(uuid.uuid1())

        if not os.path.exists("parameter_tuning"):
            os.mkdir("parameter_tuning")

        visualise_dir = os.path.join("parameter_tuning", id)

        os.mkdir(visualise_dir)

        fps = int(in_video.get(cv2.CAP_PROP_FPS))
        width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        viz_video = cv2.VideoWriter(os.path.join(visualise_dir, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames = query[frames_key]

    for i in tqdm(range(frames)):
        
        ret, frame = in_video.read()

        if not ret:
            print("Hello?")
            break

        if i == 0:
            foreground, mask = grab_cut_from_image(frame, rect_to_roirect(query[rectangles_key][0]))

            if visualise:
                mask_viz = mask.copy()
                mask_viz = mask_viz * 255

                cv2.imwrite(os.path.join(visualise_dir, "foreground.jpg"), foreground)
                cv2.imwrite(os.path.join(visualise_dir, "mask.jpg"), mask_viz)

            histogram = get_histogram(foreground, mask)
            last_rect = query[rectangles_key][0]

            if visualise:
                

                color = ["blue", "green", "red"]
                for idx, hist in enumerate(histogram):
                    plt.subplot(4, 1, idx + 1)
                    plt.plot(hist, color=color[idx])
                    plt.title(f"Hist {color[idx]}")

                for idx, hist in enumerate(histogram):
                    plt.subplot(4, 1, 4)
                    plt.plot(hist, color=color[idx])
                    plt.title(f"Hists combined")
                plt.savefig(os.path.join(visualise_dir, "hist.jpg"))

        else:
            last_rect = search_best_match(frame, histogram, last_rect, 10)
            query[rectangles_key].append(last_rect)

            if visualise:
                frame = cv2.rectangle(frame, last_rect[0:2], last_rect[2:4], (255, 0, 255), 3)
                
                viz_video.write(frame)


    if visualise:
        viz_video.release()

    in_video.release()

    return query
