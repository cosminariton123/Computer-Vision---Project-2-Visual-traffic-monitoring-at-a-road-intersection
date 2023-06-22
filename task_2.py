import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import uuid
import os

from consts import frames_key, rectangles_key
from grab_cut import grab_cut_from_image
from util import roirect_to_rect, rect_to_roirect

def get_gradient_loss(gradient, last_gradients, search_space):
    gradient = np.array(gradient)
    last_gradients = np.mean(np.array(last_gradients), axis=0)

    return np.mean(np.abs(gradient - last_gradients)) / (search_space * 2)


def compute_gradient(l_x, l_y, x, y):
    return l_x - x, l_y - y


def get_hist_loss(hists_1, hists_2):
    
    losses = list()

    for hist_1, hist_2 in zip(hists_1, hists_2):
        losses.append(cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_BHATTACHARYYA))

    return np.mean(np.array(losses))



def search_best_match(frame, histogram, last_rect, search_space, last_gradients):
    
    l_x, l_y, l_width, l_height  = rect_to_roirect(last_rect)

    matches = list()

    at_most_last_gradients = 8

    for x in range(l_x - search_space, l_x + search_space):
        for y in range(l_y - search_space, l_y + search_space):
            
            histogram_loss = get_hist_loss(get_histogram(frame[y:y+int(l_height), x:x+int(l_width)], None), histogram)
            
            aux_grad = last_gradients[at_most_last_gradients:] if len(last_gradients) > at_most_last_gradients else last_gradients

            gradient_loss = get_gradient_loss(compute_gradient(l_x, l_y, x, y), aux_grad, search_space)
            
            gradient_loss = gradient_loss / 50

            loss = gradient_loss + histogram_loss

            matches.append([[x, y], loss])

    matches.sort(key=lambda x: x[1])

    if matches[0][0][0] < l_x and matches[0][0][1] < l_y:
        l_width -= 0.4
        l_height -= 0.4

        if l_height < 60:
            l_height = 60

        if l_width < 60:
            l_width = 60

        if l_width / l_height > 2.5:
            l_width = l_height
    
    if matches[0][0][0] > l_x and matches[0][0][1] > l_y:
        l_width += 0.4
        l_height += 0.4

        if l_width / l_height > 2.5:
            l_width = l_height
        

    min_nr_last_gradients = 4
    min_frames = 100
    if len(last_gradients) > min_frames:
        last_gradients_mean = np.mean(np.array(last_gradients[min_nr_last_gradients:]), axis=0)

        if last_gradients_mean[0] <= 0 and matches[0][0][0] + l_width >= frame.shape[1]:
            return None, None
        
        if last_gradients_mean[0] >= 0 and matches[0][0][0] <= 0:
            return None, None

        if last_gradients_mean[1] <= 0 and matches[0][0][1] + l_height >= frame.shape[0]:
            return None, None

        if last_gradients_mean[1] >= 0 and matches[0][0][1] <= 0:
            return None, None

    return roirect_to_rect(matches[0][0] + [l_width, l_height]), compute_gradient(l_x, l_y, matches[0][0][0], matches[0][0][1])




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
            last_gradients = [(0, 0)]

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
            if last_rect is not None:
                last_rect, last_gradient = search_best_match(frame, histogram, last_rect, 20, last_gradients)
                last_gradients.append(last_gradient)

                if last_rect is not None:
                    curr_rect = [int(elem) for elem in last_rect]
                    query[rectangles_key].append(curr_rect)

            if visualise:
                if last_rect is not None:
                    frame = cv2.rectangle(frame, curr_rect[0:2], curr_rect[2:4], (255, 0, 255), 3)
                
                viz_video.write(frame)


    if visualise:
        viz_video.release()

    in_video.release()

    return query
