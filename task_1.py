import cv2
import numpy as np
import json

from background_extraction import get_differences_mask, extract_foreground_from_image


def process_one_image(image, background, query, visualize=False):

    THRESHOLD_BG_EXTRACT = 70
    THRESHOLD_OBJECT_PRESENCE = 25

    foreground_mask = get_differences_mask(image, background, threshold=THRESHOLD_BG_EXTRACT)
    foreground_mask = np.array(foreground_mask * 255, dtype=np.uint8)
    

    with open("manual_roi.json", "r") as f:
        rects = json.load(f)

    for lane in query:
        rect = rects[lane - 1]

        if np.mean(foreground_mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]) > THRESHOLD_OBJECT_PRESENCE:
            query[lane] = 1
        else:
            query[lane] = 0

    if visualize:
        cv2.imshow("image", image)
        cv2.imshow("background", background)

        for idx, lane in enumerate(query):
            rect = rects[lane - 1]

            cv2.imshow(f"foreground_patch_{idx}", foreground_mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])

            if query[lane] == 0:
                foreground_mask = cv2.rectangle(foreground_mask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)
            else:
                foreground_mask = cv2.rectangle(foreground_mask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

        cv2.imshow("foreground_mask", foreground_mask)
        cv2.waitKey()
        cv2.destroyAllWindows()


    return query
