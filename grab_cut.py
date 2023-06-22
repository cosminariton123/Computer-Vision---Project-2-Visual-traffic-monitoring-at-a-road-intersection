import cv2
import numpy as np


def grab_cut_from_image(image, roirect):

    bgdModel = np.zeros(shape=(1, 65), dtype=np.float64)
    fgdModel = np.zeros(shape=(1, 65), dtype=np.float64)
    rect = roirect
    mask, background_model, foreground_model = cv2.grabCut(image, mask=None, rect=rect, mode=cv2.GC_INIT_WITH_RECT, bgdModel=bgdModel, fgdModel=fgdModel, 
                                                            iterCount=1)


    foreground = np.array(np.reshape(np.array(mask == cv2.GC_PR_FGD, dtype=np.uint8), newshape=list(image.shape)[:-1] + [1]) * image, dtype=np.uint8)
    mask = np.reshape(np.array(mask == cv2.GC_PR_FGD, dtype=np.uint8), newshape=list(image.shape)[:-1] + [1])

    foreground = foreground[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    mask = mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

    return foreground, mask