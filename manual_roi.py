import cv2
import json


def make_new_roi():
    image = cv2.imread("train/Task1/01_1.jpg")

    rects = list()

    for i in range(9):
        rect = cv2.selectROI("image", image, True)
        rects.append(rect)

    print(rects[0])

    for rect in rects:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 255), 3)
    

    with open("manual_roi.json", "w") as f:
        json.dump(rects, f)

    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()



def view_current_roi():
    image = cv2.imread("train/Task1/01_1.jpg")

    with open("manual_roi.json", "r") as f:
        rects = json.load(f)

    for rect in rects:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 255), 3)

    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def main():
    view_current_roi()
    
    



if __name__ == "__main__":
    main()

