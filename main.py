import os
import cv2
from multiprocessing import Pool

from task_1 import process_one_image
from task_2 import process_one_video
from background_extraction import get_mean_background_image
from consts import frames_key, rectangles_key


def solve_task_1(root_dir, context_videos_dir, output_dir, visualize=False):
    output_dir = os.path.join(output_dir, "Task1")
    TASK_DIR = os.path.join(root_dir, "Task1")
    
    videos_paths = [os.path.join(context_videos_dir, filepath) for filepath in os.listdir(context_videos_dir)]

    with Pool(len(videos_paths)) as p:
        backgrounds = p.map(get_mean_background_image, videos_paths)

    
    images = [cv2.imread(os.path.join(TASK_DIR, filepath)) for filepath in os.listdir(TASK_DIR) if ".jpg" in filepath]
    queries_paths = [os.path.join(TASK_DIR, filepath) for filepath in os.listdir(TASK_DIR) if ".txt" in filepath]

    queries = list()
    for query_path in queries_paths:
        with open(query_path, "r") as f:
            file = f.read()
            
            file = file.split("\n")[1:]
            file = {int(elem):None for elem in file}

            queries.append(file)

    aux_bg = list()

    for query_path in queries_paths:
        id = int(os.path.basename(query_path).split(".")[0].split("_")[0]) - 1
        aux_bg.append(backgrounds[id])

    backgrounds = aux_bg

    workers = len(images) if not visualize else 1
    with Pool(workers) as p:
        results = p.starmap(process_one_image, zip(images, backgrounds, queries, [visualize for _ in images]))


    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for query_name, result in zip(queries_paths, results):
        query_name = os.path.basename(query_name).replace("query", "predicted")

        with open(os.path.join(output_dir, query_name), "w") as f:
            output_string = f"{len(result)}\n"

            for elem in result:
                output_string += f"{elem} {result[elem]}\n"
            
            output_string = output_string[:-1]

            f.write(output_string)
        


def solve_task_2(root_dir, context_videos_dir, output_dir, visualize=False):
    output_dir = os.path.join(output_dir, "Task2")
    TASK_DIR = os.path.join(root_dir, "Task2")
    
    videos_paths = [os.path.join(context_videos_dir, filepath) for filepath in os.listdir(context_videos_dir)]

    with Pool(len(videos_paths)) as p:
        backgrounds = p.map(get_mean_background_image, videos_paths)

    queries_paths = [os.path.join(TASK_DIR, filepath) for filepath in os.listdir(TASK_DIR) if ".txt" in filepath]


    queries = list()
    for query_path in queries_paths:
        with open(query_path, "r") as f:
            file = f.read()
            
            file = file.split("\n")
            file = {frames_key:int(file[0].split(" ")[0]), rectangles_key:[[int(elem) for elem in file[1].split(" ")[1:]]]}

            queries.append(file)

    videos_paths = [os.path.join(TASK_DIR, filepath) for filepath in os.listdir(TASK_DIR) if ".mp4" in filepath]
    workers = len(videos_paths) if not visualize else 1
    with Pool(workers) as p:
        results = p.starmap(process_one_video, zip(videos_paths, backgrounds, queries, [visualize for _ in videos_paths]))


    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for query_name, result in zip(queries_paths, results):
        query_name = os.path.basename(query_name).split(".")[0] + "_predicted.txt"

        with open(os.path.join(output_dir, query_name), "w") as f:
            output_string = f"{result[frames_key]} -1 -1 -1 -1\n"

            for idx, elem in enumerate(result[rectangles_key]):
                output_string += f"{idx} " + " ".join(elem) + "\n"

            output_dir = output_dir[:-1]
            f.write(output_string)

def solve_task_3():
    pass



def main():
    OUTPUT_DIR = "Ariton_Cosmin_506"
    ROOT_DIR = "train"
    CONTEXT_VIDEOS_DIR = os.path.join(ROOT_DIR, "context_videos_all_tasks")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    
    #solve_task_1(ROOT_DIR, CONTEXT_VIDEOS_DIR, OUTPUT_DIR, True)

    solve_task_2(ROOT_DIR, CONTEXT_VIDEOS_DIR, OUTPUT_DIR, False)

if __name__ == "__main__":
    main()
