import argparse
import os
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
# from nbtlib import *
from tqdm import tqdm

def create_bar(name: str, max: int):
    return tqdm(total=max, ncols=100, bar_format='{desc} |{bar}| {n_fmt}/{total_fmt}; ETA: {remaining}; {elapsed}', desc=name)

texture_path = 'textures'
color_map = {}

p_endings = ["_top.png", "_bottom.png"]

def sort_textures():
    files = os.listdir(texture_path)
    if not os.path.exists("sorted"):
        os.mkdir("sorted")
    def has_full_alpha(img):
        return any(pixel[3] < 255 for row in img for pixel in row)
    bar = create_bar('Sorting', len(files))
    for f in files:
        bar.update()
        valid = True
        for e in p_endings:
            if f.endswith(e):
                valid = False
                break
        if not valid or not f.endswith(".png"):
            continue
        path = os.path.join(texture_path, f)
        img: np.ndarray = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img.shape == (16, 16, 4) and not has_full_alpha(img):
            shutil.copyfile(path, "sorted"+os.sep+f)

def load_textures():
    if not os.path.exists("sorted"):
        sort_textures()
        print("")
    files = os.listdir("sorted")
    bar = create_bar('Loading textures:', len(files))

    def setbarp(text: str):
        bar.set_description(text)

    for texture_file in files:
        bar.update()
        # setbarp(texture_file.split('.')[0])
        texture = cv2.imread(os.path.join("sorted", texture_file))
        avg_color: np.ndarray = np.mean(texture, axis=(0, 1)) # blue green red
        rgb = (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))
        bgr = (rgb[2], rgb[1], rgb[0])
        color_map[bgr] = texture_file
    bar.close()

# if not os.path.exists("avg"):
#     os.mkdir("avg")
# def save_avg(col: tuple[int, int, int], fn: str):
#     image = np.zeros((16, 16, 3), dtype=np.uint8)
#     image[:] = col
#     cv2.imwrite("avg"+os.sep+fn, image)


# print("")

def euclidean_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

def find_nearest_color(target_color, color_list) -> tuple[int, int, int]:
    nearest_color = min(color_list, key=lambda color: euclidean_distance(color, target_color))
    return nearest_color

# SIZE = (70, 50)

# def save_frame(frame):
#     image = np.zeros((SIZE[0], SIZE[1], 3), dtype=np.uint8)

def toIntTuple(nptuple):
    ls = []
    for i in nptuple:
        ls.append(int(i))
    return tuple(ls)

colorcache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
frame_bars: list[tqdm] = []

def preload_sorted():
    ret: dict[str, np.ndarray] = {}
    ls = os.listdir("sorted")
    for t in ls:
        img = cv2.imread("sorted"+os.sep+t)
        ret[t] = img
    return ret


def process_frame(sorted_textures: dict[str, np.ndarray], 
                  image: np.ndarray, index: int = 0,
                  block_size: tuple[int, int] = (70, 50), 
                  output_name: str = None):
    # Resize frame
    resized_frame = cv2.resize(image, block_size)
    # print(f"{len(resized_frame)}, {len(resized_frame[0])}")
    # ret = np.zeros((SIZE[0], SIZE[1], 3), dtype=np.uint8)
    bar = create_bar(f'Frame {index}', block_size[1]*block_size[0])
    frame_bars.append(bar)
    # bar = IncrementalBar(f'Frame {index}', max = block_size[1]*block_size[0], suffix='%(index)d/%(max)d; ETA: %(eta_td)s; %(elapsed_td)s')
    blocked = np.zeros((block_size[1]*16, block_size[0]*16, 3), dtype=np.uint8) # 800 1120 3
    for y, row in enumerate(resized_frame):
        for x, pixel in enumerate(row):
            pixel: np.ndarray[np.uint8, np.uint8, np.uint8]
            tpixel = toIntTuple(tuple(pixel))
            if tpixel in colorcache:
                nearest_col = colorcache[tpixel]
            else:
                nearest_col = find_nearest_color(pixel, color_map.keys())
                colorcache[tpixel] = nearest_col
            # ret[y, x] = nearest_col
            texture = color_map[nearest_col]
            block: np.ndarray = sorted_textures[texture] # 16 16 3
            blocked[y*16:y*16 + 16, x*16:x*16 + 16] = block
            bar.update()
    # cv2.imwrite("res"+os.sep+f"f_{index}.png", ret)
    if output_name is None:
        cv2.imwrite("res"+os.sep+f"{index}.png", blocked)
    else:
        cv2.imwrite("res"+os.sep+output_name, blocked)
    # save_blocked(index, processed_frame)



def make_video(fps: float, block_size: tuple[int, int], use_ffmpeg = True, output_name: str = "output.avi"):
    if use_ffmpeg:
        path = "res"+os.sep+output_name
        subprocess.run(["ffmpeg", "-framerate", str(fps), "-i", "res"+os.sep+"%d.png", path], stdout=subprocess.DEVNULL)
        print("Created using FFMPEG "+output_name.split(".")[-1].upper()+" video: "+path)
        return
    size = tuple(element * 16 for element in block_size)
    writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'),fps, tuple(reversed(size)))
    frames = []
    for frame_file in sorted(os.listdir("res")):
        if frame_file.endswith('.png'):
            frames.append(frame_file)
    bar = create_bar(f'Making video', len(frames))
    for frame in frames:
        frame_path = os.path.join("res", frame)
        frame = cv2.imread(frame_path)
        writer.write(frame)
        bar.update()
    writer.release()

def blockize_image(path: str = "input.png", block_size: tuple[int, int] = (70, 50)):
    img: np.ndarray = cv2.imread(path)
    print(f"Processing image \"{path}\"; Size: {img.shape[0]}x{img.shape[1]} ({block_size[0]}x{block_size[1]} blocks)")
    out_name = ".".join(path.split(".")[:-1])+f"_{block_size[0]}x{block_size[1]}_output.png"
    textures = preload_sorted()
    process_frame(textures, img, -1, block_size, out_name)

class FrameRange:
    def __init__(self, f:int = 0, t:int = -1) -> None:
        if t != -1:
            if f > t:
                raise ValueError("Invalid range")
        self.f = f
        self.t = t
    
    def inrange(self, n: int) -> bool:
        if self.t == -1:
            return n >= self.f
        else:
            return n >= self.f and n <= self.t
    
    def size(self, total: int) -> int:
        if self.t == -1:
            return total-self.f+1
        else:
            return self.t-self.f+1
    
    @staticmethod
    def parse(text: str):
        if "-" in text:
            raise ValueError("Must be positive")
        parts = text.split('..')
        if len(parts) == 1:
            if parts[0].startswith('..'):
                return FrameRange(t=int(parts[0][2:]))
            elif parts[0].endswith('..'):
                return FrameRange(f=int(parts[0][:-2]))
            else:
                return FrameRange(f=int(parts[0]))
        elif len(parts) == 2:
            if parts[0] and parts[1]:
                return FrameRange(f=int(parts[0]), t=int(parts[1]))
            elif parts[0]:
                return FrameRange(f=int(parts[0]))
            elif parts[1]:
                return FrameRange(t=int(parts[1]))
        else:
            raise ValueError("Invalid input format")

def savecache():
    with open("cache.txt", "w") as f:
        for k, v in colorcache.items():
        #    print(f"K {type(k)} {type(k[0])}")
        #    print(f"V {type(v)} {type(v[0])}")
           l = f"{' '.join(str(int(it)) for it in k)}:{' '.join(str(it) for it in k)}"
           f.write(l+"\n")

def loadcache():
    if not os.path.exists("cache.txt"):
        return
    return
    with open("cache.txt") as f:
        lines = f.read().split("\n")
        for l in lines:
            if l == "":
                continue
            kv = l.split(":")
            k = []
            for i in kv[0].split(" "): 
                k.append(int(i))
            v = []
            for i in kv[1].split(" "):
                v.append(int(i))
            colorcache[tuple(k)] = tuple(v)

def writedebug(text: str):
    with open("debug.txt", "a") as f:
        f.write(text+"\n")

def blockize_video(path: str = "input.mp4", block_size: tuple[int, int] = (70, 50), frame_range = FrameRange(), output_ext: str = "avi"):
    # Load video and extract frames
    video_path = path
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    result_frame_count = frame_range.size(frame_count-1)
    if frame_range.f == 0 and frame_range.t == -1:
        result_fps = fps
    else:
        result_fps = (result_frame_count * fps) / frame_count
    print(f"Processing video \"{path}\"; FPS {result_fps}; Total frames: {int(result_frame_count)}|{int(frame_count)}")
    sorted_textures = preload_sorted()
    threads: list[threading.Thread] = []
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # t = threading.Thread(name=f"F{index}", target=blockize_image, args=(frame, index))
        # t.start()
        if os.path.exists("res"+os.sep+f"{index}.png"):
            index += 1
            continue
        if not frame_range.inrange(index):
            writedebug(f"not in range {index}")
            index += 1
            continue
        if len(threads) >= thread_count:
            while any(t.is_alive() for t in threads):
                time.sleep(0.001)
            threads.clear()
            for bar in frame_bars:
                bar.close()
            frame_bars.clear()
        t = threading.Thread(name=f"Frame {index}", target=process_frame, args=(sorted_textures, frame, index, block_size))
        t.start()
        threads.append(t)
        # process_frame(frame, index, block_size)
        # print("")
        index += 1
    cap.release()
    while any(t.is_alive() for t in threads):
        time.sleep(0.001)
    threads.clear()
    for bar in frame_bars:
        bar.close()
    frame_bars.clear()
    # print("")
    out_name = ".".join(path.split(".")[:-1])+"_output."+output_ext
    make_video(result_fps, block_size, output_name=out_name)

thread_count = 5

def main():
    global thread_count
    parser = argparse.ArgumentParser(description="Convert image or video to minecraft blocks image")
    parser.add_argument("--input", "-i", type=str, default="input.mp4", help="Input file name (default: input.mp4). Must be .mp4 or .png")
    parser.add_argument("--size", "-s", type=str, help="Specify size as a string in format {width}x{height}")
    parser.add_argument("--range", "-r", default="0..", type=str, help="Specify frames indexes to process from video in format {from}..{to} (all inclusive)(default is 0..). Examples: ..5 (from 0 to five), 3.. (from 3 to end), 2..6 (from 2 to 6)")
    parser.add_argument("--ext", "-e", choices=["avi", "mp4", "gif"], default="avi", help="Specify output video extension")
    parser.add_argument("--threads", "-tc", type=int, help="Thread count", default=5)

    args: argparse.Namespace = parser.parse_args()
    thread_count = args.threads

    path: str = args.input
    if not os.path.exists(path):
        print("Unknown file "+path)
        return
    size: str = args.size
    if size != None:
        if "x" not in size or len(size) < 3 or size.count("x") > 1:
            print(f"\"{size}\" is not valid size")
            parser.print_help()
            return
    if size != None:
        try:
            ls = size.split("x")
            w = int(ls[0])
            h = int(ls[1])
            content_size = (w, h)
        except:
            print(f"\"{size}\" is not valid size")
            parser.print_help()
            return

    input_type = -1
    if path.endswith(".png"):
        input_type = 1
        if size == None:
            img: np.ndarray = cv2.imread(path)
            content_size: tuple[int, int] = img.shape[:2]
            del img
    elif path.endswith(".mp4"):
        input_type = 2
        if size == None:
            vid = cv2.VideoCapture(path)
            w: int = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            h: int = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            vid.release()
            content_size = (w, h)
    else:
        print("Unknown format: ."+path.split(".")[-1])
        parser.print_help()
        return
    
    if not os.path.exists("res"):
        os.mkdir("res")
    load_textures()
    loadcache()
    
    if input_type == 1:
        blockize_image(path, content_size)
    elif input_type == 2:
        try:
            frame_range = FrameRange.parse(args.range)
        except Exception as e:
            print(f"Failed to parse frame range \"{args.range}\"")
            parser.print_help()
            return
        blockize_video(path, content_size, frame_range, output_ext=args.ext)
    else:
        print("Unknown type: ."+path.split(".")[-1])
        parser.print_help()
        return
    savecache()
    print(f"{type(next(iter(colorcache.values())))} {type(next(iter(colorcache.values()))[0])}")




if __name__ == "__main__":
    main()