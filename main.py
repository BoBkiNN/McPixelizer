import click
from pathlib import Path
import os
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
# from nbtlib import *
from tqdm import tqdm
from dataclasses import dataclass

def create_bar(name: str, max: int, position: int | None = None):
    return tqdm(total=max, ncols=100, 
                bar_format='{desc} |{bar}| {n_fmt}/{total_fmt}; ETA: {remaining}; {elapsed}', 
                desc=name, position=position)

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

@dataclass
class Context:
    results_folder: Path
    block_size: tuple[int, int]

    @property
    def frames_folder(self):
        return self.results_folder / "frames"


def process_frame(sorted_textures: dict[str, np.ndarray], 
                  image: np.ndarray, ctx: Context, index: int = 0,
                  output: Path | None = None, bar_position: int | None = None):
    block_size = ctx.block_size
    # Resize frame
    resized_frame = cv2.resize(image, block_size)
    # print(f"{len(resized_frame)}, {len(resized_frame[0])}")
    # ret = np.zeros((SIZE[0], SIZE[1], 3), dtype=np.uint8)
    bar = create_bar(f'Frame {index}', block_size[1]*block_size[0], position=bar_position)
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
    if output is None:
        rp = ctx.frames_folder / f"{index}.png"
        ctx.frames_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rp), blocked)
    else:
        rp = ctx.results_folder / output.name
        cv2.imwrite(str(rp), blocked)
    bar.close()
    # save_blocked(index, processed_frame)



def make_video(fps: float, ctx: Context, output: Path, use_ffmpeg = True):
    if use_ffmpeg:
        i = ctx.frames_folder / "%d.png"
        subprocess.run(["ffmpeg", "-framerate", str(fps), "-i", str(i), str(output)], stdout=subprocess.DEVNULL)
        print("Created using FFMPEG "+output.suffix.upper()+" video: "+output)
        return
    size = tuple(element * 16 for element in ctx.block_size)
    writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'),fps, tuple(reversed(size)))
    frames = []
    frame_files = os.listdir(ctx.frames_folder)
    for frame_file in sorted(frame_files):
        if frame_file.endswith('.png'):
            frames.append(frame_file)
    bar = create_bar(f'Making video', len(frames))
    for frame_file in frames:
        frame_path = str(ctx.frames_folder) / frame_file
        frame = cv2.imread(frame_path)
        writer.write(frame)
        bar.update()
    writer.release()
    bar.close()

def blockize_image(ctx: Context, input_file: Path = Path("input.png")):
    block_size = ctx.block_size
    img: np.ndarray = cv2.imread(str(input_file))
    print(f"Processing image \"{input_file}\"; Size: {img.shape[0]}x{img.shape[1]} ({block_size[0]}x{block_size[1]} blocks)")
    out_name = input_file.stem+f"_{block_size[0]}x{block_size[1]}_output.png"
    out_path = ctx.results_folder / out_name
    textures = preload_sorted()
    process_frame(textures, img, ctx, -1, out_path)

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

def blockize_video(video_path: Path, ctx: Context, frame_range = FrameRange(), output_ext: str = "avi"):
    # Load video and extract frames
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    result_frame_count = frame_range.size(frame_count-1)
    if frame_range.f == 0 and frame_range.t == -1:
        result_fps = fps
    else:
        result_fps = (result_frame_count * fps) / frame_count
    print(f"Processing video \"{video_path}\"; FPS {result_fps}; Total frames: {int(result_frame_count)}|{int(frame_count)}")
    sorted_textures = preload_sorted()

    threads: list[threading.Thread] = []
    index = 0
    slot_index = 0

    tqdm.set_lock(threading.RLock())

    video_bar = create_bar("Overall frames", int(
        frame_count), position=thread_count)
    
    error: Exception | None = None

    while cap.isOpened():
        video_bar.refresh()
        if error is not None:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = ctx.frames_folder / f"{index}.png"
        if frame_path.is_file():
            index += 1
            video_bar.update()
            continue
        if not frame_range.inrange(index):
            writedebug(f"not in range {index}")
            index += 1
            video_bar.update()
            continue
        if len(threads) >= thread_count:
            # Wait for batch to finish
            while any(t.is_alive() for t in threads):
                time.sleep(0.001)

            # Close frame bars
            for bar in frame_bars:
                bar.close()
            frame_bars.clear()
            threads.clear()
            slot_index = 0

        # Assign slot position for frame bar
        current_slot = slot_index
        slot_index += 1

        def handle(*args, bar_position=None):
            nonlocal error
            try:
                process_frame(*args, bar_position=bar_position)
            except Exception as e:
                error = e
            video_bar.update()

        t = threading.Thread(
            name=f"Frame {index}",
            target=handle,
            args=(sorted_textures, frame, ctx, index),
            kwargs={"bar_position": current_slot}
        )
        t.start()
        threads.append(t)

        index += 1

    cap.release()
    while any(t.is_alive() for t in threads):
        time.sleep(0.001)
    threads.clear()
    for bar in frame_bars:
        bar.close()
    frame_bars.clear()
    video_bar.close()

    if error is not None:
        raise RuntimeError("Exception processing video") from error

    block_size = ctx.block_size

    out_name = video_path.stem+f"_{block_size[0]}x{block_size[1]}_output."+output_ext
    out_path = ctx.results_folder / out_name
    make_video(result_fps, ctx, out_path)


thread_count = 5


@click.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, path_type=Path),
    default=Path("input.mp4"),
    help="Input file name (default: input.mp4). Must be .mp4 or .png"
)
@click.option(
    "--size", "-s",
    type=str,
    help="Specify size as a string in format {width}x{height}"
)
@click.option(
    "--range", "-r",
    default="0..",
    type=str,
    help="Specify frames indexes to process from video in format {from}..{to} "
         "(all inclusive)(default is 0..). Examples: ..5 (from 0 to five), 3.. (from 3 to end), 2..6 (from 2 to 6)"
)
@click.option(
    "--ext", "-e",
    type=click.Choice(["avi", "mp4", "gif"]),
    default="avi",
    help="Specify output video extension"
)
@click.option(
    "--threads", "-tc",
    type=int,
    default=5,
    help="Thread count. Default is 5"
)
@click.option(
    "--name", "-n",
    type=str,
    default="main",
    help="Project name. Used to have separate folders. Default is main"
)
def main(input: Path, size: str, range: str, ext: str, threads: int, name: str):
    global thread_count
    thread_count = threads

    if not input.is_file():
        click.echo("Unknown file "+input)
        return
    if size != None:
        try:
            ls = size.split("x")
            w = int(ls[0])
            h = int(ls[1])
            content_size = (w, h)
        except:
            click.echo(f"\"{size}\" is not valid size")
            return

    input_type = -1
    input_ext = input.suffix
    if input_ext == ".png":
        input_type = 1
        if size == None:
            img: np.ndarray = cv2.imread(str(input))
            content_size: tuple[int, int] = img.shape[:2]
            del img
    elif input_ext == ".mp4":
        input_type = 2
        if size == None:
            vid = cv2.VideoCapture(str(input))
            w: int = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            h: int = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            vid.release()
            content_size = (w, h)
    else:
        click.echo("Unknown format: "+input.suffix)
        return
    
    results_dir = Path("res") / name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resolved block size: {content_size}")
    context = Context(results_folder=results_dir, block_size=content_size)
    print(f"Results folder: {results_dir}")

    load_textures()
    loadcache()
    
    if input_type == 1:
        blockize_image(context, input)
    elif input_type == 2:
        try:
            frame_range = FrameRange.parse(range)
        except Exception as e:
            click.echo(f"Failed to parse frame range \"{range}\"")
            return
        blockize_video(input, context, frame_range, output_ext=ext)
    else:
        click.echo("Unknown type: "+input.suffix)
        return
    savecache()
    print(f"Results folder: {results_dir}")

if __name__ == "__main__":
    main()