# SPDX-License-Identifier: LGPL-3.0-or-later
# Author: Kristian Lytje

from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import concurrent.futures
import argparse
import os

try:
    from .plot_helper import plot_file, plot_fits
    from .plot_animator import (animation_groups, locked_residual_axes,
                                render_frame_png, frame_duration_ms, assemble_gif)
except ImportError:
    from plot_helper import plot_file, plot_fits
    from plot_animator import (animation_groups, locked_residual_axes,
                               render_frame_png, frame_duration_ms, assemble_gif)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="plot", description="AUSAXS plotting utility.")
    parser.add_argument("--folder", type=str, default=".", help="Folder to search for .plot files.")
    parser.add_argument("--max_depth", type=int, default=4, help="Maximum depth to search for .plot files.")
    parser.add_argument("--max_files", type=int, default=30, help="Maximum number of files to process.")
    parser.add_argument("--big", action="store_true", help="Use big font size.")
    parser.add_argument("--medium", action="store_true", help="Use medium font size.")
    parser.add_argument("--title", type=str, help="Title for the plot.")
    parser.add_argument("--loop", type=int, default=0, help="Loop count for generated GIFs (0 = loop forever).")
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--fps", type=float, help="Frames per second for generated GIF animations.")
    timing.add_argument("--duration", type=float, help="Total length in seconds of generated GIF animations.")
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    folder = args.folder
    max_depth = args.max_depth
    max_files = args.max_files
    if args.big:
        params = {
            'legend.fontsize': 28,
            'figure.figsize': (10, 8),
            'axes.labelsize': 28,
            'axes.titlesize': 28,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'lines.markersize': 12,
            'lines.linewidth': 2.5,
            'backend': 'Agg'
        }
    elif args.medium:
        params = {
            'legend.fontsize': 24,
            'figure.figsize': (10, 8),
            'axes.labelsize': 24,
            'axes.titlesize': 24,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'lines.markersize': 10,
            'lines.linewidth': 2,
            'backend': 'Agg'
        }
    else:
        params = {
            'legend.fontsize': 14,
            'figure.figsize': (10, 8),
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'lines.markersize': 5,
            'backend': 'Agg'
        }
    title = args.title

    plt.rcParams.update(params)

    def get_depth(path):
        return path.count(os.sep)

    with concurrent.futures.ProcessPoolExecutor(8) as executor:
        futures = []
        gif_jobs = []  # (output gif path, [ordered frame-render futures])
        invoke_depth = get_depth(folder)
        for currentpath, _, files in os.walk(folder):
            if max_depth < get_depth(currentpath) - invoke_depth:
                continue

            # enumerated frame sequences are detected from the .png.plot data files
            groups = animation_groups([f for f in files if f.endswith(".png.plot")])
            grouped_frames = {name for frames in groups.values() for name in frames}

            # a long animation sequence should not trip the too-many-files guard
            if max_files < len([f for f in files if f not in grouped_frames]):
                print(f"Skipping {currentpath} because it has too many files.")
                continue

            # animation frames are pulled out of the normal rendering: a shared axis
            # range is computed across the sequence and locked into every frame so the
            # axes do not fluctuate, then the frames are rendered straight into the gif
            for stem, frames in groups.items():
                gif_path = os.path.join(currentpath, stem + ".gif")
                frame_paths = [os.path.join(currentpath, f) for f in frames]
                intensity_ylim, residual_ylim = locked_residual_axes(frame_paths)
                frame_futures = [executor.submit(render_frame_png, p, intensity_ylim, residual_ylim)
                                 for p in frame_paths]
                gif_jobs.append((gif_path, frame_futures))

            fit_files = []
            ausaxs_file = ""
            for file in files:
                extension = file.split(".")[-1]
                if file == "ausaxs.fit":
                    ausaxs_file = os.path.join(currentpath, file)
                    continue
                if file in grouped_frames:  # rendered into a gif, not as a standalone image
                    continue
                match extension:
                    case "fit" | "xvg":
                        fit_files.append(os.path.join(currentpath, file))
                    case "plot":
                        futures.append(executor.submit(plot_file, os.path.join(currentpath, file)))

            if ausaxs_file:
                futures.append(executor.submit(plot_fits, ausaxs_file, fit_files, title))

        concurrent.futures.wait(futures + [fut for _gif, futs in gif_jobs for fut in futs])

        # frames are rendered (with locked axes); stitch each sequence into its gif
        for gif_path, frame_futures in gif_jobs:
            try:
                frames_png = [fut.result() for fut in frame_futures]
                duration = frame_duration_ms(len(frames_png), args.fps, args.duration)
                assemble_gif(frames_png, gif_path, duration, args.loop)
                print(f"Wrote animation {gif_path} ({len(frames_png)} frames, {duration} ms/frame)")
            except Exception as e:
                print(f"Failed to write {gif_path}: {e}")

if __name__ == "__main__":
    freeze_support()
    main()