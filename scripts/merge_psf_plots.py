#!/usr/bin/python3

""" Merge all the already created psf plots of a single frame
    in the current working directory based on their positions
    in the frame """

import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

files_list = os.listdir(os.getcwd())

initial_set = set(file[0:20] for file in files_list if len(file) > 25)

for initial in initial_set:
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5))
    for file_name in files_list:
        if file_name.startswith(initial):
            ix = (int(file_name[file_name.find("None)")+5]),
                  int(file_name[-5]))

            if ix == (0, 0):
                axes[2,0].imshow(mpimg.imread(file_name))
            elif ix == (0, 1):
                axes[1,0].imshow(mpimg.imread(file_name))
            elif ix == (0, 2):
                axes[0,0].imshow(mpimg.imread(file_name))

            elif ix == (1, 0):
                axes[2,1].imshow(mpimg.imread(file_name))
            elif ix == (1, 1):
                axes[1,1].imshow(mpimg.imread(file_name))
            elif ix == (1, 2):
                axes[0,1].imshow(mpimg.imread(file_name))

            elif ix == (2, 0):
                axes[2,2].imshow(mpimg.imread(file_name))
            elif ix == (2, 1):
                axes[1,2].imshow(mpimg.imread(file_name))
            elif ix == (2, 2):
                axes[0,2].imshow(mpimg.imread(file_name))

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{initial}.png", dpi=300)
    plt.clf()
    plt.close(fig)
