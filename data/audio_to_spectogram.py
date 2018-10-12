import glob
import os

import numpy as np
import scipy.misc
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

from keras_code.data_loaders.SpectrogramGenerator import SpectrogramGenerator

config = {
    "pixel_per_second": 50,
    "input_shape": [129, 500, 1]
}

sg = SpectrogramGenerator(source='.', config=config, shuffle=False, run_only_once=True)

target_height, target_width, target_channels = config["input_shape"]


def segment_file(d):
    f, spectogram_dir_path, index = d

    image = sg.audioToSpectrogram(f, config["pixel_per_second"], target_height)
    image = np.expand_dims(image, -1)  # add dimension for mono channel

    height, width, channels = image.shape

    assert target_height == height, "Heigh mismatch {} vs {}".format(target_height, height)

    num_segments = width // target_width

    for i in range(0, num_segments):
        slice_start = i * target_width
        slice_end = slice_start + target_width

        slice = image[:, slice_start:slice_end]

        # Ignore black images
        if slice.max() == 0 and slice.min() == 0:
            continue

        file_name = os.path.join(spectogram_dir_path, "{}{}.png".format(index, i))
        scipy.misc.imsave(file_name, np.squeeze(slice))

    return f


if __name__ == '__main__':
    dataset_dir = '/media/work/audio/musiclid/youtube_spoken/'
    output_path_raw = os.path.join(dataset_dir, 'raw')
    output_path_spectograms = os.path.join(dataset_dir, 'spectograms')

    pool = Pool()

    for language in os.listdir(output_path_raw):
        for source_name in os.listdir(os.path.join(output_path_raw, language)):
            files = glob.glob(os.path.join(output_path_raw, language, source_name, "*.mp3"))

            spectogram_dir_path = os.path.join(output_path_spectograms, language, source_name)
            if not os.path.exists(spectogram_dir_path):
                os.makedirs(spectogram_dir_path)

            data = [(f, spectogram_dir_path, i) for i, f in enumerate(files)]
            for f in tqdm(pool.imap(segment_file, data), 'spectograms for {}/{}'.format(language, source_name),
                          total=len(files)):
                pass
