import glob
import os
import subprocess
from collections import Counter

from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

file_counter = Counter()

from download_youtube import clean_filename


def segment_file(d):
    f, segment_dir_path = d
    cleaned_filename = clean_filename(os.path.basename(f))
    cleaned_filename = cleaned_filename[:-3]

    output_filename = os.path.join(segment_dir_path, cleaned_filename + "_%03d.wav")

    command = ["ffmpeg", "-nostats", "-loglevel", "0", "-y", "-i", f, "-map", "0", "-ac", "1", "-ar",
               "16000", "-f", "segment",
               "-segment_time", "10", output_filename]
    subprocess.call(command)

    return f


if __name__ == '__main__':
    output_path_raw = '/media/work/audio/musiclid/youtube_spoken/raw'
    output_path_segmented = '/media/work/audio/musiclid/youtube_spoken/segmented'

    pool = Pool()

    for language in os.listdir(output_path_raw):
        for source_name in os.listdir(os.path.join(output_path_raw, language)):
            files = glob.glob(os.path.join(output_path_raw, language, source_name, "*.mp3"))

            segment_dir_path = os.path.join(output_path_segmented, language, source_name)
            if not os.path.exists(segment_dir_path):
                os.makedirs(segment_dir_path)

            data = [(f, segment_dir_path) for f in files]
            data = []
            for f in tqdm(pool.imap(segment_file, data), 'segmenting files in {}/{}'.format(language, source_name),
                          total=len(files)):
                pass

            file_counter[language] += len(glob.glob(os.path.join(segment_dir_path, "*.wav")))

    print file_counter
