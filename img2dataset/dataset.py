from img2dataset import download
import shutil
import os

if __name__ == "__main__":
    output_dir = os.path.abspath("data")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    download(
        url_list="list.txt",
        image_size=256,
        output_folder=output_dir,
        processes_count=32,
        thread_count=256,
        resize_mode="border",
    )