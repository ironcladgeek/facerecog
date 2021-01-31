from pathlib import Path


def get_image_files(path):
    path = Path(path)
    image_extensions = ['.jpg', '.png', '.jpeg']
    return [x for x in path.iterdir() if x.suffix.lower() in image_extensions]

