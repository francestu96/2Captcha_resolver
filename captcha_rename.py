from glob import glob
import random
import sys
import os

if len(sys.argv) < 2:
    print("usage: captcha_rename <root_image_directory>")
    exit(0)

root_dir = sys.argv[1]
images_paths = random.sample([y for x in os.walk(root_dir) for y in glob(os.path.join(x[0], '*.jpg'))], 500000)
for image_path in images_paths:
    # regex: xxx_solution_xxx.jpg
    new_path = image_path.split("_")[-2] + ".jpg"
    os.rename(image_path, ".\\captcha_images\\" + new_path)

