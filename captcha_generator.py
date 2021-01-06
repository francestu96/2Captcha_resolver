from  utils.utils import random_text
from captcha.image import ImageCaptcha
from glob import glob
from PIL import Image
from claptcha import Claptcha

WIDTH = 250
HEIGHT = 80
IMAGES_PER_TEXT_LENGTH = 50
OUTPUT_DIR = ".\\captcha_images\\"
FONTS = glob("C:\\Windows\\fonts\\*.ttf")
CHARACTERS = [x for x in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789']

for font in FONTS:
    for len in range(5, 10):
        for i in range(IMAGES_PER_TEXT_LENGTH):
            captcha_text = random_text(CHARACTERS, len)
            claptcha = Claptcha(captcha_text, font, size=(WIDTH, HEIGHT), resample=Image.NEAREST, noise=0.8)
            claptcha.write(OUTPUT_DIR + captcha_text + ".png")

            captcha_text = random_text(CHARACTERS, len)
            image_captcha = ImageCaptcha(WIDTH, HEIGHT, [font])
            image_captcha.generate_image(captcha_text)
            image_captcha.write(captcha_text, OUTPUT_DIR + captcha_text + ".png")