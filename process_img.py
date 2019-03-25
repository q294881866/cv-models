from PIL import Image
import piexif
import matplotlib.pyplot as plt

import os


def make_square(im, min_size=224, fill_color=(0, 0, 0, 0)):
    img = im.resize((168, 224), Image.ANTIALIAS)
    x, y = img.size
    print(img.size)
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    if size > min_size:
        print(size)
        new_im = new_im.resize((224, 224), Image.ANTIALIAS)

    return new_im


def img_list(dirname):
    for filename in os.listdir(dirname):
        path = dirname + filename
        if os.path.isdir(path):
            path += '/'
            img_list(path)
        else:
            try:
                if path.endswith(".jpg"):
                    img = Image.open(path)
                    img._getexif()
                    piexif.remove(path)
                    new_img = make_square(img)
                    new_path = path.replace(".jpg", ".png")
                    new_img.save(new_path)
                    os.remove(path)
            except :
                print(path)
                os.remove(path)


## 图片裁剪 jpg 转 png
img_list("/Users/peipengfei/Downloads/javaDesign/")

print("---end")

