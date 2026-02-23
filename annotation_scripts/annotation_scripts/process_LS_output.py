import json, cv2
import numpy as np

from typing import List

####################################################

STUDENT_NO = "25100812"
SURNAME = "Liang"

save_file = f"dataset/{STUDENT_NO}_{SURNAME}"
label_json_path = "annotations.json"

####################################################
####################################################

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def rle_to_mask(rle: List[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image

####################################################
####################################################

prefix = "/data/local-files/?d="

with open(label_json_path) as f:
    data = json.load(f)

for i, d in enumerate(data):
    img_name = d["image"]
    print(img_name)
    filepath = img_name.replace(prefix, "").replace("%5C", "/").replace("_clip", "/clip").replace("_rgb_", "/annotation/").replace("labelling/images", save_file)

    image = rle_to_mask(
    d["tag"][0]['rle'], 
    d["tag"][0]['original_height'], 
    d["tag"][0]['original_width']
    )

    cv2.imwrite(filepath, image)

    print(f"saved mask {i+1} to file {filepath}")

print(f"Task completed - {i+1} masks saved")
