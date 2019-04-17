import os
import numpy as np
import pickle
from PIL import Image, ImageFont


def save_snapshot(submit_config, net, fname_postfix):
    """save network snapshot as a serialized file, load_snapshot are used save/restore trained networks"""
    dump_fname = os.path.join(submit_config.results_dir, "network_%s.pickle" % fname_postfix)
    with open(dump_fname, "wb") as f:
        pickle.dump(net, f)


def load_snapshot(fname):
    """load the network from the serialized pickle file."""
    with open(fname, "rb") as f:
        return pickle.load(f)


def save_image(submit_config, img_np, filename):
    """Converts a given image tensor to an image and saves it to the specified file name."""
    array_to_image(img_np).save(os.path.join(submit_config.results_dir, filename))


def draw_text_on_image(img_draw, word, position, fontsize, color):
    font = ImageFont.truetype("LiberationSans-Regular.ttf", fontsize)
    img_draw.text(position, word, color, font=font)


def array_to_image(img_np):
    """Converts the image array to a PIL image."""
    arr = img_np.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    if arr.dtype in [np.float32, np.float64]:
        # Transform pixel color data to color values in the usual byte representation (uint8).
        arr = clip_to_uint8(arr)
    else:
        assert arr.dtype == np.uint8
    return Image.fromarray(arr, 'RGB')


def image_to_array(image_pil):
    """Converts the PIL image to an image array with our desired format: CHW and normalized to [-0.5,0.5]"""
    arr = np.array(image_pil, dtype=np.uint8)
    return arr.transpose([2, 0, 1]).astype(np.float32) / 255.0 - 0.5


def clip_to_uint8(arr):
    """Transforms array values to range [0.0, 1.0], then to [0.0, 255.0], adds 0.5 to round up values
    with a decimal value >=.5, clips values to make sure they are in range [0.0, 255.0] again and lastly
    converts the values to uint8 for the RGB color values."""
    return np.clip((arr + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def crop_np(img, x, y, w, h):
    """Crops a numpy array image.
    First slicing dimension: All color channels are of course kept when cropping.
    Second slicing dimension: Get only values with height >=y and <h.
    Third slicing dimension: Get only values with width >=x and <w."""
    return img[:, y:h, x:w]


def infer_image(net, img):
    """Run an image through the network (apply reflect padding when needed and crop back to original dimensions.)"""
    w = img.shape[2]
    h = img.shape[1]
    # Calculates the padding for this image's width and height, so the total w and h are a multiple of 32.
    pw, ph = ((w+31)//32)*32-w, ((h+31)//32)*32-h
    padded_img = img
    if pw != 0 or ph != 0:
        # (0, 0) => pad nothing in the channel axis
        # (0, ph) => pad nothing before the height, pad "ph" number of values after the image height.
        # (0, pw) => pad nothing before the width, pad "pw" number of values after the image width.
        # reflect padding pads the values of the edge of the array in reverse order (as if the padding was a reflection)
        padded_img = np.pad(img, ((0, 0), (0, ph), (0, pw)), 'reflect')
    # Expand a dimension at the start of this array to fit the dimension count of 4 (expected by network.run).
    # TODO: check w+pw = padded_img.shape[2] and for height
    inferred = net.run(np.expand_dims(padded_img, axis=0), width=w+pw, height=h+ph)
    # Crop back to original dimensions and convert to pixel values.
    return clip_to_uint8(crop_np(inferred[0], 0, 0, w, h))
