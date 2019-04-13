import os
import sys
import numpy as np
import PIL.Image
import tensorflow as tf

import dnnlib
import dnnlib.submission.submit as submit
from dnnlib.submission.run_context import RunContext
import dnnlib.tflib.tfutil as tfutil
from dnnlib.tflib.autosummary import autosummary

import util


class ValidationSet:
    def __init__(self, submit_config):
        self.images = None
        self.submit_config = submit_config
        return

    def load(self, dataset_dir):
        import glob

        # Retrieve the path containing all validation images.
        abs_dirname = os.path.join(dataset_dir, '*')
        # Compiles the file pattern to all file names of the images.
        fnames = sorted(glob.glob(abs_dirname))
        if len(fnames) == 0:
            print('\nERROR: No files found using the following glob pattern:', abs_dirname, '\n')
            sys.exit(1)

        images = []
        for fname in fnames:
            try:
                # In this loop, open every image in the RGB format.
                im = PIL.Image.open(fname).convert('RGB')
                # Convert the image to numpy array
                arr = np.array(im, dtype=np.float32)
                print(arr.shape)  # TODO: [h,w,3] ?
                # Converts HWC to CHW and normalizes color values to range [-0.5, 0.5]
                reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
                # Appends transformed image array to the list of all images.
                images.append(reshaped)
            except OSError as e:
                print('Skipping file', fname, 'due to error: ', e)
        # Assigns all images of the dataset as a list of numpy arrays to this variable.
        self.images = images

    def evaluate(self, net, iteration, noise_func):
        avg_psnr = 0.0
        for idx in range(len(self.images)):
            orig_img = self.images[idx]
            w = orig_img.shape[2]
            h = orig_img.shape[1]

            noisy_img = noise_func(orig_img)
            pred255 = util.infer_image(net, noisy_img)
            orig255 = util.clip_to_uint8(orig_img)
            assert (pred255.shape[2] == w and pred255.shape[1] == h)

            sqerr = np.square(orig255.astype(np.float32) - pred255.astype(np.float32))
            s = np.sum(sqerr)
            cur_psnr = 10.0 * np.log10((255*255)/(s / (w*h*3)))
            avg_psnr += cur_psnr

            # Saves the prediction of all images of this iteration in the results directory.
            #util.save_image(self.submit_config, pred255, "img_{0}_val_{1}_pred.png".format(iteration, idx))

            #if iteration == 0:
                #util.save_image(self.submit_config, orig_img, "img_{0}_val_{1}_orig.png".format(iteration, idx))
                #util.save_image(self.submit_config, noisy_img, "img_{0}_val_{1}_noisy.png".format(iteration, idx))
        avg_psnr /= len(self.images)
        print('Average PSNR: %.2f' % autosummary('PSNR_avg_psnr', avg_psnr))


def validate(submit_config: submit.SubmitConfig, tf_config: dict, noise: dict, dataset: dict, network_snapshot: str):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**dataset)

    ctx = RunContext(submit_config)

    tfutil.init_tf(tf_config)

    with tf.device("/gpu:0"):
        net = util.load_snapshot(network_snapshot)
        validation_set.evaluate(net, 0, noise_augmenter.add_validation_noise_np)
    ctx.close()


def infer_image(tf_config: dict, network_snapshot: str, image: str, out_image: str):
    tfutil.init_tf(tf_config)
    net = util.load_snapshot(network_snapshot)
    im = PIL.Image.open(image).convert('RGB')
    arr = np.array(im, dtype=np.float32)
    reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
    pred255 = util.infer_image(net, reshaped)
    t = pred255.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    PIL.Image.fromarray(t, 'RGB').save(os.path.join(out_image))
    print('Inferred image saved in', out_image)
