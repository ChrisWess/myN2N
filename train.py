import tensorflow as tf
import numpy as np

import dnnlib
from dnnlib.tflib.optimizer import Optimizer
from dnnlib.tflib.autosummary import autosummary
from dnnlib.tflib.generic_network import Network
import dnnlib.tflib.tfutil as tfutil
import dnnlib.submission.submit as submit
import dnnlib.submission.run_context as run_context
import dnnlib.util as util

from util import save_image, save_snapshot
from validation import ValidationSet
from dataset import create_dataset


class AugmentGaussian:
    def __init__(self, validation_stddev, train_stddev_rng_range):
        self.validation_stddev = validation_stddev  # the fixed standard deviation for all validation images
        self.train_stddev_range = train_stddev_rng_range  # the range of values the standard deviation can adopt.

    def add_train_noise_tf(self, x: tf.Tensor) -> tf.Tensor:
        """Defines an operation to add gaussian noise to a tensor of an image with a random
        standard deviation in the specified range."""
        # x is the image with normalized pixel values [-0.5, 0.5]
        (minval, maxval) = self.train_stddev_range
        shape = tf.shape(x)
        # Gets a random factor as the standard deviation for all pixels of an image, so every image has a different
        # standard deviation of gaussian noise.
        rng_stddev = tf.random_uniform(shape=[1, 1, 1], minval=minval/255.0, maxval=maxval/255.0)
        return x + tf.random_normal(shape) * rng_stddev  # normal distribution creates the gaussian noise.

    def add_validation_noise_np(self, x: np.array) -> np.array:
        """Adds gaussian noise to a numpy array of an image with same range [-0.5, 0.5] and a fixed standard-dev"""
        return x + np.random.normal(size=x.shape) * (self.validation_stddev / 255.0)


class AugmentPoisson:
    def __init__(self, lam_max, validation_chi):
        self.lam_max = lam_max
        self.validation_chi = validation_chi

    def add_train_noise_tf(self, x: tf.Tensor) -> tf.Tensor:
        """Defines an operation to add poisson noise to a tensor of an image with a random chi-value
        in the range from 0.001 to the specified max value."""
        chi_rng = tf.random_uniform(shape=[1, 1, 1], minval=0.001, maxval=self.lam_max)
        # Add 0.5 to pixel to make it positive as arg for poisson distribution.
        # Subtract 0.5 at the end to fit in range again.
        return tf.random_poisson(chi_rng*(x+0.5), shape=[])/chi_rng - 0.5

    def add_validation_noise_np(self, x: np.array) -> np.array:
        """Adds poisson noise to an numpy array of an image with a fixed chi-value"""
        return np.random.poisson(self.validation_chi * (x + 0.5)) / self.validation_chi - 0.5


"""
Probably not possible here!
class AugmentTextOverlays:
    def __init__(self, max_words, max_wordlength, fontsize):
        self.max_words = max_words
        self.max_wl = max_wordlength
        self.range_fontsize = fontsize

    def add_train_noise_tf(self, x: tf.Tensor) -> tf.Tensor:
        print("todo")

    def add_validation_noise_np(self, x: np.array) -> np.array:
        print("todo")"""


def compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate):
    # Calculates after which iteration to start the ramping down process.
    ramp_down_start_iter = iteration_count * (1 - ramp_down_perc)
    if i >= ramp_down_start_iter:
        t = ((i - ramp_down_start_iter) / ramp_down_perc) / iteration_count
        smooth = (0.5+np.cos(t * np.pi)/2)**2
        return learning_rate * smooth
    return learning_rate


def train(
        submit_config: submit.SubmitConfig,
        iteration_count: int,
        eval_interval: int,
        minibatch_size: int,
        learning_rate: float,
        ramp_down_perc: float,
        noise: dict,
        tf_config: dict,
        net_config: dict,
        optimizer_config: dict,
        validation_config: dict,
        train_tfrecords: str):

    # **dict as argument means: take all additional named arguments to this function
    # and insert them into this parameter as dictionary entries.
    noise_augmenter = noise.func(**noise.func_kwargs)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**validation_config)

    # Create a run context (hides low level details, exposes simple API to manage the run)
    ctx = run_context.RunContext(submit_config)

    # Initialize TensorFlow graph and session using good default settings
    tfutil.init_tf(tf_config)

    dataset_iter = create_dataset(train_tfrecords, minibatch_size, noise_augmenter.add_train_noise_tf)

    # Construct the network using the Network helper class and a function defined in config.net_config
    with tf.device("/gpu:0"):
        net = Network(**net_config)

    # Optionally print layer information
    net.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device("/cpu:0"):
        # Placeholder for the learning rate. This will get ramped down dynamically.
        lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])

        noisy_input, noisy_target, clean_target = dataset_iter.get_next()
        noisy_input_split = tf.split(noisy_input, submit_config.num_gpus)
        noisy_target_split = tf.split(noisy_target, submit_config.num_gpus)  # Split over multiple GPUs
        clean_target_split = tf.split(clean_target, submit_config.num_gpus)

    # Define the loss function using the Optimizer helper class, this will take care of multi GPU
    opt = Optimizer(learning_rate=lrate_in, **optimizer_config)

    for gpu in range(submit_config.num_gpus):
        with tf.device("/gpu:%d" % gpu):
            net_gpu = net if gpu == 0 else net.clone()

            denoised = net_gpu.get_output_for(noisy_input_split[gpu])

            meansq_error = tf.reduce_mean(tf.square(noisy_target_split[gpu] - denoised))
            # Create an autosummary that will average over all GPUs
            with tf.control_dependencies([autosummary("Loss", meansq_error)]):
                opt.register_gradients(meansq_error, net_gpu.trainables)

    train_step = opt.apply_updates()  # Defines the update function of the optimizer.

    # Create a log file for Tensorboard
    summary_log = tf._api.v1.summary.FileWriter(submit_config.results_dir)
    summary_log.add_graph(tf.get_default_graph())

    print('Training...')
    time_maintenance = ctx.get_time_since_last_update()
    ctx.update()  # TODO: why parameterized in reference?

    # The actual training loop
    for i in range(iteration_count):
        # Whether to stop the training or not should be asked from the context
        if ctx.should_stop():
            break

        # Dump training status
        if i % eval_interval == 0:

            time_train = ctx.get_time_since_last_update()
            time_total = ctx.get_time_since_start()

            # Evaluate 'x' to draw one minbatch of inputs. Executes the operations defined in the dataset iterator.
            [source_mb, target_mb] = tfutil.run([noisy_input, clean_target])
            # Runs the noisy images through the network without training it. It is just for observing/evaluating.
            denoised = net.run(source_mb)
            # array shape: [minibatch_size, channel_size, height, width]
            save_image(submit_config, denoised[0], "img_{0}_y_pred.png".format(i))
            save_image(submit_config, target_mb[0], "img_{0}_y.png".format(i))
            save_image(submit_config, source_mb[0], "img_{0}_x_aug.png".format(i))

            validation_set.evaluate(net, i, noise_augmenter.add_validation_noise_np)

            print('iter %-10d time %-12s sec/eval %-7.1f sec/iter %-7.2f maintenance %-6.1f' % (
                autosummary('Timing/iter', i),
                dnnlib.util.format_time(autosummary('Timing/total_sec', time_total)),
                autosummary('Timing/sec_per_eval', time_train),
                autosummary('Timing/sec_per_iter', time_train / eval_interval),
                autosummary('Timing/maintenance_sec', time_maintenance)))

            dnnlib.tflib.autosummary.save_summaries(summary_log, i)
            ctx.update()
            time_maintenance = ctx.get_last_update_interval() - time_train

        lrate = compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate)
        # Apply the lrate value to the lrate_in placeholder for the optimizer.
        tfutil.run([train_step], {lrate_in: lrate})  # Run the training update through the network in our session.

    print("Elapsed time: {0}".format(util.format_time(ctx.get_time_since_start())))
    save_snapshot(submit_config, net, 'final')

    # Summary log and context should be closed at the end
    summary_log.close()
    ctx.close()
