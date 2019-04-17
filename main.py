import argparse
import sys

import dnnlib.submission.submit as submit
import dnnlib.util as util

import train
import validation
import network

# Submit config
# ------------------------------------------------------------------------------------------

submit_config = submit.SubmitConfig()
submit_config.num_gpus = 1
submit_config.print_info = False
submit_config.ask_confirmation = False
submit_config.results_dir = "results"

# Tensorflow config
# ------------------------------------------------------------------------------------------

tf_config = util.EasyDict()
# TODO: Find out what graph pruning is (Network works without this option explicitly turned on)
tf_config["graph_options.place_pruned_graph"] = True

# Network config
# ------------------------------------------------------------------------------------------

net_config = util.EasyDict(func=network.autoencoder)

# Optimizer config
# ------------------------------------------------------------------------------------------

# Optimizer using mostly default settings defined in module optimizer.
# learning rate gets ramped down during training and is therefore set in method train.
optimizer_config = util.EasyDict(beta1=0.9, beta2=0.99, epsilon=1e-8)

# Noise augmentation config
# ------------------------------------------------------------------------------------------


gaussian_noise_config = util.EasyDict(
    func=train.AugmentGaussian,
    func_kwargs=util.EasyDict(
        train_stddev_rng_range=(0.0, 50.0),
        validation_stddev=25.0
    )
)
poisson_noise_config = util.EasyDict(
    func=train.AugmentPoisson,
    func_kwargs=util.EasyDict(
        lam_max=50.0,
        validation_chi=30.0
    )
)
text_noise_config = util.EasyDict(
    func=train.AugmentTextOverlays,
    func_kwargs=util.EasyDict(
        min_words=50,
        max_words=100,
        max_wordlength=30,
        fontsize=(15, 25)
    )
)

# ------------------------------------------------------------------------------------------
# Preconfigured validation sets
validation_config = util.EasyDict(dataset_dir='prepare_dataset/dataset')

corruption_types = {
    'gaussian': gaussian_noise_config,
    'poisson': poisson_noise_config,
    'text': text_noise_config
}

# Default train config
# ------------------------------------------------------------------------------------------

train_config = util.EasyDict(
    iteration_count=1001,
    eval_interval=1000,
    minibatch_size=4,
    run_func_name="train.train",
    learning_rate=0.0003,
    ramp_down_perc=0.3,
    noise=gaussian_noise_config,
    train_tfrecords='dataset_records/train.tfrecords',
    tf_config=tf_config,
    net_config=net_config,
    optimizer_config=optimizer_config,
    validation_config=validation_config
)

# Default validation run config
# ------------------------------------------------------------------------------------------
validate_config = util.EasyDict(
    tf_config=tf_config,
    run_func_name="validation.validate",
    dataset=validation_config,
    network_snapshot=None,
    noise=gaussian_noise_config
)

# Setup utility
# ------------------------------------------------------------------------------------------


def error(*print_args):
    print(*print_args)
    sys.exit(1)


# ------------------------------------------------------------------------------------------
examples = '''examples:

  # Train a network using the BSD300 dataset:
  python %(prog)s train --train-tfrecords=dataset_records/bsd300.tfrecords

  # Run a set of images through a pre-trained network:
  python %(prog)s validate --network-snapshot=results/network_final.pickle --dataset-dir=dataset_records/kodak
'''

if __name__ == "__main__":
    def training(args):
        """Setting up and finally running network training with user specific arguments."""
        if args:
            # reconfigure if long_train was specified
            if 'long_train' in args and args.long_train:
                train_config.iteration_count = 500000
                train_config.eval_interval = 5000
                train_config.ramp_down_perc = 0.5
        else:
            print('running with defaults in train_config')

        # Choose noise (default is gaussian)
        noise = 'gaussian'
        if 'noise' in args:
            if args.noise not in corruption_types:
                error('Unknown noise type', args.noise)
            else:
                noise = args.noise
        train_config.noise = corruption_types[noise]

        # Choose different tfrecords if given.
        if 'train_tfrecords' in args and args.train_tfrecords is not None:
            train_config.train_tfrecords = args.train_tfrecords

        # Apply the train function as the function to run.
        submit_config.run_func = train.train
        print(train_config)
        # Run this training setup.
        submit.submit_run(submit_config, **train_config)

    def validate(args):
        """Setting up and finally running network validation with user specific arguments."""
        if args.dataset_dir is None:
            error('Must select dataset with --dataset-dir')
        else:
            # Validation set must be given.
            validate_config.dataset = {
                'dataset_dir': args.dataset_dir
            }
        # Checks and sets the type of noise.
        if args.noise not in corruption_types:
            error('Unknown noise type', args.noise)
        validate_config.noise = corruption_types[args.noise]
        # Specify the pickled file of the trained network.
        if args.network_snapshot is None:
            error('Must specify trained network filename with --network-snapshot')
        validate_config.network_snapshot = args.network_snapshot
        # Choose the validate function as the function to run.
        submit_config.run_func = validation.validate
        # Runs this validation setup.
        submit.submit_run(submit_config, **validate_config)

    def infer_image(args):
        """Infering a chosen image to a trained and pickled network."""
        if args.image is None:
            error('Must specify image file with --image')
        if args.out is None:
            error('Must specify output image file with --out')
        if args.network_snapshot is None:
            error('Must specify trained network filename with --network-snapshot')
        # Note: there's no dnnlib.submission.submit_run here. This is for quick interactive
        # testing, not for long-running training or validation runs.
        validation.infer_image(tf_config, args.network_snapshot, args.image, args.out)

    # Uses train-function by default
    parser = argparse.ArgumentParser(
        description='Train a network or run a set of images through a trained network.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--run-dir-root', help='Working dir for a training or a validation run. Will contain training and validation results.')
    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    # Parameters for training
    parser_train = subparsers.add_parser('train', help='Train a network')
    parser_train.add_argument('--noise', default='gaussian', help='Type of noise corruption (one of: gaussian, poisson)')
    parser_train.add_argument('--long-train', default=False, help='Train for a very long time (500k iterations or 500k*minibatch image)')
    parser_train.add_argument('--train-tfrecords', help='Filename of the training set tfrecords file')
    parser_train.set_defaults(func=training)

    # Parameters for validation
    parser_validate = subparsers.add_parser('validate', help='Run a set of images through the network')
    parser_validate.add_argument('--dataset-dir', help='Load all images from a directory (*.png, *.jpg/jpeg, *.bmp)')
    parser_validate.add_argument('--network-snapshot', help='Trained network pickle')
    parser_validate.add_argument('--noise', default='gaussian', help='Type of noise corruption (one of: gaussian, poisson)')
    parser_validate.set_defaults(func=validate)

    # Parameters to infer an image
    parser_infer_image = subparsers.add_parser('infer-image', help='Run one image through the network without adding any noise')
    parser_infer_image.add_argument('--image', help='Image filename')
    parser_infer_image.add_argument('--out', help='Output filename')
    parser_infer_image.add_argument('--network-snapshot', help='Trained network pickle')
    parser_infer_image.set_defaults(func=infer_image)

    args = parser.parse_args()
    if args.run_dir_root is not None:
        submit_config.result_dir = args.result_dir

    if args.command is not None:
        args.func(args)
    else:
        # Train if no subcommand was given
        training(args)
