import os
import pprint
import re
import time

from .. import util


class SubmitConfig(util.EasyDict):
    """Strongly typed config dict needed to submit runs.
    This is nothing more than a wrapper for config attributes.

    Attributes:
        num_gpus: Number of GPUs used/requested for the run.
        print_info: Whether to print debug information when submitting.
        ask_confirmation: Whether to ask a confirmation before submitting.
        run_id: Automatically populated value during submit.
        run_name: Automatically populated value during submit.
        task_name: Automatically populated value during submit.
        host_name: Automatically populated value during submit.
    """

    def __init__(self):
        super().__init__()

        # submit (set these)
        self.num_gpus = 1
        self.print_info = False
        self.ask_confirmation = False
        self.results_dir = None

        # (automatically populated)
        self.run_func = None
        self.run_func_kwargs = None
        self.run_id = None
        self.run_name = None
        self.task_name = None
        self.host_name = "localhost"


def submit_run(submit_config: SubmitConfig, run_func_name: str, **run_func_kwargs) -> None:
    """Create a run dir, gather files related to the run and launch the run."""
    submit_config.run_func_name = run_func_name
    submit_config.run_func_kwargs = run_func_kwargs

    _create_run_dir(submit_config)
    submit_config.task_name = str(submit_config.run_id) + "N2N"

    # save submit config to file.
    with open(os.path.join(submit_config.results_dir, "submit_config.txt"), "w") as f:
        pprint.pprint(submit_config, stream=f, indent=4, width=200, compact=False)

    # print submit config to console if specified.
    if submit_config.print_info:
        print("\nSubmit config:\n")
        pprint.pprint(submit_config, indent=4, width=200, compact=False)
        print()

    if submit_config.ask_confirmation:
        if not util.ask_yes_no("Continue submitting the job?"):
            return

    run_wrapper(submit_config)


def run_wrapper(submit_config: SubmitConfig) -> None:
    """Wrap the actual run function call for handling logging, exceptions etc."""

    # when running locally, redirect stderr to stdout, log stdout to a file, and force flushing
    logger = util.Logger(file_name=os.path.join(submit_config.results_dir, "log.txt"), file_mode="w", should_flush=True)

    try:
        print("dnnlib: Running {0}() on {1}...".format(submit_config.run_func_name, submit_config.host_name))
        start_time = time.time()
        # Start the actual run by executing the function (train, validate) given to the submit config.
        submit_config.run_func(submit_config=submit_config, **submit_config.run_func_kwargs)
        # Run is finished.
        print("dnnlib: Finished {0}() in {1}.".format(
            submit_config.run_func_name, util.format_time(time.time() - start_time)))
    except:
        raise
    finally:
        # Empty file is created to show that program has stopped.
        open(os.path.join(submit_config.results_dir, "_finished.txt"), "w").close()

    logger.close()


def _create_run_dir(submit_config: SubmitConfig) -> None:
    """Create a new dir with results with increasing ID number at the start."""

    # Creates the directory for putting run results, if it does not exist yet.
    if not os.path.exists(submit_config.results_dir):
        print("Creating the results dir: {}".format(submit_config.results_dir))
        os.makedirs(submit_config.results_dir)

    # Run id is based on existing directory names of the results dir.
    submit_config.run_id = _get_next_run_id(submit_config.results_dir)
    # Name of the run is a concat of the run id with a fixed string.
    submit_config.run_name = str(submit_config.run_id) + "Run_Out_N2N"
    # Directory of results for this run is named as the run name.
    direc = os.path.join(submit_config.results_dir, submit_config.run_name)

    if os.path.exists(direc):
        raise RuntimeError("The dir already exists! ({0})".format(direc))

    print("Creating the results dir for this run: {}".format(direc))
    os.makedirs(direc)
    # The results dir for this run is set to the new directory.
    submit_config.results_dir = direc


def _get_next_run_id(res_dir: str) -> int:
    """Reads all directory names in a given directory (non-recursive)
    and returns the next (increasing) run id. Assumes IDs are numbers at the start of the directory names."""
    dir_names = [d for d in os.listdir(res_dir) if os.path.isdir(os.path.join(res_dir, d))]
    r = re.compile("^\\d+")  # Regex: match one or more digits at the start of the string
    run_id = 0

    for dir_name in dir_names:
        m = r.match(dir_name)

        if m is not None:
            i = int(m.group())
            run_id = max(run_id, i + 1)

    return run_id