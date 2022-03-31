import contextlib
import pathlib
import sys
from typing import Callable
from typing import Iterator, Optional, Union, Iterable, List
from typing import TYPE_CHECKING

import numpy as np
import stim

from simmer.csv_out import CSV_HEADER
from simmer.collection_work_manager import CollectionWorkManager
from simmer.existing_data import ExistingData


if TYPE_CHECKING:
    import simmer


def iter_collect(*,
                 num_workers: int,
                 tasks: Union[Iterator['simmer.Task'],
                              Iterable['simmer.Task']],
                 print_progress: bool = False,
                 hint_num_tasks: Optional[int] = None,
                 additional_existing_data: Optional[ExistingData] = None,
                 ) -> Iterator['simmer.SampleStats']:
    """Collects error correction statistics using multiple worker processes.

    Note: if max_batch_size and max_batch_seconds are both not used (or
    explicitly set to None), a default batch-size-limiting mechanism will be
    chosen.

    Args:
        num_workers: The number of worker processes to use.
        tasks: Decoding problems to sample.
        print_progress: When True, progress status is printed to stderr. Uses
            bash escape sequences to erase and rewrite the status as things
            progress.
        hint_num_tasks: If `tasks` is an iterator or a generator, its length
            can be given here so that progress printouts can say how many cases
            are left.
        additional_existing_data: Defaults to None (no additional data).
            Statistical data that has already been collected, in addition to
            anything included in each task's `previous_stats` field.

    Yields:
        simmer.SamplerStats values recording incremental statistical data
        as it is collected by workers.
    """
    if hint_num_tasks is None:
        try:
            # noinspection PyTypeChecker
            hint_num_tasks = len(tasks)
        except TypeError:
            pass

    with CollectionWorkManager(tasks=iter(tasks),
                               additional_existing_data=additional_existing_data) as manager:
        manager.start_workers(num_workers)

        while manager.fill_work_queue():
            # Show status on stderr.
            if print_progress:
                status = manager.status(num_circuits=hint_num_tasks)
                print(status, flush=True, file=sys.stderr, end='')

            # Wait for a worker to finish a job.
            sample = manager.wait_for_next_sample()

            # Erase stderr status message.
            if print_progress:
                erase_current_line = f"\r\033[K"
                erase_previous_line = f"\033[1A" + erase_current_line
                num_prev_lines = len(status.split('\n')) - 1
                print(erase_current_line + erase_previous_line * num_prev_lines + '\033[0m', file=sys.stderr, end='', flush=False)

            yield sample


def collect(*,
            num_workers: int,
            tasks: Union[Iterator['simmer.Task'], Iterable['simmer.Task']],
            existing_data_filepaths: Iterable[Union[str, pathlib.Path]] = (),
            save_resume_filepath: Union[None, str, pathlib.Path] = None,
            progress_callback: Optional[Callable[['simmer.SampleStats'], None]] = None,
            print_progress: bool = False,
            hint_num_tasks: Optional[int] = None,
            ) -> List['simmer.SampleStats']:
    """
    Args:
        num_workers: The number of worker processes to use.
        tasks: Decoding problems to sample.
        save_resume_filepath: Defaults to None (unused). If set to a filepath,
            results will be saved to that file while they are collected. If the
            python interpreter is stopped or killed, calling this method again
            with the same save_resume_filepath will load the previous results
            from the file so it can resume where it left off.

            The stats in this file will be counted in addition to each task's
            previous_stats field (as opposed to overriding the field).
        existing_data_filepaths: CSV data saved to these files will be loaded,
            included in the returned results, and count towards things like
            max_shots and max_errors.
        progress_callback: Defaults to None (unused). If specified, then each
            time new sample statistics are acquired from a worker this method
            will be invoked with the new simmer.SamplerStats.
        print_progress: When True, progress status is printed to stderr. Uses
            bash escape sequences to erase and rewrite the status as things
            progress.
        hint_num_tasks: If `tasks` is an iterator or a generator, its length
            can be given here so that progress printouts can say how many cases
            are left.

    Returns:
        A list of sample statistics, one from each problem. The list is not in
        any specific order. This is the same data that would have been written
        to a CSV file, but aggregated so that each problem has exactly one
        sample statistic instead of potentially multiple.
    """
    # Load existing data.
    additional_existing_data = ExistingData()
    for existing in existing_data_filepaths:
        additional_existing_data += ExistingData.from_file(existing)

    if save_resume_filepath in existing_data_filepaths:
        raise ValueError("save_resume_filepath in existing_data_filepaths")

    with contextlib.ExitStack() as exit_stack:
        # Open save/resume file.
        if save_resume_filepath is not None:
            save_resume_filepath = pathlib.Path(save_resume_filepath)
            if save_resume_filepath.exists():
                additional_existing_data += ExistingData.from_file(save_resume_filepath)
                save_resume_file = exit_stack.enter_context(
                        open(save_resume_filepath, 'a'))
            else:
                save_resume_file = exit_stack.enter_context(
                        open(save_resume_filepath, 'w'))
                print(CSV_HEADER, file=save_resume_file, flush=True)
        else:
            save_resume_file = None

        # Collect data.
        result = ExistingData()
        result.data = dict(additional_existing_data.data)
        for sample in iter_collect(
            num_workers=num_workers,
            tasks=tasks,
            print_progress=print_progress,
            hint_num_tasks=hint_num_tasks,
            additional_existing_data=additional_existing_data,
        ):
            result.add_sample(sample)
            if save_resume_file is not None:
                print(sample.to_csv_line(), file=save_resume_file, flush=True)
            if progress_callback is not None:
                progress_callback(sample)

        return list(result.data.values())


def post_selection_mask_from_last_detector_coords(
        *,
        circuit: stim.Circuit,
        last_coord_minimum: Optional[int]) -> Optional[np.ndarray]:
    lvl = last_coord_minimum
    if lvl is None:
        return None
    coords = circuit.get_detector_coordinates()
    n = circuit.num_detectors + circuit.num_observables
    result = np.zeros(shape=(n + 7) // 8, dtype=np.uint8)
    for k, v in coords.items():
        if len(v) and v[-1] >= lvl:
            result[k >> 3] |= 1 << (k & 7)
    return result