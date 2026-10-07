import os
import yaml
import uuid
from typing import Optional, Literal
from itertools import product
from datetime import datetime as dt

import fire

from src.utils.general_utilities import print_c, ColorType
from src.models.adapters import infer_model_provider


_LOGS_DIR = "./data/task_results"
_JOBS_DIR = "./experiments/jobs"
_MAX_JOB_PER_FILE = 200
_SLEEP_TIME = 0


def create_maze_jobs(
    job_name: str,
    lm_1_name: Optional[list[str]] = None,
    lm_2_name: Optional[list[str]] = None,
    lm_grader_name: list[str] = ["gpt-4.1"],
    lm_1_reasoning_effort: Optional[list[str]] = None,
    lm_2_reasoning_effort: Optional[list[str]] = None,
    lm_grader_reasoning_effort: Optional[list[str]] = None,
    num_samples: int = 1,
    max_turns: Optional[list[int]] = None,
    maze_size: Optional[list[int]] = None,
    min_path_length: Optional[list[int]] = None,
    trap_density: Optional[list[float]] = None,
    wall_density: Optional[list[float]] = None,
    split_1_proportion: Optional[list[float]] = None,
    split_2_proportion: Optional[list[float]] = None,
    split_proportion_pair: Optional[list[float]] = None,
    random_seed: Optional[list[int]] = None,
    game_mode: Literal["collab", "one-shot"] = "collab",
    split_view: list[bool] = [False],
    user_intermediate: list[bool] = [True],
    language_pairs: Optional[list[tuple[str, str]]] = None,
    lm_pair: Optional[list[tuple[str, str]]] = None,
    reasoning_effort_pair: Optional[list[tuple[str, str]]] = None,
    show_same_agent: Optional[list[bool]] = None,
    show_same_is_model_based: Optional[list[bool]] = None,
    show_same_override_value: Optional[list[bool]] = None,
    critic_steps: Optional[list[int]] = None,
    sleep_time: int = 0,
    pre_maze_game_overview_filepath: Optional[list[str]] = None,
    pre_maze_game_filter_maze_size: Optional[list[tuple[int, int]]] = None,
    pre_maze_game_specific_filepath: Optional[list[str]] = None,
    pre_maze_game_filter_success: Optional[list[bool]] = None,
    pre_maze_game_filter_min_outcome: Optional[list[float]] = None,
    pre_maze_game_filter_max_outcome: Optional[list[float]] = None,
    pre_maze_game_force_model_match: Optional[list[bool]] = [True],
    pre_maze_game_filter_lm_pair: Optional[list[tuple[str, str]]] = None,
    pre_maze_game_filter_lm_1_name: Optional[list[str]] = None,
    pre_maze_game_filter_lm_2_name: Optional[list[str]] = None,
    pre_maze_game_filter_split_proportion_pair: Optional[list[tuple[str, str]]] = None,
    pre_maze_game_filter_split_1_proportion: Optional[list[float]] = None,
    pre_maze_game_filter_split_2_proportion: Optional[list[float]] = None,
    pre_maze_game_relay_point: Optional[list[int]] = None,
    pre_maze_game_force_model_match_lm_1: Optional[list[bool]] = None,
    pre_maze_game_force_model_match_lm_2: Optional[list[bool]] = None,
):

    # Create log folder if it doesn't exist
    log_folder = os.path.join(_LOGS_DIR, job_name, "logs")
    os.makedirs(log_folder, exist_ok=True)

    params = locals()

    active_params = {k: v for k, v in params.items(
    ) if isinstance(v, list) and len(v) > 0}

    # If not asymmetric by design, ensure both splits have the same proportion
    if "split_1_proportion" in active_params and "split_2_proportion" not in active_params:
        active_params["split_2_proportion"] = [
            v for v in active_params["split_1_proportion"]]
    if "pre_maze_game_filter_split_1_proportion" in active_params and "pre_maze_game_filter_split_2_proportion" not in active_params:
        active_params["pre_maze_game_filter_split_2_proportion"] = [
            v for v in active_params["pre_maze_game_filter_split_1_proportion"]]
    if "pre_maze_game_filter_split_proportion_pair" in active_params and (
        "pre_maze_game_filter_split_1_proportion" in active_params
        or "pre_maze_game_filter_split_2_proportion" in active_params
    ):
        raise ValueError(
            "Cannot specify both 'pre_maze_game_filter_split_proportion_pair' and 'pre_maze_game_filter_split_1_proportion'/'pre_maze_game_filter_split_2_proportion'. Use 'pre_maze_game_filter_split_proportion_pair' for pairs.")

    if "lm_pair" in active_params and ("lm_1_name" in active_params or "lm_2_name" in active_params):
        raise ValueError(
            "Cannot specify both 'lm_pair' and 'lm_1_name'/'lm_2_name'. Use 'lm_pair' for pairs.")
    if "reasoning_effort_pair" in active_params and (
        "lm_1_reasoning_effort" in active_params
        or "lm_2_reasoning_effort" in active_params
    ):
        raise ValueError(
            "Cannot specify both 'reasoning_effort_pair' and 'lm_1_reasoning_effort'/"
            "'lm_2_reasoning_effort'. Use 'reasoning_effort_pair' for pairs.")
    if "pre_maze_game_filter_lm_pair" in active_params and \
            ("pre_maze_game_filter_lm_1_name" in active_params or "pre_maze_game_filter_lm_2_name" in active_params):
        raise ValueError(
            "Cannot specify both 'pre_maze_game_filter_lm_pair' and 'pre_maze_game_filter_lm_1_name'/'pre_maze_game_filter_lm_2_name'. Use 'pre_maze_game_filter_lm_pair' for pairs.")

    if "pre_maze_game_force_model_match" in active_params and \
            (
                "pre_maze_game_force_model_match_lm_1" in active_params
                or "pre_maze_game_force_model_match_lm_2" in active_params
            ):
        raise ValueError(
            "Cannot specify both 'pre_game_force_model_match' and 'pre_maze_game_force_model_match_lm_1'/'pre_maze_game_force_model_match_lm_2'")

    unique_combinations_with_keys = [
        dict(zip(active_params.keys(), combination))
        for combination in product(*active_params.values())
    ]

    random_seeds = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 79, 80
    ]

    job_count = 0
    jobs = []
    total_jobs_count = len(unique_combinations_with_keys) * num_samples
    per_file_count = min(_MAX_JOB_PER_FILE, total_jobs_count)
    for i, job_params in enumerate(unique_combinations_with_keys):
        job_id = str(uuid.uuid4())[:5]
        job_str = f"python -m src.run_maze_task --job_folder={job_name} --job_id={job_id} \\\n--game_mode={game_mode} \\\n"
        lm1, lm2 = job_params.get('lm_1_name'), job_params.get('lm_2_name')
        pre_lm1, pre_lm2 = job_params.get(
            'pre_maze_game_filter_lm_1_name'), job_params.get('pre_maze_game_filter_lm_2_name')
        for k, v in job_params.items():
            if k == "language_pairs":
                al1 = v[0]
                al2 = v[1]
                job_str += f"--agent_1_language={al1} \\\n--agent_2_language={al2} \\\n"
            elif k == 'lm_pair':
                lm1 = v[0]
                lm2 = v[1]
                job_str += f"--lm_1_name={lm1} \\\n--lm_2_name={lm2} \\\n"
            elif k == 'reasoning_effort_pair':
                re1 = v[0]
                re2 = v[1]
                job_str += f"--lm_1_reasoning_effort={re1} \\\n--lm_2_reasoning_effort={re2} \\\n"
            elif k == 'pre_maze_game_lm_pair':
                pre_lm1 = v[0]
                pre_lm2 = v[1]
                job_str += f"--pre_maze_game_filter_lm_1_name={pre_lm1} \\\n--pre_maze_game_filter_lm_2_name={pre_lm2} \\\n"
            elif k == 'split_proportion_pair':
                sp1 = v[0]
                sp2 = v[1]
                job_str += f"--split_1_proportion={sp1} \\\n--split_2_proportion={sp2} \\\n"
            elif k == 'pre_maze_game_filter_split_proportion_pair':
                sp1 = v[0]
                sp2 = v[1]
                job_str += f"--pre_maze_game_filter_split_1_proportion={sp1} \\\n--pre_maze_game_filter_split_2_proportion={sp2} \\\n"
            else:
                if k == 'maze_size':
                    v = [v, v]
                if isinstance(v, bool):
                    v = str(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    v = f"[{v[0]},{v[1]}]"

                job_str += f"--{k}={v} \\\n"

        if lm1 is None:
            raise ValueError("lm_1_name must be provided")

        # Add model providers.
        job_str += f"--lm_1_provider={infer_model_provider(lm1)} \\\n"
        if lm2:
            job_str += f"--lm_2_provider={infer_model_provider(lm2)} \\\n"
        job_str += f"--lm_grader_provider={infer_model_provider(job_params['lm_grader_name'])} \\\n"

        if pre_lm1:
            job_str += f"--pre_maze_game_filter_lm_1_provider={infer_model_provider(pre_lm1)} \\\n"
        if pre_lm2:
            job_str += f"--pre_maze_game_filter_lm_2_provider={infer_model_provider(pre_lm2)} \\\n"

        # Add log folder for outputs.
        job_count_n = (i * num_samples) - (job_count * _MAX_JOB_PER_FILE)
        n_jobs_str = []
        for n in range(num_samples):
            _job_str_n = job_str + \
                f"--random_seed={random_seeds[n % len(random_seeds)]} "
            # Current [Batch, job_number, sample_number].
            job_log_num = f"{job_count}_{int(job_count_n)}_{n}"
            # Log each job to its own unique log file.
            _job_str_n += f"> {log_folder}/batch_{job_log_num}_{job_id}.log 2>&1"
            _job_str_n = f'\necho "running job: {job_log_num} {job_id} n={n}"\n' + _job_str_n
            _job_str_n += f'\necho "--> [{job_count_n + n + 1: 4d}/{per_file_count}] job completed!"\n'
            n_jobs_str.append(_job_str_n)
        jobs.extend(n_jobs_str)

        if len(jobs) >= _MAX_JOB_PER_FILE:
            if _SLEEP_TIME > 0:
                first_job = jobs[0]
                first_job = first_job.replace("src.run_maze_task",
                                              f"src.run_maze_task --sleep_time={_SLEEP_TIME}")
                jobs[0] = first_job
            save_jobs_to_file(jobs=jobs, sub_folder=job_name,
                              job_name=f"batch_{job_count}.sh")
            jobs = []
            job_count += 1

    if len(jobs) > 0:
        if _SLEEP_TIME > 0:
            first_job = jobs[0]
            first_job = first_job.replace(
                "src.run_maze_task",
                f"src.run_maze_task --sleep_time={_SLEEP_TIME}")
            jobs[0] = first_job
        save_jobs_to_file(jobs=jobs, sub_folder=job_name,
                          job_name=f"batch_{job_count}.sh")

    # Finally, create a run all jobs script. Each batch_*.sh is independent, so
    # they are launched together and waited on; set MAX_PARALLEL_BATCHES to 1 to
    # run them one at a time.
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    job_folder = os.path.join(_JOBS_DIR, job_name)
    script = f"""#!/bin/bash
# created on {timestamp}
# Runs every batch in this folder. Per-job stdout goes to
# ./data/task_results/{job_name}/logs/.
set -uo pipefail
cd "$(dirname "$0")/../../.."

MAX_PARALLEL_BATCHES="${{MAX_PARALLEL_BATCHES:-4}}"
batches=({job_folder}/batch_*.sh)
echo "launching ${{#batches[@]}} batch(es), up to $MAX_PARALLEL_BATCHES at a time"

pids=()
for batch in "${{batches[@]}}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL_BATCHES" ]; do wait -n; done
    echo "  starting $batch"
    bash "$batch" &
    pids+=($!)
done

failed=0
for pid in "${{pids[@]}}"; do
    wait "$pid" || failed=$((failed + 1))
done
if [ "$failed" -ne 0 ]; then
    echo "$failed batch(es) reported an error; check the logs"
    exit 1
fi
echo "all batches finished"
"""
    with open(os.path.join(_JOBS_DIR, job_name, "run_all_jobs.sh"), 'w') as f:
        f.write(script)


def save_jobs_to_file(jobs: list[str], sub_folder: str, job_name: str):

    folder = os.path.join(_JOBS_DIR, sub_folder)
    os.makedirs(folder, exist_ok=True)

    fname = os.path.join(folder, job_name)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")

    jobs_str = "\n".join(jobs)
    # Add shell script header
    jobs_str_ = f"#!/bin/bash\n# Generated on {timestamp}\n"
    jobs_str_ += f"# total jobs in file: {len(jobs)}\n"
    jobs_str_ += jobs_str
    jobs_str_ += '\necho "---> All jobs completed!"\nexit\n'

    with open(fname, 'w') as f:
        f.write(jobs_str_)

    print_c(f"Saved {len(jobs)} jobs to {fname}", ColorType.GREEN)


def main(
    config_file: Optional[str] = None,
    job_name: Optional[str] = None,
    lm_1_name: Optional[list[str]] = None,
    lm_2_name: Optional[list[str]] = None,
    lm_grader_name: list[str] = ["gpt-4.1"],
    num_samples: int = 3,
    max_turns: Optional[list[int]] = None,
    maze_size: Optional[list[int]] = None,
    min_path_length: Optional[list[int]] = None,
    trap_density: Optional[list[float]] = None,
    wall_density: Optional[list[float]] = None,
    split_proportion_pair: Optional[list[float]] = None,
    split_1_proportion: Optional[list[float]] = None,
    split_2_proportion: Optional[list[float]] = None,
    random_seed: Optional[list[int]] = None,
    game_mode: Literal["collab", "one-shot"] = "collab",
    split_view: list[bool] = [False],
    user_intermediate: list[bool] = [True],
    language_pairs: Optional[list[tuple[str, str]]] = None,
    lm_pair: Optional[list[tuple[str, str]]] = None,
    max_jobs_per_file: int = _MAX_JOB_PER_FILE,
    show_same_agent: Optional[list[bool]] = None,
    show_same_is_model_based: Optional[list[bool]] = None,
    show_same_override_value: Optional[list[bool]] = None,
    sleep_time: int = 0,
    critic_steps: Optional[list[int]] = None,
    pre_maze_game_overview_filepath: Optional[list[str]] = None,
    pre_maze_game_specific_filepath: Optional[list[str]] = None,
    pre_maze_game_filter_maze_size: Optional[list[tuple[int, int]]] = None,
    pre_maze_game_filter_success: Optional[list[bool]] = None,
    pre_maze_game_filter_min_outcome: Optional[list[float]] = None,
    pre_maze_game_filter_max_outcome: Optional[list[float]] = None,
    pre_maze_game_filter_lm_pair: Optional[list[tuple[str, str]]] = None,
    pre_maze_game_filter_lm_1_name: Optional[list[str]] = None,
    pre_maze_game_filter_lm_2_name: Optional[list[str]] = None,
    pre_maze_game_filter_lm_1_provider: Optional[list[str]] = None,
    pre_maze_game_filter_lm_2_provider: Optional[list[str]] = None,
    pre_maze_game_filter_split_proportion_pair: Optional[list[tuple[str, str]]] = None,
    pre_maze_game_filter_split_1_proportion: Optional[list[float]] = None,
    pre_maze_game_filter_split_2_proportion: Optional[list[float]] = None,
    pre_maze_game_relay_point: Optional[list[int]] = None,
    pre_maze_game_force_model_match: Optional[list[bool]] = [True],
    pre_maze_game_force_model_match_lm_1: Optional[list[bool]] = None,
    pre_maze_game_force_model_match_lm_2: Optional[list[bool]] = None,
):
    params = locals()

    global _MAX_JOB_PER_FILE
    _MAX_JOB_PER_FILE = max_jobs_per_file

    if sleep_time > 0:
        global _SLEEP_TIME
        _SLEEP_TIME = sleep_time

    if not config_file:
        assert job_name is not None, "job_name must be provided if config_file is not specified"
        assert lm_1_name is not None, "lm_1_name must be provided if config_file is not specified"
        create_maze_jobs(**params)
    else:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        assert isinstance(
            config, dict), "Config file must contain a dictionary"
        assert 'job_name' in config, "Config file must contain 'job_name'"

        create_maze_jobs(**config)


if __name__ == "__main__":
    """Example usage:

python -m src.create_jobs create_maze_jobs \
    --job_name maze_jobs \
    --num_samples 5 \
    --lm_1_name '["gpt-4o"]' \
    --lm_2_name '["gpt-4.1","gemini-2.5-flash"]' \
    --lm_grader_name '["gemini-2.5-flash"]' \
    --max_turns [10] \
    --maze_size [5]

OR: 

python -m src.create_maze_jobs \
--config_file experiments/job_configs/maze_splits.yaml
    """

    fire.Fire(main)
