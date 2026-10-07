import os
from dataclasses import asdict, dataclass, replace as dc_replace
import yaml
from tqdm import tqdm
from typing import Literal, overload, Optional, Any
import functools
from glob import glob
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor
import warnings

import tiktoken
import numpy as np
import pandas as pd

from ..run_maze_task import (
    extract_maze_route, extract_maze_route_one_shot,
    _convert_dicts_to_messages,
    MazeTaskSaveResult
)
from ..models.inference import LMFactory, _SupportedProviders
from .reusable_classes import Conversation
from ..tasks.maze_game import MazeTaskConfig, MazeGame, MazeConfig
from .general_utilities import print_c, ColorType, normalize_deployment_name


@functools.lru_cache(maxsize=1)
def _get_encoder() -> "tiktoken.Encoding":
    """
    The token encoder, loaded on first use.

    tiktoken fetches the encoding the first time it is asked for one. Loading
    it lazily keeps importing this module, and every notebook, from needing
    network access when no token counts are wanted.
    """
    return tiktoken.encoding_for_model("gpt-4o")


def _assign_agent_id_from_messages(
    lm1_name: str,
    lm2_name: str,
    messages: list[dict]
) -> tuple[Optional[str], Optional[str]]:
    """
    Assign agent IDs based on the model names in the messages.
    If the model names are the same, we assign 'agent_1' and 'agent_2' alternatively.
    """
    name_agent_id_pairs = [
        (msg['model_name'], msg.get('agent_id')) for msg in messages if msg['model_name'] is not None]
    agent_id_1, agent_id_2 = None, None
    for model_name, agent_id in name_agent_id_pairs:
        if model_name == lm1_name and agent_id_1 is None:
            agent_id_1 = agent_id
        if model_name == lm2_name and agent_id != agent_id_1 and agent_id_2 is None:
            agent_id_2 = agent_id

    return (agent_id_1, agent_id_2)


def get_message_token_statistics(
    messages: list[dict],
    lm1_name: str,
    lm2_name: str,
    num_pre_turns: int = 0
) -> dict:

    lm1_id, lm2_id = _assign_agent_id_from_messages(
        lm1_name, lm2_name, messages)

    lst_1, lst_2 = [], []
    lst_1_pre, lst_2_pre = [], []
    agent_turn = 0
    for message in messages:
        msg_agent_id = message.get('agent_id', None)
        if msg_agent_id is None:
            # If agent_id is not specified, we assume it's a user message
            # and skip it for token counting.
            continue

        if msg_agent_id != lm1_id and msg_agent_id != lm2_id:
            continue
        agent_turn += 1

        content = message['message']
        if not content:
            continue
        num_tokens = len(_get_encoder().encode(content))

        if agent_turn <= num_pre_turns:
            if msg_agent_id == lm1_id:
                lst_1_pre.append(num_tokens)
            else:
                lst_2_pre.append(num_tokens)
        else:
            if msg_agent_id == lm1_id:
                lst_1.append(num_tokens)
            else:
                lst_2.append(num_tokens)

    stats = {
        f"agent_1_total_tokens": sum(lst_1),
        f"agent_1_median_tokens": np.median(lst_1) if lst_1 else 0,
        f"agent_1_mean_tokens": np.mean(lst_1) if lst_1 else 0,
        f"agent_1_std_tokens": np.std(lst_1) if lst_1 else 0,
        f"agent_2_total_tokens": sum(lst_2),
        f"agent_2_median_tokens": np.median(lst_2) if lst_2 else 0,
        f"agent_2_mean_tokens": np.mean(lst_2) if lst_2 else 0,
        f"agent_2_std_tokens": np.std(lst_2) if lst_2 else 0,
        f"total_tokens": sum(lst_1) + sum(lst_2),
        "median_tokens": np.median(lst_1 + lst_2) if lst_1 + lst_2 else 0,
        f"mean_tokens": np.mean(lst_1 + lst_2) if lst_1 + lst_2 else 0,
        f"std_tokens": np.std(lst_1 + lst_2) if lst_1 and lst_2 else 0,
    }

    if lst_1_pre and lst_2_pre:
        stats.update({
            f"agent_1_pre_turns_total_tokens": sum(lst_1_pre),
            f"agent_1_pre_turns_median_tokens": np.median(lst_1_pre) if lst_1_pre else 0,
            f"agent_1_pre_turns_mean_tokens": np.mean(lst_1_pre) if lst_1_pre else 0,
            f"agent_1_pre_turns_std_tokens": np.std(lst_1_pre) if lst_1_pre else 0,
            f"agent_2_pre_turns_total_tokens": sum(lst_2_pre),
            f"agent_2_pre_turns_median_tokens": np.median(lst_2_pre) if lst_2_pre else 0,
            f"agent_2_pre_turns_mean_tokens": np.mean(lst_2_pre) if lst_2_pre else 0,
            f"agent_2_pre_turns_std_tokens": np.std(lst_2_pre) if lst_2_pre else 0,
            f"total_pre_turns_tokens": sum(lst_1_pre) + sum(lst_2_pre),
            f"median_pre_turns_tokens": np.median(lst_1_pre + lst_2_pre),
            f"mean_pre_turns_tokens": np.mean(lst_1_pre + lst_2_pre),
            f"std_pre_turns_tokens": np.std(lst_1_pre + lst_2_pre),
        })
    else:
        stats.update({
            f"agent_1_pre_turns_total_tokens": 0,
            f"agent_1_pre_turns_median_tokens": 0,
            f"agent_1_pre_turns_mean_tokens": 0,
            f"agent_1_pre_turns_std_tokens": 0,
            f"agent_2_pre_turns_total_tokens": 0,
            f"agent_2_pre_turns_median_tokens": 0,
            f"agent_2_pre_turns_mean_tokens": 0,
            f"agent_2_pre_turns_std_tokens": 0,
            f"total_pre_turns_tokens": 0,
            f"median_pre_turns_tokens": 0,
            f"mean_pre_turns_tokens": 0,
            f"std_pre_turns_tokens": 0,
        })

    return stats


def get_message_word_statistics(
    messages: list[dict],
    lm1_name: str,
    lm2_name: str,
    num_pre_turns: int = 0
) -> dict:

    lm1_id, lm2_id = _assign_agent_id_from_messages(
        lm1_name, lm2_name, messages)

    word_lst_1, word_lst_2 = [], []
    char_lst_1, char_lst_2 = [], []

    word_lst_1_pre, word_lst_2_pre = [], []
    char_lst_1_pre, char_lst_2_pre = [], []

    agent_turn = 0
    for message in messages:
        msg_agent_id = message.get('agent_id', None)
        if msg_agent_id is None:
            # If agent_id is not specified, we assume it's a user message
            # and skip it for word counting.
            continue

        if msg_agent_id != lm1_id and msg_agent_id != lm2_id:
            continue

        agent_turn += 1
        content = message['message']
        if not content:
            continue

        num_words = len(content.split())
        num_characters = len(content)

        if agent_turn <= num_pre_turns:
            if msg_agent_id == lm1_id:
                word_lst_1_pre.append(num_words)
                char_lst_1_pre.append(num_characters)
            else:
                word_lst_2_pre.append(num_words)
                char_lst_2_pre.append(num_characters)
        else:
            if msg_agent_id == lm1_id:
                word_lst_1.append(num_words)
                char_lst_1.append(num_characters)
            else:
                word_lst_2.append(num_words)
                char_lst_2.append(num_characters)

    stats = {
        f"agent_1_total_words": sum(word_lst_1),
        f"agent_1_median_words": np.median(word_lst_1) if word_lst_1 else 0,
        f"agent_1_mean_words": np.mean(word_lst_1) if word_lst_1 else 0,
        f"agent_1_std_words": np.std(word_lst_1) if word_lst_1 else 0,
        f"agent_2_total_words": sum(word_lst_2),
        f"agent_2_median_words": np.median(word_lst_2) if word_lst_2 else 0,
        f"agent_2_mean_words": np.mean(word_lst_2) if word_lst_2 else 0,
        f"agent_2_std_words": np.std(word_lst_2) if word_lst_2 else 0,
        f"total_words": sum(word_lst_1) + sum(word_lst_2),
        f"median_words": np.median(word_lst_1 + word_lst_2) if word_lst_1 + word_lst_2 else 0,
        f"mean_words": (np.mean(word_lst_1) + np.mean(word_lst_2)) / 2 if word_lst_1 and word_lst_2 else 0,
        f"std_words": (np.std(word_lst_1) + np.std(word_lst_2)) / 2 if word_lst_1 and word_lst_2 else 0,
        f"agent_1_total_characters": sum(char_lst_1),
        f"agent_1_median_characters": np.median(char_lst_1) if char_lst_1 else 0,
        f"agent_1_mean_characters": np.mean(char_lst_1) if char_lst_1 else 0,
        f"agent_1_std_characters": np.std(char_lst_1) if char_lst_1 else 0,
        f"agent_2_total_characters": sum(char_lst_2),
        f"agent_2_median_characters": np.median(char_lst_2) if char_lst_2 else 0,
        f"agent_2_mean_characters": np.mean(char_lst_2) if char_lst_2 else 0,
        f"agent_2_std_characters": np.std(char_lst_2) if char_lst_2 else 0,
        f"total_characters": sum(char_lst_1) + sum(char_lst_2),
        f"median_characters": np.median(char_lst_1 + char_lst_2) if char_lst_1 + char_lst_2 else 0,
        f"mean_characters": (np.mean(char_lst_1) + np.mean(char_lst_2)) / 2 if char_lst_1 and char_lst_2 else 0,
        f"std_characters": (np.std(char_lst_1) + np.std(char_lst_2)) / 2 if char_lst_1 and char_lst_2 else 0,
    }
    if word_lst_1_pre and word_lst_2_pre:
        stats.update({
            f"agent_1_pre_turns_total_words": sum(word_lst_1_pre),
            f"agent_1_pre_turns_median_words": np.median(word_lst_1_pre) if word_lst_1_pre else 0,
            f"agent_1_pre_turns_mean_words": np.mean(word_lst_1_pre) if word_lst_1_pre else 0,
            f"agent_1_pre_turns_std_words": np.std(word_lst_1_pre) if word_lst_1_pre else 0,
            f"agent_2_pre_turns_total_words": sum(word_lst_2_pre),
            f"agent_2_pre_turns_median_words": np.median(word_lst_2_pre) if word_lst_2_pre else 0,
            f"agent_2_pre_turns_mean_words": np.mean(word_lst_2_pre) if word_lst_2_pre else 0,
            f"agent_2_pre_turns_std_words": np.std(word_lst_2_pre) if word_lst_2_pre else 0,
            f"agent_1_pre_turns_total_characters": sum(char_lst_1_pre),
            f"agent_1_pre_turns_median_characters": np.median(char_lst_1_pre) if char_lst_1_pre else 0,
            f"agent_1_pre_turns_mean_characters": np.mean(char_lst_1_pre) if char_lst_1_pre else 0,
            f"agent_1_pre_turns_std_characters": np.std(char_lst_1_pre) if char_lst_1_pre else 0,
            f"agent_2_pre_turns_total_characters": sum(char_lst_2_pre),
            f"agent_2_pre_turns_median_characters": np.median(char_lst_2_pre) if char_lst_2_pre else 0,
            f"agent_2_pre_turns_mean_characters": np.mean(char_lst_2_pre) if char_lst_2_pre else 0,
            f"agent_2_pre_turns_std_characters": np.std(char_lst_2_pre) if char_lst_2_pre else 0,
            f"total_pre_turns_words": sum(word_lst_1_pre) + sum(word_lst_2_pre),
            f"median_pre_turns_words": (np.median(word_lst_1_pre) + np.median(word_lst_2_pre)) / 2 if word_lst_1_pre and word_lst_2_pre else 0,
            f"mean_pre_turns_words": (np.mean(word_lst_1_pre) + np.mean(word_lst_2_pre)) / 2 if word_lst_1_pre and word_lst_2_pre else 0,
            f"std_pre_turns_words": (np.std(word_lst_1_pre) + np.std(word_lst_2_pre)) / 2 if word_lst_1_pre and word_lst_2_pre else 0,
            f"total_pre_turns_characters": sum(char_lst_1_pre) + sum(char_lst_2_pre),
            f"median_pre_turns_characters": (np.median(char_lst_1_pre) + np.median(char_lst_2_pre)) / 2 if char_lst_1_pre and char_lst_2_pre else 0,
            f"mean_pre_turns_characters": (np.mean(char_lst_1_pre) + np.mean(char_lst_2_pre)) / 2 if char_lst_1_pre and char_lst_2_pre else 0,
            f"std_pre_turns_characters": (np.std(char_lst_1_pre) + np.std(char_lst_2_pre)) / 2 if char_lst_1_pre and char_lst_2_pre else 0,
        })
    else:
        stats.update({
            f"agent_1_pre_turns_total_words": 0,
            f"agent_1_pre_turns_median_words": 0,
            f"agent_1_pre_turns_mean_words": 0,
            f"agent_1_pre_turns_std_words": 0,
            f"agent_2_pre_turns_total_words": 0,
            f"agent_2_pre_turns_median_words": 0,
            f"agent_2_pre_turns_mean_words": 0,
            f"agent_2_pre_turns_std_words": 0,
            f"agent_1_pre_turns_total_characters": 0,
            f"agent_1_pre_turns_median_characters": 0,
            f"agent_1_pre_turns_mean_characters": 0,
            f"agent_1_pre_turns_std_characters": 0,
            f"agent_2_pre_turns_total_characters": 0,
            f"agent_2_pre_turns_median_characters": 0,
            f"agent_2_pre_turns_mean_characters": 0,
            f"agent_2_pre_turns_std_characters": 0,
            f"total_pre_turns_words": 0,
            f"median_pre_turns_words": 0,
            f"mean_pre_turns_words": 0,
            f"std_pre_turns_words": 0,
            f"total_pre_turns_characters": 0,
            f"median_pre_turns_characters": 0,
            f"mean_pre_turns_characters": 0,
            f"std_pre_turns_characters": 0,
        })

    return stats


def _convert_string_to_bool(value: str) -> Optional[bool]:
    if value is None:
        return None
    elif isinstance(value, str):
        if value.lower().strip() in ['true', '1']:
            return True
        elif value.lower().strip() in ['false', '0']:
            return False
        else:
            return None
    elif isinstance(value, int):
        return value != 0
    elif isinstance(value, bool):
        return value
    else:
        return None


def transform_files_batch_to_dicts(
    files: list[str],
    experiment_file_extension: str = "result.yaml",
    language_subset: Optional[list[str]] = None,
    lm_subset: Optional[list[str]] = None,
    add_messages: bool = False,
    max_workers: Optional[int] = None
) -> list[dict]:

    def _process_file(fname: str) -> dict | None:
        try:
            return transform_single_file_to_dict(
                fname,
                experiment_file_extension=experiment_file_extension,
                language_subset=language_subset,
                lm_subset=lm_subset,
                add_messages=add_messages
            )
        except Exception as e:
            print(f"Error processing file {fname}: {e}")
            return None

    if max_workers is None:
        cpu_count = os.cpu_count() or 2
        max_workers = max(1, cpu_count - 1)  # Leave one core free

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(_process_file, files), total=len(files))
        )

    # Filter out None results
    transformed_dicts = [result for result in results if result is not None]
    return transformed_dicts


def _get_pre_task_config_fields(task_config):
    pre_task_config = task_config.get("pre_maze_game_config", {})

    if not pre_task_config:
        return {}

    row = {}
    for k, v in pre_task_config.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                row[f"pre_task_{k}_{sub_k}"] = sub_v
        else:
            row[f"pre_task_{k}"] = v
    return row


def _get_outcome_fields(outcome: dict, prefix=None) -> dict:

    _prefix = f"{prefix}_" if prefix else ""
    if not outcome:
        return {
            f"{_prefix}outcome_success": None,
            f"{_prefix}outcome_num_steps": None,
            f"{_prefix}outcome_optimal_num_steps": None,
            f"{_prefix}outcome_steps_from_goal": None,
            f"{_prefix}outcome_invalid_reason": None,
            f"{_prefix}outcome_solution_type": None,
            f"{_prefix}outcome_best_position": None,
            f"{_prefix}outcome_path": None,
            f"{_prefix}outcome_parsing_error": None,
            f"{_prefix}weighted_outcome": None,

        }

    row = {}
    for k, v in outcome.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                row[f"{_prefix}{k}_{sub_k}"] = sub_v
        else:
            row[f"{_prefix}{k}"] = v

    if 'weighted_outcome' not in outcome:
        o_steps = outcome.get("outcome_optimal_num_steps", 0)
        steps = outcome.get("outcome_steps_from_goal", None)
        if steps is not None and o_steps > 0:
            per_outcome = (o_steps - steps) / o_steps
        else:
            per_outcome = 0
        if outcome.get("outcome_success", False):
            per_outcome = 1.0
    else:
        per_outcome = outcome['weighted_outcome']

    row[f"{_prefix}weighted_outcome"] = per_outcome

    return row


def get_latest_grading_result(
    fname: str,
    maze_results_dict: Optional[dict] = None,
    judge_model_allow_list: Optional[list[str]] = None
) -> dict:
    grading_data = {}

    if judge_model_allow_list is None:
        judge_model_allow_list = [
            'gpt-4.1_2025-04-14',
            'gemini-2.5-flash',
            'o3_2025-04-16'
        ]

    # Check for updated grading files
    grading_files = sorted(glob(fname))
    if grading_files:
        latest_grading_file = grading_files[-1]
        with open(latest_grading_file, 'r') as f:
            grading_data = yaml.safe_load(f)

        judge_model_name = grading_data.get('judge_model_name')
        if judge_model_name is None or judge_model_name not in judge_model_allow_list:
            grading_data = {}

    if grading_data and maze_results_dict:
        if grading_data.get('updated_messages'):
            maze_results_dict['messages'] = grading_data['updated_messages']
            del grading_data['updated_messages']

        maze_results_dict.update(grading_data)

    return grading_data


def transform_single_file_to_dict(
    fname: str,
    experiment_file_extension: str = "result.yaml",
    language_subset: Optional[list[str]] = None,
    lm_subset: Optional[list[str]] = None,
    add_messages: bool = False,
    use_latest_grading: bool = True
) -> dict | None:

    if not os.path.exists(fname):
        return None

    with open(fname, 'r') as f:
        maze_results_dict = yaml.safe_load(f)

    if use_latest_grading:
        grading_fname = fname.replace(
            experiment_file_extension, "") + "*_grading.yaml"
        _ = get_latest_grading_result(grading_fname, maze_results_dict)

    f_task_config = fname.replace(
        experiment_file_extension, 'task_config.yaml')
    if not os.path.exists(f_task_config):
        # legacy support for old task_config structure
        maze_config = maze_results_dict.get("maze_config", {})
        agent_language_1, agent_language_2 = 'English', 'English'
        random_seed = None
        pre_task_config_fields = {}
    else:
        with open(f_task_config, 'r') as f:
            task_config = yaml.safe_load(f)
        maze_config = task_config.get("maze_config", {})
        agent_language_1 = task_config.get("agent_1_language", 'English')
        agent_language_2 = task_config.get("agent_2_language", 'English')
        random_seed = task_config.get("random_seed", None)

        pre_task_config_fields = _get_pre_task_config_fields(task_config)

    if language_subset is not None:
        if agent_language_1 not in language_subset or agent_language_2 not in language_subset:
            return None

    if lm_subset is not None:
        lm_1_name = maze_results_dict.get('lm_1_name', None)
        lm_2_name = maze_results_dict.get('lm_2_name', None)
        if lm_1_name not in lm_subset or lm_2_name not in lm_subset:
            return None

    num_pre_turns = maze_results_dict.get("num_pre_turns", 0)
    if num_pre_turns is not None:
        num_pre_turns = int(num_pre_turns)
    else:
        num_pre_turns = 0

    show_same_agent_value = _convert_string_to_bool(maze_results_dict.get(
        "show_same_agent_value", None))

    ms = tuple(maze_config["size"])
    lm1 = maze_results_dict['lm_1_name']
    lm2 = maze_results_dict['lm_2_name']

    token_stats = get_message_token_statistics(
        messages=maze_results_dict['messages'], lm1_name=lm1, lm2_name=lm2,
        num_pre_turns=num_pre_turns)
    word_stats = get_message_word_statistics(
        messages=maze_results_dict['messages'], lm1_name=lm1, lm2_name=lm2,
        num_pre_turns=num_pre_turns)

    row = {
        "num_messages": maze_results_dict["num_messages"],
        "lm_1_name": lm1,
        "lm_2_name": lm2,
        "judge_model_name": maze_results_dict.get("judge_model_name", None),
        'maze_size': ms,
        'user_intermediate': _convert_string_to_bool(
            maze_results_dict.get("user_intermediate")),
        'split_view': _convert_string_to_bool(
            maze_results_dict.get("split_view")),
        'split_1_proportion': maze_config.get("split_1_proportion", None),
        'split_2_proportion': maze_config.get("split_2_proportion", None),
        'wall_density': maze_config.get("wall_density", None),
        'trap_density': maze_config.get("trap_density", None),
        'filepath': fname,
        'agent_1_language': agent_language_1,
        'agent_2_language': agent_language_2,
        'num_pre_turns': num_pre_turns,
        'num_messages': maze_results_dict.get("num_messages", 0),
        'random_seed': random_seed,
        "show_same_agent_value": show_same_agent_value,
        "critic_steps": maze_results_dict.get("critic_steps", 0),
        "continuation_filepath": maze_results_dict.get("continuation_filepath", None),
    }
    row.update(token_stats)
    row.update(word_stats)
    row.update(pre_task_config_fields)

    row.update(_get_outcome_fields(
        outcome=maze_results_dict.get("outcome", {})))
    row.update(_get_outcome_fields(outcome=maze_results_dict.get(
        "pre_critic_outcome", {}), prefix="pre_critic"))

    if add_messages:
        row['messages'] = maze_results_dict.get('messages', [])

    return row


def retrieve_experiment_files(
    experiment_groups: list[str] | list[tuple[str, bool]],
    base_folder: str = "./data/task_results/",
    experiment_file_extension: str = "result.yaml",
    verbosity: int = 0
) -> tuple[list[list[str]], list[pd.DataFrame]]:
    """
    Retrieve experiment files and precomputed DataFrames for the given experiment groups.

    Args:
        experiment_groups: A list of experiment group names or tuples of (name, override).
        base_folder: The base folder where experiment results are stored.
        experiment_file_extension: The file extension for experiment result files.
        verbosity: print level

    Returns:
        A tuple containing:
            - A list of lists of experiment file paths. (one list per group)
            - A list of precomputed DataFrames.
    """

    file_batches = []
    precomputed_dfs = []
    for eg in tqdm(experiment_groups):
        eg_override = False
        if isinstance(eg, tuple):
            eg, eg_override = eg
        eg_folder = os.path.join(base_folder, eg)
        if not os.path.exists(eg_folder):
            print(f"Experiment group folder {eg_folder} does not exist.")
            continue

        # Check for pre-computed file
        df_pre = None
        precomputed_filepath = os.path.join(eg_folder, 'precomputed.csv')
        precomputed_filepath_exists = os.path.exists(precomputed_filepath)
        if not eg_override and precomputed_filepath_exists:
            try:
                df_pre = pd.read_csv(precomputed_filepath)
                precomputed_dfs.append(df_pre)
            except Exception as e:
                print(f"Error reading {precomputed_filepath}: {e}")

        # create_maze_jobs writes a logs/ folder per job, but a folder assembled
        # by hand (e.g. data/task_results/example_relay_source) has none. The
        # count is only used for the coverage percentages printed below, so a
        # missing folder must not be fatal.
        logs_folder = os.path.join(eg_folder, 'logs')
        num_log_files = (len(os.listdir(logs_folder))
                         if os.path.isdir(logs_folder) else 0)
        result_files = [os.path.join(eg_folder, f, experiment_file_extension)
                        for f in os.listdir(eg_folder) if f != 'logs' and f != 'precomputed.csv']
        result_files_exist: list[str] = []
        if verbosity > 0:
            print(
                f'{len(result_files): 4d}/{num_log_files: 4d} result files. \t {eg_folder}')

        if len(result_files) > 0:
            result_files_exist = [f for f in result_files if os.path.exists(f)]
            if verbosity > 0:
                if num_log_files:
                    print(
                        f'  -> {len(result_files_exist) / num_log_files: .2%} log files exist')
                print(
                    f'  -> {len(result_files_exist) / len(result_files): .2%} result files exist')
            if precomputed_filepath_exists:
                if verbosity > 0:
                    print(
                        f'  -> precomputed file found: {precomputed_filepath_exists}')
                if df_pre is not None:
                    if verbosity > 0:
                        print(
                            f'  -> {len(df_pre) / len(result_files_exist) if len(result_files_exist) > 0 else len(df_pre): .2%}'
                            ' precomputed of files')
                    if len(result_files_exist) != len(df_pre):
                        print(
                            f'  -> WARNING: {len(result_files_exist) - len(df_pre)} extra result files found! removing cache')
                        df_pre = None

        else:
            print('  -> no result files found')

        # Always append a file batch to keep order the same as the experiment groups
        if df_pre is None:
            file_batches.append(result_files_exist)
        else:
            file_batches.append([])

        total_files = sum([len(fs) for fs in file_batches])
        total_precomputed = sum([len(df) for df in precomputed_dfs])
        if verbosity > 0:
            print(
                f'Total files found to load: {total_files}'
                f'[{total_precomputed} precomputed]')

    return file_batches, precomputed_dfs


@overload
def get_experiment_data(
    experiment_groups: list[str] | list[tuple[str, bool]],
    *,
    return_as_df: Literal[True] = True,
    language_subset: Optional[list[str]] = ...,
    lm_subset: Optional[list[str]] = ...,
    add_messages: bool = ...,
    base_folder: str = ...,
    experiment_file_extension: str = ...,
    override_all: bool = ...
) -> pd.DataFrame: ...


@overload
def get_experiment_data(
    experiment_groups: list[str] | list[tuple[str, bool]],
    *,
    return_as_df: Literal[False],
    language_subset: Optional[list[str]] = ...,
    lm_subset: Optional[list[str]] = ...,
    add_messages: bool = ...,
    base_folder: str = ...,
    experiment_file_extension: str = ...,
    override_all: bool = ...
) -> list[dict]: ...


def get_experiment_data(
    experiment_groups: list[str] | list[tuple[str, bool]],
    language_subset: Optional[list[str]] = None,
    lm_subset: Optional[list[str]] = None,
    add_messages: bool = False,
    return_as_df: bool = True,
    base_folder: str = "./data/task_results/",
    experiment_file_extension: str = "result.yaml",
    override_all: bool = False
) -> pd.DataFrame | list[dict]:

    if override_all:
        experiment_groups = [(eg, True) if not isinstance(
            eg, tuple) else (eg[0], True) for eg in experiment_groups]

    file_batches, precomputed_dfs = retrieve_experiment_files(
        experiment_groups,
        base_folder=base_folder,
        experiment_file_extension=experiment_file_extension)

    if file_batches:
        df_dicts = []
        for egroup, files in zip(experiment_groups, file_batches):
            if not files:
                continue

            egroup_fpath = egroup[0] if isinstance(egroup, tuple) else egroup

            df_dicts_group = transform_files_batch_to_dicts(
                files,
                experiment_file_extension=experiment_file_extension,
                language_subset=language_subset,
                lm_subset=lm_subset,
                add_messages=add_messages)
            df_dicts.extend(df_dicts_group)

            # Save precomputed DataFrame.
            df_egroup = pd.DataFrame(df_dicts_group)
            precomputed_filepath = os.path.join(
                base_folder, egroup_fpath, 'precomputed.csv')
            df_egroup.to_csv(precomputed_filepath, index=False)
            print(f'Precomputed DataFrame saved to: {precomputed_filepath}')

    else:
        df_dicts = []

    if return_as_df:

        dfs = pd.DataFrame(df_dicts)
        if precomputed_dfs:
            dfs = pd.concat([dfs, *precomputed_dfs], ignore_index=True)

        return dfs
    else:
        return df_dicts


def filter_suspect_results(
    df: pd.DataFrame,
    min_num_messages: int,
    inplace: bool = False,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """Filter out suspect results based on message count and outcome presence.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        min_num_messages (int): The minimum number of messages required.
        inplace (bool): Whether to modify the DataFrame in place.
        verbose (bool): Whether to print verbose output.

    Returns:
        pd.DataFrame | None: The filtered DataFrame or None if inplace is True.
    """

    df_size = len(df)

    # check if some minimal number of messages has been sent
    keep_messages = df['num_messages'] >= min_num_messages

    if verbose:
        print(f'Start size: {df_size}')
        print(f"[num messages] Filtered down to {int(keep_messages.sum())} results")

    # check if there was an outcome_path captured
    keep = keep_messages & df['outcome_path'].notna()

    if verbose:
        print(f"[outcome path] Filtered down to {int(keep.sum())} results")

    if inplace:
        # Drop in place rather than rebinding a local name, which would leave
        # the caller's DataFrame untouched.
        df.drop(index=df.index[~keep], inplace=True)
        return None

    return df[keep].copy()


def load_solution_visual(
    f: str,
    show_raw: bool = False,
    print_messages: bool = False
):
    """
    Load the visual representation of a solution from a file.

    Args:
        f (str): The file path to the solution file.
        show_raw (bool): Whether to show the raw extracted route.
        print_messages (bool): Whether to print the messages from the result.

    Returns:
        Tuple: A tuple containing the 
            - str: visual representation of the proposed maze solution
            - MazeOutcome: parsed outcome
            - str: agent(s) used to generate solution
            - Optional[int]: relay point
            - MazeTaskSaveResult: result
    """
    with open(f, 'r') as file:
        sample_result = yaml.safe_load(file)
        sample_result = MazeTaskSaveResult.from_dict(sample_result)
    with open(f.replace('result.yaml', 'task_config.yaml'), 'r') as file:
        sample_config = yaml.safe_load(file)
        sample_config = MazeTaskConfig.from_dict(sample_config)

    maze = MazeGame(
        maze=sample_result.maze_coordinates,
        config=sample_config.config,
        maze_split_1=sample_result.split_1,
        maze_split_2=sample_result.split_2,
    )
    outcome = sample_result.outcome

    s = maze.display_solution(
        solution=outcome.outcome_path,
        maze=maze.maze
    )

    if not sample_result.lm_2_name or sample_result.lm_2_name == 'None':
        agents = sample_result.lm_1_name
    else:
        agents = f"{sample_result.lm_1_name} - {sample_result.lm_2_name}"

    relay_point = None
    if sample_config.pre_maze_game_config is not None:
        relay_point = sample_config.pre_maze_game_config.relay_point

    if show_raw:
        print(sample_result.extracted_route_full)
    if print_messages:
        print("Messages:")
        for i, m in enumerate(sample_result.messages):
            if not m['model_name']:
                continue
            color = ColorType.BLUE if (i % 2 == 0) else ColorType.RED
            print_c(f"[{i+1}] {m['agent_id']}:\n{m['message']}\n", color=color)

    return s, outcome, agents, relay_point, sample_result


@dataclass
class ModelMetaFeatures:
    model: str
    size: str
    thinking: bool
    distilled: bool
    _builder: str
    open: bool

    @functools.cached_property
    def builder(self) -> str:
        builder_to_print_name = {
            'openai': 'OpenAI',
            'google': 'Google',
            'anthropic': 'Anthropic',
            'meta': 'Meta',
            'alibaba': 'Alibaba',
            'cohere': 'Cohere',
            'xai': 'xAI',
            'deepseek': 'DeepSeek',
            'moonshot': 'Moonshot AI',
        }

        return builder_to_print_name.get(self._builder, self._builder)

    @functools.cached_property
    def model_pp(self) -> str:
        return model_name_print_mapping(self.model)


@overload
def get_model_meta_features(
    model: str,
    return_as_dict: Literal[True]
) -> dict:
    ...


@overload
def get_model_meta_features(
    model: str,
    return_as_dict: Literal[False]
) -> ModelMetaFeatures:
    ...


def get_model_meta_features(
    model: str,
    return_as_dict: bool = False,
    verbose: bool = False
) -> dict | ModelMetaFeatures:
    """Get metadata features for a specific model.

    Args:
        model (str): The model name.
        return_as_dict (bool): Whether to return the features as a dictionary.
        verbose (bool): Whether to print verbose messages.

    Returns:
        dict | ModelMetaFeatures: The metadata features for the model.
    """

    if model is None or not isinstance(model, str):
        if verbose:
            print(f'warning: model is None or not a string {model}')
        if return_as_dict:
            return {}
        else:
            raise ValueError(f'Invalid model name: {model}')

    model_to_features_map = {
        'o3_2025-04-16': ModelMetaFeatures(
            model='o3_2025-04-16',
            size='large',
            thinking=True,
            distilled=False,
            _builder='openai',
            open=False
        ),
        'o4-mini_2025-04-16': ModelMetaFeatures(
            model='o4-mini_2025-04-16',
            size='medium',
            thinking=True,
            distilled=True,
            _builder='openai',
            open=False
        ),
        'gpt-5_2025-08-07': ModelMetaFeatures(
            model='gpt-5_2025-08-07',
            size='large',
            thinking=True,
            distilled=False,
            _builder='openai',
            open=False
        ),
        'gpt-5-mini_2025-08-07': ModelMetaFeatures(
            model='gpt-5-mini_2025-08-07',
            size='medium',
            thinking=True,
            distilled=True,
            _builder='openai',
            open=False
        ),
        'gpt-5-nano_2025-08-07': ModelMetaFeatures(
            model='gpt-5-nano_2025-08-07',
            size='small',
            thinking=False,
            distilled=True,
            _builder='openai',
            open=False
        ),
        'gpt-4.1_2025-04-14': ModelMetaFeatures(
            model='gpt-4.1_2025-04-14',
            size='large',
            thinking=False,
            distilled=False,
            _builder='openai',
            open=False
        ),
        'gpt-4.1-mini_2025-04-14': ModelMetaFeatures(
            model='gpt-4.1-mini_2025-04-14',
            size='medium',
            thinking=False,
            distilled=True,
            _builder='openai',
            open=False
        ),
        'gpt-4.1-nano_2025-04-14': ModelMetaFeatures(
            model='gpt-4.1-nano_2025-04-14',
            size='small',
            thinking=False,
            distilled=True,
            _builder='openai',
            open=False
        ),
        'gpt-4.1-nano-2025-04-14': ModelMetaFeatures(
            model='gpt-4.1-nano-2025-04-14',
            size='small',
            thinking=False,
            distilled=True,
            _builder='openai',
            open=False
        ),
        'openai/gpt-oss-120b': ModelMetaFeatures(
            model='openai/gpt-oss-120b',
            size='large',
            thinking=False,
            distilled=False,
            _builder='openai',
            open=True
        ),
        'openai/gpt-oss-20b': ModelMetaFeatures(
            model='openai/gpt-oss-20b',
            size='small',
            thinking=False,
            distilled=True,
            _builder='openai',
            open=True
        ),
        'command-r7b-12-2024': ModelMetaFeatures(
            model='command-r7b-12-2024',
            size='small',
            thinking=False,
            distilled=True,
            _builder='cohere',
            open=True
        ),
        'command-r-08-2024': ModelMetaFeatures(
            model='command-r-08-2024',
            size='medium',
            thinking=False,
            distilled=False,
            _builder='cohere',
            open=True
        ),
        'command-a-03-2025': ModelMetaFeatures(
            model='command-a-03-2025',
            size='large',
            thinking=False,
            distilled=False,
            _builder='cohere',
            open=True
        ),
        'command-a-reasoning-08-2025': ModelMetaFeatures(
            model='command-a-reasoning-08-2025',
            size='large',
            thinking=False,
            distilled=False,
            _builder='cohere',
            open=True
        ),
        'gemma-3-27b-it': ModelMetaFeatures(
            model='gemma-3-27b-it',
            size='medium',
            thinking=False,
            distilled=False,
            _builder='google',
            open=True
        ),
        'gemini-2.5-pro': ModelMetaFeatures(
            model='gemini-2.5-pro',
            size='large',
            thinking=True,
            distilled=False,
            _builder='google',
            open=False
        ),
        'gemini-2.5-flash': ModelMetaFeatures(
            model='gemini-2.5-flash',
            size='medium',
            thinking=True,
            distilled=True,
            _builder='google',
            open=False
        ),
        'gemini-2.5-flash-lite-preview-06-17': ModelMetaFeatures(
            model='gemini-2.5-flash-lite-preview-06-17',
            size='small',
            thinking=False,
            distilled=True,
            _builder='google',
            open=False
        ),
        'Llama-4-Maverick-17B-128E-Instruct-FP8': ModelMetaFeatures(
            model='Llama-4-Maverick-17B-128E-Instruct-FP8',
            size='large',
            thinking=False,
            distilled=False,
            _builder='meta',
            open=True
        ),
        'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8': ModelMetaFeatures(
            model='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
            size='large',
            thinking=False,
            distilled=False,
            _builder='meta',
            open=True
        ),
        'meta-llama/Llama-4-Scout-17B-16E-Instruct': ModelMetaFeatures(
            model='meta-llama/Llama-4-Scout-17B-16E-Instruct',
            size='medium',
            thinking=False,
            distilled=True,
            _builder='meta',
            open=True
        ),
        'DeepSeek-R1-0528': ModelMetaFeatures(
            model='DeepSeek-R1-0528',
            size='large',
            thinking=True,
            distilled=False,
            _builder='deepseek',
            open=True
        ),
        'DeepSeek-R1': ModelMetaFeatures(
            model='DeepSeek-R1',
            size='large',
            thinking=True,
            distilled=False,
            _builder='deepseek',
            open=True
        ),
        'grok-4': ModelMetaFeatures(
            model='grok-4',
            size='large',
            thinking=True,
            distilled=False,
            _builder='xai',
            open=False
        ),
        'grok-3': ModelMetaFeatures(
            model='grok-3',
            size='large',
            thinking=True,
            distilled=False,
            _builder='xai',
            open=False
        ),
        'grok-3-mini': ModelMetaFeatures(
            model='grok-3-mini',
            size='small',
            thinking=True,
            distilled=False,
            _builder='xai',
            open=False
        ),
        'claude-opus-4-1-20250805': ModelMetaFeatures(
            model='claude-opus-4-1-20250805',
            size='large',
            thinking=True,
            distilled=False,
            _builder='anthropic',
            open=False
        ),
        'claude-opus-4-20250514': ModelMetaFeatures(
            model='claude-opus-4-20250514',
            size='large',
            thinking=True,
            distilled=False,
            _builder='anthropic',
            open=False
        ),
        'claude-sonnet-4-20250514': ModelMetaFeatures(
            model='claude-sonnet-4-20250514',
            size='medium',
            thinking=True,
            distilled=False,
            _builder='anthropic',
            open=False
        ),
        'claude-3-7-sonnet-20250219': ModelMetaFeatures(
            model='claude-3-7-sonnet-20250219',
            size='medium',
            thinking=False,
            distilled=False,
            _builder='anthropic',
            open=False
        ),
        'claude-3-5-haiku-20241022': ModelMetaFeatures(
            model='claude-3-5-haiku-20241022',
            size='small',
            thinking=True,
            distilled=False,
            _builder='anthropic',
            open=False
        ),
        'moonshotai/Kimi-K2-Instruct': ModelMetaFeatures(
            model='moonshotai/Kimi-K2-Instruct',
            size='large',
            thinking=True,
            distilled=False,
            _builder='moonshot',
            open=True
        ),
        'Qwen/Qwen3-235B-A22B-fp8-tput': ModelMetaFeatures(
            model='Qwen/Qwen3-235B-A22B-fp8-tput',
            size='large',
            thinking=True,
            distilled=False,
            _builder='alibaba',
            open=True
        ),
        'Qwen/Qwen2.5-72B-Instruct-Turbo': ModelMetaFeatures(
            model='Qwen/Qwen2.5-72B-Instruct-Turbo',
            size='medium',
            thinking=False,
            distilled=False,
            _builder='alibaba',
            open=True
        ),
    }

    meta_features = model_to_features_map.get(model)
    if meta_features is None:
        # Some table keys write the date with an underscore. Match on the
        # canonical spelling so one entry serves both forms.
        wanted = normalize_deployment_name(model)
        for key, features in model_to_features_map.items():
            if normalize_deployment_name(key) == wanted:
                meta_features = features
                break

    if meta_features is None:
        if verbose:
            print(f'warning: model not found {model}')
        if return_as_dict:
            return {}
        else:
            raise ValueError(f'model not found {model}')

    # The entry carries the table's own spelling of the name. Callers merge on
    # this value (see add_model_features), so hand back the name that was asked
    # for or the join silently produces nulls.
    if meta_features.model != model:
        meta_features = dc_replace(meta_features, model=model)

    if return_as_dict:
        return asdict(meta_features)

    return meta_features


def model_name_print_mapping(
    model: str,
    reverse: bool = False,
    verbose: bool = False
) -> str:
    """Remap model names to a more readable format.

    Args:
        model (str): The original model name.
    Returns:
        str: The remapped model name.
    """

    if model is None or not isinstance(model, str):
        if verbose:
            print(f'warning: model is None or not a string {model}')
        return model

    print_map = {
        'o3_2025-04-16': 'o3',
        'o4-mini_2025-04-16': 'o4-mini',
        'gpt-4.1_2025-04-14': 'gpt-4.1',
        'gpt-4.1-mini_2025-04-14': 'gpt-4.1-mini',
        'gpt-4.1-nano_2025-04-14': 'gpt-4.1-nano',
        'gpt-4.1-nano-2025-04-14': 'gpt-4.1-nano',
        'gpt-5_2025-08-07': 'gpt-5',
        'gpt-5-mini_2025-08-07': 'gpt-5-mini',
        'gpt-5-nano_2025-08-07': 'gpt-5-nano',
        'gemini-2.5-pro': 'gemini-2.5-pro',
        'gemini-2.5-flash': 'gemini-2.5-flash',
        'gemini-2.5-flash-lite-preview-06-17': 'gemini-2.5-flash-lite',
        'gemma-3-27b-it': 'gemma-3-27b-it',
        'command-a-reasoning-08-2025': 'command-a-reasoning',
        'command-a-03-2025': 'command-a',
        'command-r-08-2024': 'command-r',
        'command-r7b-12-2024': 'command-r7b',
        'Llama-4-Maverick-17B-128E-Instruct-FP8': 'llama-4-maverick',
        'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8': 'llama-4-maverick',
        'meta-llama/Llama-4-Scout-17B-16E-Instruct': 'llama-4-scout',
        'openai/gpt-oss-120b': 'gpt-oss-120b',
        'openai/gpt-oss-20b': 'gpt-oss-20b',
        'DeepSeek-R1-0528': 'deepseek-R1',
        'DeepSeek-R1': 'deepseek-R1',
        'grok-4': 'grok-4',
        'grok-3': 'grok-3',
        'grok-3-mini': 'grok-3-mini',
        'claude-opus-4-1-20250805': 'claude-opus-4.1',
        'claude-opus-4-20250514': 'claude-opus-4',
        'claude-sonnet-4-20250514': 'claude-sonnet-4',
        'claude-3-7-sonnet-20250219': 'claude-sonnet-3.7',
        'claude-3-5-haiku-20241022': 'claude-haiku-3.5',
        'moonshotai/Kimi-K2-Instruct': 'kimi-k2',
        'Qwen/Qwen3-235B-A22B-fp8-tput': 'qwen-3-235B',
        'Qwen/Qwen2.5-72B-Instruct-Turbo': 'qwen-2.5-72B-it',
    }

    if reverse:
        print_map = {v: k for k, v in print_map.items()}

    remapped = print_map.get(model)
    if remapped is None:
        if verbose:
            print(f'warning! model not found {model} -- returning input')
        remapped = model

    return remapped


def add_model_features(
    df: pd.DataFrame,
    join_col_1: str = "lm_1_name",
    join_col_2: Optional[str] = None,
) -> pd.DataFrame:

    lm_1_models = df[join_col_1].unique().tolist()

    lm_2_models = []
    if join_col_2 is not None:
        lm_2_models = df[join_col_2].unique().tolist()

    lm_1_features = [
        get_model_meta_features(model, return_as_dict=True) for model in lm_1_models
    ]

    lm_2_features = [
        get_model_meta_features(model, return_as_dict=True) for model in lm_2_models
    ]

    # get_model_meta_features returns {} for a model that is not in the table, so
    # a frame built from those rows has no columns to join on. Drop the empty
    # ones first and skip the merge entirely if nothing resolved, rather than
    # failing on a missing join key.
    lm_1_features = [f for f in lm_1_features if f]
    lm_2_features = [f for f in lm_2_features if f]

    if lm_1_features:
        df_features = pd.DataFrame(lm_1_features)
        # Add prefix "lm_1_" to each feature except "model"
        df_features = df_features.rename(
            columns=lambda x: f"lm_1_{x}" if x != "model" else join_col_1)
        df = df.merge(df_features, on=join_col_1, how="left")

    if lm_2_features:
        df_features = pd.DataFrame(lm_2_features)
        # Add prefix "lm_2_" to each feature except "model"
        df_features = df_features.rename(
            columns=lambda x: f"lm_2_{x}" if x != "model" else join_col_2)
        df = df.merge(df_features, on=join_col_2, how="left")

    return df


def add_derived_columns(df):
    """
    Add the columns every analysis needs to results loaded by
    get_experiment_data, in place: model pairs (lm_pair, lm_pair_unique),
    display names (*_pp), parsed maze_size, and defaults for older results.
    """
    df['maze_size'] = df['maze_size'].apply(
        lambda x: eval(x) if isinstance(x, str) else x)

    if 'num_pre_turns' not in df.columns:
        df['num_pre_turns'] = 0
    df['num_pre_turns'] = df['num_pre_turns'].astype(float).astype(int)
    order = sorted(df['num_pre_turns'].unique().tolist())
    df['num_pre_turns'] = pd.Categorical(
        df['num_pre_turns'], categories=order, ordered=True)

    def _format_long_pair(pair):
        return f"\n{pair[0]}\n{pair[1]}\n"

    df['lm_1_name'] = df['lm_1_name'].fillna('NA')
    df['lm_2_name'] = df['lm_2_name'].fillna('NA')
    df['lm_pair'] = df['lm_1_name'] + ' - ' + df['lm_2_name']
    df['lm_pair_unique'] = df.apply(lambda x: _format_long_pair(
        sorted([x['lm_1_name'], x['lm_2_name']])), axis=1)

    df['num_pre_turns'] = df['num_pre_turns'].fillna(0)
    df['num_pre_turns'] = df['num_pre_turns'].astype(float).astype(int)
    if "show_same_agent_value" not in df.columns:
        df['show_same_agent_value'] = None
    df['show_same_agent_value'] = df['show_same_agent_value'].astype(str)
    df['same_value_visible'] = df['show_same_agent_value'].astype(
        str).apply(lambda x: True if (x.lower() == 'true' or x.lower() == 'false') else False)

    df['agent_1_language'] = df['agent_1_language'].fillna('English')
    df['agent_2_language'] = df['agent_2_language'].fillna('English')

    if "pre_task_lm_1_name" in df.columns and "pre_task_lm_2_name" in df.columns:
        df['pre_task_lm_1_name'] = df['pre_task_lm_1_name'].fillna('NA')
        df['pre_task_lm_2_name'] = df['pre_task_lm_2_name'].fillna('NA')
        df['pre_task_lm_pair'] = df['pre_task_lm_1_name'] + \
            ' - ' + df['pre_task_lm_2_name']
        df['pre_task_lm_pair_unique'] = df.apply(
            lambda x: _format_long_pair(
                sorted([x['pre_task_lm_1_name'], x['pre_task_lm_2_name']])),
            axis=1
        )

    df['lm_1_name_pp'] = df['lm_1_name'].apply(model_name_print_mapping)
    df['lm_2_name_pp'] = df['lm_2_name'].apply(model_name_print_mapping)
    df['lm_pair_pp'] = df['lm_1_name_pp'] + ' - ' + df['lm_2_name_pp']
    df['lm_pair_unique_pp'] = df.apply(lambda x: _format_long_pair(
        sorted([x['lm_1_name_pp'], x['lm_2_name_pp']])), axis=1)


def get_model_ordering(
    df,
    ordering_type: Literal["micro", "macro"] = "macro",
    order_col='weighted_outcome',
    panel_cols=['split_1_proportion'],
    single_model: bool = False,
    lm_1_key: str = 'lm_1_name',
    lm_2_key: str = 'lm_2_name',
    lm_pair_key: str = 'lm_pair',
) -> tuple[list[str], pd.DataFrame]:
    """ Get the model ordering based on the specified ordering type.

    This is useful for nicely displaying the results in a table.

    Args:
        df (pd.DataFrame): The DataFrame containing the experiment results.
        ordering_type (str): The type of ordering to use. Can be 'micro' or 'macro'.
            - 'micro': Orders models based on the mean of the specified order_col.
            - 'macro': Orders models based on the mean of the specified order_col
              grouped by lm_pair and panel_cols.
        order_col (str): The column to use for ordering the models.
        panel_cols (list): The columns to use for grouping in 'macro' ordering.
        single_model (bool): Whether to treat the experiment as a single model.
        lm_1_key (str): The key to use for the first language model.
        lm_2_key (str): The key to use for the second language model.
        lm_pair_key (str): The key to use for the language model pair.
    """

    _df = df.copy()
    if single_model:
        _df[lm_pair_key] = _df[lm_1_key]

    models = _df[lm_1_key].unique().tolist()
    results = []
    for model in models:
        df_model = _df[(
            (_df[lm_1_key] == model) |
            (_df[lm_2_key] == model))].copy()

        if ordering_type == "micro":
            model_result = df_model[order_col].agg(
                ['mean', 'sem']).reset_index().rename(columns={order_col: 'mean'})
        elif ordering_type == "macro":
            model_result = df_model.groupby(
                [lm_pair_key] + panel_cols)[order_col].agg(['mean', 'sem']).reset_index()
            if panel_cols:
                model_result = df_model.groupby(panel_cols)[order_col].agg([
                    'mean', 'sem']).reset_index()
        else:
            raise ValueError("ordering_type must be either 'micro' or 'macro'")
        model_result['model'] = model
        results.append(model_result)

    results = pd.concat(results, ignore_index=True)
    ordered_models = results.groupby('model')['mean'].mean(
    ).sort_values(ascending=False).index.tolist()

    return ordered_models, results


def get_balanced_sample(
    experiment_groups: list[str],
    sample_size_per_group: Optional[int] = None,
    groupby_columns: Optional[list[str]] = None,
    total_sample_size: Optional[int] = None,
    filter_conditions: Optional[dict[str, Any]] = None,
    enforce_self_play: bool = False,
    enforce_cross_play: bool = False,
    override_all_cached_data: bool = False,
    filter_relays: bool = True,
    min_rollout_length: int = 8,
    verbose: bool = False
):
    """
    Get a balanced sample from the experiment data.

    Args:
        experiment_groups (list[str]): The list of experiment group identifiers.
        sample_size_per_group (int, optional): The number of samples to draw from each group.
        groupby_columns (list[str], optional): The columns to group by when sampling.
        total_sample_size (int, optional): The total number of samples to draw.
        filter_conditions (dict[str, Any], optional): Conditions to filter the DataFrame before sampling.
        enforce_self_play (bool): lm_1 must be the same as lm_2
        enforce_cross_play (bool): lm_1 must NOT be the same as lm_2
        override_all_cached_data (bool, optional): Whether to override all cached data.
        verbose (bool, optional): Whether to print verbose output.

    Returns:
        pd.DataFrame: A balanced sample from the experiment data.
    """

    if enforce_self_play and enforce_cross_play:
        raise ValueError("Cannot simultaneously enforce self- and cross-play!")

    df = get_experiment_data(
        experiment_groups,
        language_subset=None,
        add_messages=False,
        return_as_df=True,
        override_all=override_all_cached_data,
    )

    add_derived_columns(df)

    if groupby_columns is None:
        groupby_columns = []

    if enforce_self_play:
        print(f'Enforcing self-play only!')
        df = df[df['lm_1_name'] == df['lm_2_name']].reset_index().copy()
    if enforce_cross_play:
        print(f'Enforcing cross-play only!')
        df = df[df['lm_1_name'] != df['lm_2_name']].reset_index().copy()

    if filter_relays:
        print(f'Enforcing no relays!')
        if 'pre_task_relay_point' in df.columns:
            df.fillna({"pre_task_relay_point": 0}, inplace=True)
            df = df[df['pre_task_relay_point'] == 0].copy()
            print('no relay left: ', len(df))
        else:
            df['pre_task_relay_point'] = 0

    df = filter_suspect_results(
        df, min_num_messages=min_rollout_length, inplace=False)
    if df is None:
        raise ValueError(
            "no valid data left after filtering for suspect results")

    if filter_conditions is not None:
        if verbose:
            print(f"Applying filter conditions to the DataFrame...({len(df)})")
        for col, values in filter_conditions.items():
            len_prior = len(df)
            if col not in df.columns:
                warnings.warn(
                    f"Filter condition column '{col}' not found in the DataFrame. "
                    f"Available columns: {df.columns.tolist()}"
                )
                continue

            if col == "maze_size":
                df = df[df[col] == eval(values)].copy()
            elif isinstance(values, list):
                df = df[df[col].isin(values)].copy()
            else:
                df = df[df[col] == values].copy()

            if verbose:
                print(
                    f'-- {col} --> {values}: prior: {len_prior}, after: {len(df)}')
        if verbose:
            print(f"Filtered DataFrame size: {len(df)}")

    # Check if all columns exist in the DataFrame
    for col in groupby_columns:
        if col not in df.columns:
            raise ValueError(
                f"Groupby column '{col}' not found in the DataFrame. "
                f"Available columns: {df.columns.tolist()}"
            )

    # Get sample that samples K rows from each groupby bucket
    if not groupby_columns:
        if not total_sample_size:
            raise ValueError(
                "If groupby_columns is empty, total_sample_size must be specified."
            )
        sample = df.sample(
            min(total_sample_size, len(df)),
            replace=False
        ).reset_index(drop=True)
    else:
        if not sample_size_per_group:
            raise ValueError(
                "If groupby_columns is specified, sample_size_per_group must be specified."
            )
        # Sample each group explicitly and concatenate, so the grouping columns
        # stay in the sample: the downstream ablation needs them, and
        # groupby(...).apply(..., include_groups=False) would drop them.
        sampled_groups = [
            group.sample(min(sample_size_per_group, len(group)), replace=False)
            for _, group in df.groupby(groupby_columns, observed=True)
        ]
        sample = pd.concat(sampled_groups).reset_index(drop=True)

        if total_sample_size is not None and len(sample) > total_sample_size:
            original_size = len(sample)
            sample = sample.sample(total_sample_size)
            warnings.warn(
                f"Sample size exceeded {total_sample_size}, downsampling applied: {original_size} -> {total_sample_size}")

    # Set the folder_id of the result to join on later
    sample['folder_id'] = sample['filepath'].apply(
        lambda file: file.split('/')[-2])

    if verbose:
        print(f"Original df size: {len(df)}")
        print(f"Sample size: {len(sample)}")
        print(
            f"Number of groups: {df.groupby(groupby_columns).ngroups}")

    return sample


@overload
def _extract_agent_ids_from_messages(
    messages: list[dict],
    require_both_agents: Literal[True] = True
) -> tuple[str, str]: ...


@overload
def _extract_agent_ids_from_messages(
    messages: list[dict],
    require_both_agents: Literal[False] = False
) -> tuple[str, Optional[str]]: ...


def _extract_agent_ids_from_messages(
    messages: list[dict],
    require_both_agents: bool = True
) -> tuple[str, str] | tuple[str, Optional[str]]:
    """Extract agent IDs from the message list."""
    agent_1_id = None
    agent_2_id = None
    for msg in messages:

        if agent_1_id and agent_2_id:
            break

        if not msg['model_name']:
            continue

        agent_id = msg.get('agent_id', None)
        if not agent_1_id:
            agent_1_id = agent_id
        elif not agent_2_id and agent_id != agent_1_id:
            agent_2_id = agent_id

    if require_both_agents and (not agent_1_id or not agent_2_id):
        raise ValueError(
            "Both agent IDs must be present in the messages. "
            f"Found: agent_1_id={agent_1_id}, agent_2_id={agent_2_id}"
        )

    if not agent_1_id:
        raise ValueError("No agent_1_id found in the messages.")

    return agent_1_id, agent_2_id


def _prepare_grading_data(
    exp_folder: str,
    game_mode: Optional[str] = "collab"
) -> tuple[MazeGame, dict, str, Optional[str], Conversation]:

    with open(os.path.join(exp_folder, 'result.yaml'), 'r') as f:
        solution = yaml.safe_load(f)
    with open(os.path.join(exp_folder, 'task_config.yaml'), 'r') as f:
        task_config = yaml.safe_load(f)

    agent_id_1, agent_id_2 = _extract_agent_ids_from_messages(
        messages=solution['messages'],
        require_both_agents=solution.get('game_mode', game_mode) == 'collab'
    )
    conversation = Conversation(
        messages=_convert_dicts_to_messages(
            solution['messages'],
        )
    )

    # create maze object
    maze_config = MazeConfig.from_config(task_config['maze_config'])
    maze = MazeGame(
        config=maze_config,
        maze=solution['maze_coordinates'],
        maze_split_1=solution.get('split_1', None),
        maze_split_2=solution.get('split_2', None),
    )

    return maze, solution, agent_id_1, agent_id_2, conversation


def update_grading_for_file(
    filepath: str,
    lm_grader_name: str,
    lm_grader_provider: _SupportedProviders,
    game_mode: Optional[str] = "collab",
    verbose: bool = False
) -> str:

    lm_grader = LMFactory(
        model_name=lm_grader_name,
        provider=lm_grader_provider).create_instance()

    exp_folder = filepath.replace('result.yaml', '')
    (
        maze,
        solution,
        agent_id_1,
        agent_id_2,
        conversation
    ) = _prepare_grading_data(exp_folder=exp_folder)

    if verbose:
        print(f"Grading file for {exp_folder} with game mode {game_mode}")
    game_mode = solution.get('game_mode', game_mode)
    if game_mode == 'one-shot':
        extracted_solution = extract_maze_route_one_shot(
            lm_grader=lm_grader,
            data=solution['messages']
        )
    elif game_mode == 'collab':
        extracted_solution = extract_maze_route(
            lm_grader=lm_grader,
            lm_1_id=agent_id_1,
            lm_2_id=agent_id_2,
            conversation=conversation
        )
    else:
        raise NotImplementedError(
            f"Game mode {game_mode} is not supported for grading."
        )
    if verbose:
        print('--verifying solution')
    updated_outcome = maze.verify_solution(
        solution=extracted_solution.processed_route,  # type: ignore
        solution_type=extracted_solution.route_schema
    )

    # Add a new grading file to the experiment folder
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    grading_file_path = os.path.join(exp_folder, f"{timestamp}_grading.yaml")

    grading_file = {
        "judge_model_name": lm_grader.adapter.model_name,
        "judge_model_provider": lm_grader.adapter.provider.value,
        "timestamp": timestamp,
        "extracted_route_full": extracted_solution.route,
        "extracted_route_clean": extracted_solution.processed_route,
        "extracted_route_schema": (
            extracted_solution.route_schema.to_dict()
            if extracted_solution.route_schema else None
        ),
        "outcome": updated_outcome.to_dict(),
    }

    if verbose:
        print(f'--saving grading file to {grading_file_path}')

    with open(grading_file_path, 'w') as f:
        yaml.safe_dump(
            grading_file,
            f,
            default_flow_style=False,
            indent=4,
            sort_keys=False
        )

    if verbose:
        print("--done!")

    return grading_file_path


def update_grading_for_batch(
    filepaths: list[str],
    lm_grader_name: str,
    lm_grader_provider: _SupportedProviders,
    max_workers=5,
    game_mode: Optional[str] = "collab",
    verbose: bool = False
) -> list[str]:
    """Update the grading for a batch of experiment files.

    Returns:
        list[str]: A list of filepaths to the newly created grading files.
    """

    # Grade files concurrently.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(lambda fp: update_grading_for_file(
                fp,
                lm_grader_name=lm_grader_name,
                lm_grader_provider=lm_grader_provider,
                game_mode=game_mode,
                verbose=verbose
            ), filepaths),
                total=len(filepaths)
            )
        )
    return results


