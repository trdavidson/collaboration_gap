"""
Evaluate strategy features in maze-solving dialogues.
"""

import os
from typing import Optional, Any
from dataclasses import asdict, field, dataclass
from tqdm import tqdm
import yaml
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt

import fire
import pandas as pd

from src.run_maze_task import (
    _FINISHED_GAME_PHRASE,
    _REAL_USER_AGENT_ID,
)
from .models.adapters import (
    _SupportedProviders, LMConfig, OutputAPI, _ReasoningEffort)
from .models.inference import LMFactory
from .utils.prompt_helper import render_template
from .utils.reusable_classes import Conversation
from .utils.general_utilities import set_random_seed
from .utils.experiment_loading import (
    _extract_agent_ids_from_messages,
    _convert_dicts_to_messages,
    get_balanced_sample
)


_MAIN_EVALUATION_FOLDER = "./data/strategy_evaluations"
_STRATEGY_EVALUATION_TEMPLATE = "outcome_strategy_eval.j2"


@dataclass
class StrategyEvaluationConfig:
    """
    Configuration for strategy evaluation experiments.
    """
    save_folder: str
    eval_model_name: str
    eval_model_provider: Optional[_SupportedProviders] = None
    eval_model_reasoning_effort: Optional[_ReasoningEffort] = None
    temperature: float = 0.2
    single_filepath: Optional[str] = None
    existing_sample_path: Optional[str] = None
    experiment_folders: list[str] = field(default_factory=lambda: [])
    sample_size_per_group: Optional[int] = None
    total_sample_size: Optional[int] = None
    groupby_columns: list[str] = field(default_factory=lambda: [])
    # Games shorter than this are dropped before sampling. Distinct from
    # stop_condition_min_turns, which decides where a dialogue is considered
    # to have ended.
    min_rollout_length: int = 8
    filter_conditions: Optional[dict[str, Any]] = None
    enforce_self_play: bool = False
    enforce_cross_play: bool = False
    recheck_stop_condition: bool = True
    stop_condition_min_turns: int = 6
    override_cached_data: bool = False
    enable_relay: bool = False
    max_turns_to_evaluate: Optional[int] = None
    save_results: bool = False
    dryrun: bool = False
    random_seed: Optional[int] = None
    verbosity: bool = False
    report_interesting: bool = True

    def __post_init__(self):
        set_random_seed(self.random_seed, verbose=True)

        if self.save_folder is None:
            self.save_folder = "temp"



@dataclass
class MapAddition:
    spacing: Optional[bool] = False
    borders: Optional[bool] = False
    rows: Optional[bool] = False
    columns: Optional[bool] = False
    other: Optional[bool] = False


@dataclass
class EvaluationResultParsed:
    agent_1_map_sharing: Optional[bool] = None
    agent_2_map_sharing: Optional[bool] = None
    agent_1_map_sharing_format: Optional[str] = None
    agent_2_map_sharing_format: Optional[str] = None
    agent_1_adopts_agent_2_map_sharing_format: Optional[bool] = None
    agent_2_adopts_agent_1_map_sharing_format: Optional[bool] = None
    agent_1_map_sharing_separate_hidden: Optional[bool] = None
    agent_2_map_sharing_separate_hidden: Optional[bool] = None
    agent_1_withheld_map_query_only: Optional[bool] = None
    agent_2_withheld_map_query_only: Optional[bool] = None
    agent_1_map_merging: Optional[bool] = None
    agent_2_map_merging: Optional[bool] = None
    map_reconciliation_timing: Optional[str] = None
    agent_1_map_addition: Optional[MapAddition] = None
    agent_2_map_addition: Optional[MapAddition] = None
    agent_1_grounding_start_goal: Optional[bool] = None
    agent_2_grounding_start_goal: Optional[bool] = None
    agent_1_assumed_origin: Optional[str] = None
    agent_2_assumed_origin: Optional[str] = None
    agents_ground_maps: Optional[bool] = None
    agent_1_map_updating: Optional[bool] = None
    agent_2_map_updating: Optional[bool] = None
    strategy_type: Optional[str] = None
    action_coordination_type: Optional[str] = None
    leadership_shift: Optional[bool] = None
    leadership_shift_direction: Optional[str] = None
    agent_1_move_format: Optional[str] = None
    agent_2_move_format: Optional[str] = None
    agent_1_num_moves_proposed: Optional[int] = None
    agent_2_num_moves_proposed: Optional[int] = None
    encountered_disagreement: Optional[bool] = None
    disagreement_type: Optional[str | dict[str, int]] = None
    ambiguity_handling_heuristic: Optional[str] = None
    disagreement_agent_1_agrees_to_wrong_move: Optional[bool] = None
    disagreement_agent_2_agrees_to_wrong_move: Optional[bool] = None
    disagreement_strategy: Optional[str] = None
    agent_1_proposed_illegal_move: Optional[bool] = None
    agent_2_proposed_illegal_move: Optional[bool] = None
    agent_1_caught_illegal_move: Optional[bool] = None
    agent_2_caught_illegal_move: Optional[bool] = None
    backtracked: Optional[bool] = None
    recognized_fatal_mistake: Optional[bool] = None
    agent_1_theory_of_mind: Optional[bool] = None
    agent_2_theory_of_mind: Optional[bool] = None
    premature_completion_used: Optional[bool] = None
    stuck_in_agreement_loop: Optional[bool] = None
    stuck_in_movement_loop: Optional[bool] = None
    agents_attempted_split_up: Optional[bool] = None
    used_environmental_deduction: Optional[bool] = None
    requested_user_clarification: Optional[bool] = None
    first_move_turn: Optional[int] = None
    interesting_observation: Optional[str] = None

    def to_dict(self, flatten: bool = False) -> dict:
        if flatten:
            result = {}
            for k, v in asdict(self).items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        result[f"{k}_{sub_k}"] = sub_v
                else:
                    result[k] = v
            return result

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResultParsed":
        _cls = cls()
        for k, v in data.items():
            try:
                if k == 'agnet_1_caught_illegal_move':
                    k = 'agent_1_caught_illegal_move'
                elif k == 'agnet_2_caught_illegal_move':
                    k = 'agent_2_caught_illegal_move'
                if k == 'agent_1_map_addition' or k == 'agent_2_map_addition':
                    v = MapAddition(**v) if isinstance(v, dict) else None
                setattr(_cls, k, v)
            except Exception as e:
                print(f"Error setting attribute {k}: {e}")
        return _cls


@dataclass
class EvaluationResult:
    file_path: str
    timestamp: str
    judge_model_name: str
    judge_model_provider: Optional[str]
    lm_1_name: str
    lm_1_provider: str
    lm_2_name: str
    lm_2_provider: str
    evaluation_output_raw: str
    evaluation_data_raw: dict
    evaluation_result_parsed: Optional[EvaluationResultParsed] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_yaml_response(
    response: str
) -> dict:

    try:
        evaluation_dict = yaml.safe_load(response.strip().replace(
            "```yaml", "").replace("```", ""))
    except Exception as e:
        evaluation_dict = {"error": f"Failed to parse YAML: {e}"}

    return evaluation_dict


def _prepare_evaluation_data(
    exp_folder: str,
    recheck_stop_condition: bool = True,
    stop_condition_min_turns: int = 6,
) -> tuple[dict, str, str, Conversation]:

    with open(os.path.join(exp_folder, 'result.yaml'), 'r') as f:
        solution = yaml.safe_load(f)

    agent_id_1, agent_id_2 = _extract_agent_ids_from_messages(
        messages=solution['messages'],
        require_both_agents=True
    )
    messages = _convert_dicts_to_messages(
        solution['messages'],
    )

    if recheck_stop_condition:
        messages_temp = []
        for turn, message in enumerate(messages):
            messages_temp.append(message)
            if (
                turn > stop_condition_min_turns and
                (
                    _FINISHED_GAME_PHRASE.lower() in message.content.lower() or
                    _FINISHED_GAME_PHRASE.replace(
                        "!", "") in message.content
                )
            ):
                break
        messages = messages_temp

    conversation = Conversation(
        messages=messages
    )

    return solution, agent_id_1, agent_id_2, conversation


def evaluate_single_dialogue(
    filepath: str,
    lm_grader_name: str,
    lm_grader_provider: Optional[_SupportedProviders] = None,
    reasoning_effort: Optional[_ReasoningEffort] = None,
    temperature: float = 0.2,
    recheck_stop_condition: bool = True,
    stop_condition_min_turns: int = 6,
    save_result: bool = False,
    verbose: bool = False,
    report_interesting: bool = True
) -> tuple[EvaluationResult, Optional[str]]:

    exp_folder = filepath.replace('result.yaml', '')

    solution, agent_id_1, agent_id_2, conversation = _prepare_evaluation_data(
        exp_folder=exp_folder,
        recheck_stop_condition=recheck_stop_condition,
        stop_condition_min_turns=stop_condition_min_turns
    )

    mapping = {
        agent_id_1: "agent_1",
        agent_id_2: "agent_2",
        _REAL_USER_AGENT_ID: "user",
    }

    dialogue = conversation.as_transcript(
        mapping=mapping,
        show_real_user_message=False,
        show_system_message=False
    )

    prompt = render_template(
        template_name=_STRATEGY_EVALUATION_TEMPLATE,
        user_data={
            "dialogue": dialogue.strip(),
            "report_interesting": report_interesting
        }
    )

    lm = LMFactory(
        model_name=lm_grader_name,
        provider=lm_grader_provider,
        config=LMConfig(
            temperature=temperature,
            reasoning_effort=reasoning_effort)
    )()

    evaluation_response = lm(prompt, num_samples=1, return_as_message=False)
    if isinstance(evaluation_response, list) and len(evaluation_response) == 1:
        evaluation_response = evaluation_response[0]

    evaluation_data = {}
    evaluation_completion = ""
    input_tokens, output_tokens = 0, 0
    if not isinstance(evaluation_response, OutputAPI):
        print(f"Expected an OutputAPI, but got {type(evaluation_response)}")
    else:
        evaluation_data = _extract_yaml_response(
            evaluation_response.completion)
        evaluation_completion = evaluation_response.completion
        input_tokens = evaluation_response.input_tokens
        output_tokens = evaluation_response.output_tokens

    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_filepath = os.path.join(
        exp_folder, f"{timestamp}_strategy_evaluation.yaml")

    if evaluation_data:
        evaluation_object = EvaluationResult(
            file_path=eval_filepath,
            timestamp=timestamp,
            judge_model_name=lm_grader_name,
            judge_model_provider=lm_grader_provider,
            lm_1_name=solution['lm_1_name'],
            lm_1_provider=solution['lm_1_provider'],
            lm_2_name=solution['lm_2_name'],
            lm_2_provider=solution['lm_2_provider'],
            evaluation_output_raw=evaluation_completion,
            evaluation_data_raw=evaluation_data,
            evaluation_result_parsed=EvaluationResultParsed.from_dict(
                evaluation_data
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        if save_result:
            with open(eval_filepath, 'w') as f:
                yaml.safe_dump(
                    asdict(evaluation_object),
                    f,
                    default_flow_style=False,
                    indent=4,
                    sort_keys=False
                )
    else:

        evaluation_object = EvaluationResult(
            file_path=eval_filepath,
            timestamp=timestamp,
            judge_model_name=lm_grader_name,
            judge_model_provider=lm_grader_provider,
            lm_1_name=solution['lm_1_name'],
            lm_1_provider=solution['lm_1_provider'],
            lm_2_name=solution['lm_2_name'],
            lm_2_provider=solution['lm_2_provider'],
            evaluation_output_raw=evaluation_completion,
            evaluation_data_raw=evaluation_data,
            evaluation_result_parsed=None
        )

    if verbose:
        print(f"Evaluating dialogue in {filepath}")
        print(f"---\nPROMPT:\n{prompt}\n---")
        print(
            f"---\nEVALUATION RESPONSE:\n{evaluation_completion}\n---")
        print(
            f"--\n{evaluation_object.lm_1_name}, {evaluation_object.lm_2_name}\n--")
        print(
            f"---\nEVALUATION DATA:\n{evaluation_object.evaluation_result_parsed}\n---")

    return evaluation_object, eval_filepath


def evaluate_dialogue_batch(
    lm_grader_name: str,
    lm_grader_provider: Optional[_SupportedProviders] = None,
    reasoning_effort: Optional[_ReasoningEffort] = None,
    temperature: float = 0.2,
    filepaths: list[str] | str = [],
    max_workers: Optional[int] = 4,
    recheck_stop_condition: bool = True,
    stop_condition_min_turns: int = 6,
    verbose: bool = True,
    report_interesting: bool = True
) -> list[tuple[EvaluationResult, Optional[str]]]:

    if isinstance(filepaths, str):
        filepaths = [filepaths]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(lambda fp: evaluate_single_dialogue(
                filepath=fp,
                lm_grader_name=lm_grader_name,
                lm_grader_provider=lm_grader_provider,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                recheck_stop_condition=recheck_stop_condition,
                stop_condition_min_turns=stop_condition_min_turns,
                verbose=verbose,
                report_interesting=report_interesting
            ), filepaths),
                total=len(filepaths)
            )
        )

    return results


def _process_evaluation_files(
    evaluations: list[EvaluationResult]
) -> pd.DataFrame:

    df_rows = []
    for evaluation in evaluations:
        if evaluation.evaluation_result_parsed is None:
            continue

        row = {
            'judge_model_name': evaluation.judge_model_name,
            'judge_model_provider': evaluation.judge_model_provider,
            'lm_1_name': evaluation.lm_1_name,
            'lm_2_name': evaluation.lm_2_name,
            'lm_1_provider': evaluation.lm_1_provider,
            'lm_2_provider': evaluation.lm_2_provider,
            'folder_id': evaluation.file_path.split('/')[-2],
            'input_tokens': evaluation.input_tokens,
            'output_tokens': evaluation.output_tokens
        }
        row.update(evaluation.evaluation_result_parsed.to_dict(flatten=True))

        df_rows.append(row)

    df_rows = pd.DataFrame(df_rows)

    return df_rows


def main(
    config_file: str,
    max_workers: Optional[int] = 4,
    verbose: bool = False,
    **overrides
):
    """
    example:
    python -m src.strategy_evaluation --config_file experiments/strategy_eval_configs/example_strategy_eval.yaml
    """

    with open(config_file, 'r') as f:
        config_object = StrategyEvaluationConfig(**yaml.safe_load(f))

    if overrides:
        for k, v in overrides.items():
            if hasattr(config_object, k):
                print(
                    f'[override] {k}: {config_object.__getattribute__(k)} --> {v}')
                config_object.__setattr__(k, v)

    df_sample = None
    if config_object.single_filepath is not None:
        filepaths = [config_object.single_filepath]
    elif isinstance(config_object.existing_sample_path, str):
        try:
            df_sample = pd.read_csv(config_object.existing_sample_path)
            filepaths = df_sample['filepath'].tolist()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'can not find sample: {config_object.existing_sample_path}')
    elif config_object.experiment_folders:
        if config_object.enforce_cross_play and config_object.enforce_self_play:
            raise ValueError("Unable to force both cross play and self play")

        df_sample = get_balanced_sample(
            experiment_groups=config_object.experiment_folders,
            sample_size_per_group=config_object.sample_size_per_group,
            groupby_columns=config_object.groupby_columns,
            total_sample_size=config_object.total_sample_size,
            filter_conditions=config_object.filter_conditions,
            override_all_cached_data=config_object.override_cached_data,
            enforce_self_play=config_object.enforce_self_play,
            enforce_cross_play=config_object.enforce_cross_play,
            filter_relays=config_object.enable_relay == False,
            min_rollout_length=config_object.min_rollout_length,
            verbose=verbose
        )
        if df_sample.empty:
            raise ValueError(
                "The sample is empty, so there is nothing to evaluate. Check "
                f"that experiment_folders={config_object.experiment_folders} "
                "contain results, that filter_conditions="
                f"{config_object.filter_conditions} match them, and that the "
                "games are at least min_rollout_length="
                f"{config_object.min_rollout_length} messages long."
            )
        filepaths = df_sample['filepath'].tolist()
    else:
        raise ValueError("No valid input provided.")

    if config_object.dryrun:
        print(f'sample size: {len(filepaths)}')
        return

    results = evaluate_dialogue_batch(
        lm_grader_name=config_object.eval_model_name,
        lm_grader_provider=config_object.eval_model_provider,
        reasoning_effort=config_object.eval_model_reasoning_effort,
        temperature=config_object.temperature,
        filepaths=filepaths,
        max_workers=max_workers,
        recheck_stop_condition=config_object.recheck_stop_condition,
        stop_condition_min_turns=config_object.stop_condition_min_turns,
        verbose=verbose,
        report_interesting=config_object.report_interesting
    )

    if config_object.save_results and df_sample is not None:
        # Create the save folder with a timestamp
        timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        full_save_path = os.path.join(
            _MAIN_EVALUATION_FOLDER, config_object.save_folder, timestamp)
        os.makedirs(full_save_path, exist_ok=True)

        # Save the config used for this run
        with open(os.path.join(full_save_path, 'config.yaml'), 'w') as f:
            yaml.safe_dump(
                asdict(config_object),
                f,
                default_flow_style=False,
                indent=4,
                sort_keys=False
            )
        # Save the sample
        df_sample.to_csv(os.path.join(
            full_save_path, 'sample.csv'), index=False)

        # Save overview of evaluated results
        df_overview = pd.DataFrame(
            {
                'judge_mode': [res[0].judge_model_name for res in results],
                'lm_1_name': [res[0].lm_1_name for res in results],
                'lm_2_name': [res[0].lm_2_name for res in results],
                'lm_1_provider': [res[0].lm_1_provider for res in results],
                'lm_2_provider': [res[0].lm_2_provider for res in results],
                'filepath': [res[1] for res in results]
            }
        )
        df_overview.to_csv(os.path.join(
            full_save_path, 'overview.csv'), index=False)

        # Save the detailed results
        results_path = os.path.join(full_save_path, 'results.csv')
        df_results = _process_evaluation_files(
            [eval_object for (eval_object, _) in results]
        )
        df_results.to_csv(results_path, index=False)
        print(f'-> results saved to: {results_path}')


if __name__ == "__main__":
    fire.Fire(main)
