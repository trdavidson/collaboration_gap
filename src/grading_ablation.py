"""
Script and methods to ablate different grading models.

This module provides functionality to:
- Configure and run grading ablation experiments
- Analyze and interpret the results of these experiments
"""

import os
from typing import Optional, Literal, Any
from dataclasses import asdict, field, dataclass
from tqdm import tqdm
import yaml
from datetime import datetime as dt
import itertools
import warnings
from concurrent.futures import ThreadPoolExecutor

import fire
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, wilcoxon, ttest_rel, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.multitest import multipletests

from .tasks.maze_game import MazeGame, MazeConfig, MazeOutcome, _MazeTypePath
from src.run_maze_task import (
    extract_maze_route,
    extract_maze_route_one_shot,
    _FINISHED_GAME_PHRASE,
    conversation_to_dicts
)
from .models.adapters import (
    _SupportedProviders, LMConfig, _ReasoningEffort)
from .models.inference import LMFactory
from .utils.prompt_helper import render_template
from .utils.reusable_classes import Conversation, UserMessage, AssistantMessage
from .utils.general_utilities import set_random_seed
from .utils.experiment_loading import (
    get_balanced_sample,
    _extract_agent_ids_from_messages,
    _convert_dicts_to_messages,
    _get_outcome_fields
)


_MAIN_GRADING_FOLDER = "./data/grading_ablations"
_GRADING_INTERPRETATION_TEMPLATE = "grading_analyses_interpretation.j2"


@dataclass
class GradingConfig:
    """
    Configuration for grading ablation experiments.
    """
    save_folder: str
    existing_sample_file: Optional[str] = None
    config_type: Literal[
        "get_grades",
        "analyses",
        "ablation",
        "update_verification"] = "get_grades"
    experiment_folders: list[str] = field(default_factory=lambda: [])
    judge_models: list[str] = field(default_factory=lambda: [])
    sample_size_per_group: Optional[int] = None
    total_sample_size: Optional[int] = None
    groupby_columns: list[str] = field(default_factory=lambda: [])
    # Games shorter than this are dropped before sampling. Raise it to exclude
    # rollouts that ended almost immediately; lower it when grading short games.
    min_rollout_length: int = 8
    temperature: float = 0.2
    reasoning_effort: Optional[_ReasoningEffort] = None
    analyses_to_perform: list[Literal[
        "self_consistency",
        "inter_consistency",
    ]] = field(default_factory=lambda: [
        "self_consistency",
        "inter_consistency"
    ])
    filter_conditions: Optional[dict[str, Any]] = None
    random_seed: Optional[int] = None
    summary_interpretation_model_name: Optional[str] = None

    def __post_init__(self):
        set_random_seed(self.random_seed, verbose=True)

        if self.config_type not in ["get_grades", "analyses", "ablation"]:
            raise ValueError(
                f"Invalid config_type: {self.config_type}. "
                f"Must be one of ['get_grades', analyses, 'ablation']")

        if self.save_folder is None:
            self.save_folder = _MAIN_GRADING_FOLDER

        # Ensure the save folder exists
        os.makedirs(self.save_folder, exist_ok=True)

        if self.config_type == "get_grades":
            if not self.experiment_folders:
                raise ValueError(
                    "experiment_folders must be provided for 'get_grades' config_type.")
            if not self.judge_models:
                raise ValueError(
                    "judge_models must be provided for 'get_grades' config_type.")
            if self.sample_size_per_group is None:
                raise ValueError(
                    "sample_size_per_group must be specified.")

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GradingConfig':
        """
        Update the configuration from a dictionary.
        """
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.
        """
        return {field.name: getattr(self, field.name)
                for field in self.__dataclass_fields__.values()}


@dataclass
class GradingData:
    judge_model_name: str
    judge_model_provider: _SupportedProviders
    timestamp: str
    extracted_route_schema: Optional[dict]
    extracted_route_full: list[dict[str, Any]]
    extracted_route_clean: Optional[_MazeTypePath]
    outcome: MazeOutcome
    num_messages: int
    updated_messages: Optional[list[dict[str, Any]]]

    def to_dict(self) -> dict:
        """
        Convert the GradingData instance to a dictionary.
        """
        return {
            "judge_model_name": self.judge_model_name,
            "judge_model_provider": self.judge_model_provider,
            "timestamp": self.timestamp,
            "extracted_route_schema": self.extracted_route_schema,
            "extracted_route_full": self.extracted_route_full,
            "extracted_route_clean": self.extracted_route_clean,
            "outcome": self.outcome.to_dict(),
            "num_messages": self.num_messages,
            "updated_messages": self.updated_messages
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            judge_model_name=data["judge_model_name"],
            judge_model_provider=data["judge_model_provider"],
            timestamp=data["timestamp"],
            extracted_route_schema=data["extracted_route_schema"],
            extracted_route_full=data["extracted_route_full"],
            extracted_route_clean=data["extracted_route_clean"],
            outcome=MazeOutcome.from_dict(data["outcome"]),
            num_messages=data.get("num_messages"),
            updated_messages=data.get("updated_messages")
        )


def _prepare_grading_data(
    exp_folder: str,
    game_mode: Optional[str] = "collab",
    recheck_stop_condition: bool = False,
    stop_condition_min_turns: int = 6,
) -> tuple[MazeGame, dict, str, Optional[str], Conversation]:

    with open(os.path.join(exp_folder, 'result.yaml'), 'r') as f:
        solution = yaml.safe_load(f)
    with open(os.path.join(exp_folder, 'task_config.yaml'), 'r') as f:
        task_config = yaml.safe_load(f)

    agent_id_1, agent_id_2 = _extract_agent_ids_from_messages(
        messages=solution['messages'],
        require_both_agents=solution.get('game_mode', game_mode) == 'collab'
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
    lm_grader_provider: Optional[_SupportedProviders] = None,
    reasoning_effort: Optional[_ReasoningEffort] = None,
    temperature: float = 0.2,
    game_mode: Optional[str] = "collab",
    recheck_stop_condition: bool = False,
    stop_condition_min_turns: int = 6,
    verbose: bool = False
) -> str:

    lm_grader = LMFactory(
        model_name=lm_grader_name,
        provider=lm_grader_provider,
        config=LMConfig(
            temperature=temperature,
            reasoning_effort=reasoning_effort
        )
    ).create_instance()

    exp_folder = filepath.replace('result.yaml', '')
    (
        maze,
        solution,
        agent_id_1,
        agent_id_2,
        conversation
    ) = _prepare_grading_data(
        exp_folder=exp_folder,
        recheck_stop_condition=recheck_stop_condition,
        stop_condition_min_turns=stop_condition_min_turns
    )

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

    annotated_messages = None
    if recheck_stop_condition:
        # Annotate messages with model names and providers
        annotated_messages = conversation_to_dicts(
            result=conversation,
            skip_system=True
        )

    grading_file = GradingData(
        judge_model_name=lm_grader.adapter.model_name,
        judge_model_provider=lm_grader.adapter.provider.value,
        timestamp=timestamp,
        extracted_route_full=extracted_solution.route,
        extracted_route_clean=extracted_solution.processed_route,
        extracted_route_schema=extracted_solution.route_schema.to_dict(
        ) if extracted_solution.route_schema else None,
        outcome=updated_outcome,
        num_messages=len(conversation.messages),
        updated_messages=annotated_messages
    )

    if verbose:
        print(f'--saving grading file to {grading_file_path}')

    with open(grading_file_path, 'w') as f:
        yaml.safe_dump(
            grading_file.to_dict(),
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
    lm_grader_provider: Optional[_SupportedProviders] = None,
    reasoning_effort: Optional[_ReasoningEffort] = None,
    temperature: float = 0.2,
    max_workers=5,
    game_mode: Optional[str] = "collab",
    recheck_stop_condition: bool = False,
    stop_condition_min_turns: int = 6,
    verbose: bool = False
) -> list[str]:
    """Update the grading for a batch of experiment files.

    Returns:
        list[str]: A list of filepaths to the newly created grading files.
    """

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(lambda fp: update_grading_for_file(
                fp,
                lm_grader_name=lm_grader_name,
                lm_grader_provider=lm_grader_provider,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                game_mode=game_mode,
                recheck_stop_condition=recheck_stop_condition,
                stop_condition_min_turns=stop_condition_min_turns,
                verbose=verbose
            ), filepaths),
                total=len(filepaths)
            )
        )
    return results


def _process_grading_files(grading_files: list[str]) -> pd.DataFrame:

    updated_grades = []
    for filepath in grading_files:
        with open(filepath, 'r') as f:
            grading = yaml.safe_load(f)
        row = {
            'judge_model_name': grading['judge_model_name'],
            'judge_model_provider': grading['judge_model_provider'],
            'folder_id': filepath.split('/')[-2],
        }
        row.update(_get_outcome_fields(grading['outcome']))
        updated_grades.append(row)

    df_grades = pd.DataFrame(updated_grades)

    return df_grades


def get_grades(
        grading_config: GradingConfig, verbose: bool = False
) -> str:

    # Create the save folder with a timestamp
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_save_path = os.path.join(
        _MAIN_GRADING_FOLDER, grading_config.save_folder, timestamp)
    os.makedirs(full_save_path, exist_ok=True)

    with open(os.path.join(full_save_path, 'config.yaml'), 'w') as f:
        yaml.safe_dump(
            asdict(grading_config),
            f,
            default_flow_style=False,
            indent=4,
            sort_keys=False
        )

    if grading_config.existing_sample_file is not None:
        sample = pd.read_csv(grading_config.existing_sample_file)
    else:
        sample = get_balanced_sample(
            experiment_groups=grading_config.experiment_folders,
            sample_size_per_group=grading_config.sample_size_per_group,
            groupby_columns=grading_config.groupby_columns,
            total_sample_size=grading_config.total_sample_size,
            filter_conditions=grading_config.filter_conditions,
            min_rollout_length=grading_config.min_rollout_length,
            verbose=verbose
        )

    if sample.empty:
        raise ValueError(
            "The sample is empty, so there is nothing to grade. Check that "
            f"experiment_folders={grading_config.experiment_folders} contain "
            "results, that filter_conditions="
            f"{grading_config.filter_conditions} match them, and that the "
            "games are at least min_rollout_length="
            f"{grading_config.min_rollout_length} messages long."
        )

    files = sample['filepath'].tolist()
    if verbose:
        print(f"Sample size: {len(files)}")

    # Save the sample DataFrame to a CSV file
    sample.to_csv(os.path.join(full_save_path, 'sample.csv'), index=False)

    for i, judge_model in enumerate(grading_config.judge_models):
        if verbose:
            print(
                f"[{i}/{len(grading_config.judge_models)}]"
                f" Grading with model {judge_model}")

        # Update the grading for the batch
        graded_files = update_grading_for_batch(
            files,
            lm_grader_name=judge_model,
            lm_grader_provider=None,
            reasoning_effort=grading_config.reasoning_effort,
            temperature=grading_config.temperature,
            game_mode="collab",
            verbose=False,
            max_workers=5
        )

        df_files = pd.DataFrame({
            'judge_model': [judge_model] * len(graded_files),
            'filepath': graded_files
        })

        # Save the graded files DataFrame
        df_files.to_csv(os.path.join(
            full_save_path, f'graded_file_paths_{i}_{judge_model}.csv'), index=False)

        # Save the grades DataFrame
        try:
            df_grades = _process_grading_files(graded_files)
            df_grades.to_csv(
                os.path.join(
                    full_save_path,
                    f'graded_files_{i}_{judge_model}.csv'
                ),
                index=False
            )
        except Exception as e:
            raise ValueError(
                f"Error processing grading files for model {judge_model}: {e}")

    if verbose:
        print(f"Grading completed. Results saved to {full_save_path}")

    return full_save_path


def perform_grade_analysis(
        grading_config: GradingConfig, verbose: bool = False
) -> None:
    """
    Post-process the grades after they have been collected to generate the 
    analysis files.
    """

    # we start by reading in the sample file
    sample_file = os.path.join(
        grading_config.save_folder, 'sample.csv')
    if not os.path.exists(sample_file):
        raise FileNotFoundError(
            f"Sample file not found at {sample_file}. "
            "Please run get_grades first to generate the sample.")
    sample = pd.read_csv(sample_file)

    if 'judge_model_name' not in sample.columns:
        warnings.warn(
            'No "judge_model_name" column found - setting to gpt-4.1')
        sample['judge_model_name'] = 'gpt-4.1-2025-04-14'

    # In the save_folder, we expect several graded files each with the prefix
    # "graded_files_" followed by a judge model name.
    graded_result_filepaths = [
        os.path.join(grading_config.save_folder, f)
        for f in os.listdir(grading_config.save_folder)
        if f.startswith('graded_files_') and f.endswith('.csv')
    ]
    graded_results = []
    for graded_file in graded_result_filepaths:
        if verbose:
            print(f"Processing graded file: {graded_file}")
        df_grades = pd.read_csv(graded_file)
        graded_results.append(df_grades)

    if not graded_results:
        raise ValueError(
            "No graded files found in the save folder. "
            "Please run get_grades first to generate the grades.")
    else:
        if verbose:
            print(
                f"Found {len(graded_results)} graded files to process."
            )
            for grader in graded_results:
                print(f'- {grader.judge_model_name.unique()}')

    if "self_consistency" in grading_config.analyses_to_perform:
        if verbose:
            print("Performing self-consistency analysis...")

        self_model = sample["judge_model_name"].iloc[0]
        self_grades = []
        for df in graded_results:
            if df['judge_model_name'].iloc[0] == self_model:
                df_ = df.copy()
                self_grades.append(df_)

        if not self_grades:
            print(
                f"No self-consistency grades found for model {self_model}. "
                "Please ensure the model is included in the grading process.")
        else:
            _perform_grading_ablation(
                sample_df=sample,
                graded_results=self_grades,
                save_path=os.path.join(
                    grading_config.save_folder, f"self_consistency"
                ),
                same_graders=True,
                interpretation_model=grading_config.summary_interpretation_model_name
            )

    if "inter_consistency" in grading_config.analyses_to_perform:
        if verbose:
            print("Performing inter-consistency analysis...")

        self_model = sample["judge_model_name"].iloc[0]
        other_models = []
        for df in graded_results:
            if df['judge_model_name'].iloc[0] != self_model:
                df_ = df.copy()
                other_models.append(df_)

        if not other_models:
            print(
                f"No inter-consistency grades found for models other than {self_model}. "
                "Please ensure other models are included in the grading process.")
        else:
            _perform_grading_ablation(
                sample_df=sample,
                graded_results=other_models,
                save_path=os.path.join(
                    grading_config.save_folder,
                    f"inter_consistency_{self_model}"
                ),
                same_graders=False,
                interpretation_model=grading_config.summary_interpretation_model_name
            )


@dataclass
class GraderAnalysisResults:
    """Results container for grader analysis"""
    icc_weighted: float
    fleiss_kappa_binary: float
    pairwise_correlations: dict[tuple[str, str], float]
    majority_vs_reference: dict[str, float]
    conditional_analysis: dict[str, dict[str, float]]
    agent_effects: dict[str, float]

    def summary(self, apply_fdr: bool = True, alpha: float = 0.05) -> str:
        """Generate a summary report of the analysis

        Args:
            apply_fdr: Whether to apply FDR correction to p-values
            alpha: Significance level for FDR correction
        """
        report = []
        report.append("=== GRADER ANALYSIS SUMMARY ===")

        # Handle NaN values gracefully in formatting
        if np.isfinite(self.icc_weighted):
            report.append(f"ICC (Weighted Scores): {self.icc_weighted:.4f}")
        else:
            report.append(f"ICC (Weighted Scores): {self.icc_weighted}")

        if np.isfinite(self.fleiss_kappa_binary):
            report.append(
                f"Fleiss' Kappa (Binary): {self.fleiss_kappa_binary:.4f}")
        else:
            report.append(
                f"Fleiss' Kappa (Binary): {self.fleiss_kappa_binary}")

        report.append("\nPairwise Correlations:")
        for (g1, g2), corr in self.pairwise_correlations.items():
            if np.isfinite(corr):
                report.append(f"  {g1} vs {g2}: {corr:.4f}")
            else:
                report.append(f"  {g1} vs {g2}: {corr}")

        report.append("\nMajority Vote vs Reference:")
        for test, pval in self.majority_vs_reference.items():
            if np.isfinite(pval):
                report.append(f"  {test}: p={pval:.4f}")
            else:
                report.append(f"  {test}: {pval}")

        if self.conditional_analysis:
            report.append("\nConditional Analysis:")
            for condition, results in self.conditional_analysis.items():
                report.append(f"  {condition}:")
                for metric, value in results.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        report.append(f"    {metric}: {value:.4f}")
                    elif isinstance(value, (int, float)):
                        report.append(f"    {metric}: {value}")
                    else:
                        report.append(f"    {metric}: {value}")

        if self.agent_effects:
            report.append("\nAgent-Pair Effects:")
            for k, v in sorted(self.agent_effects.items()):
                if isinstance(v, (int, float)) and np.isfinite(v):
                    report.append(f"  {k}: {v:.4f}")
                else:
                    report.append(f"  {k}: {v}")

        # Add FDR correction notes
        if apply_fdr:
            fdr_notes = self._fdr_correction_notes(alpha)
            if fdr_notes:
                report.append("\n--- Multiple Testing Correction (FDR) ---")
                report.extend(fdr_notes)

        return "\n".join(report)

    def _fdr_correction_notes(self, alpha: float = 0.05) -> list[str]:
        """Generate FDR correction notes for p-values"""
        notes = []

        # Collect all p-values with their labels
        pvalues = []
        labels = []

        # Primary tests (majority vs reference)
        pvalue_keys = ['weighted_outcome_pvalue', 'binary_outcome_pvalue']
        for test_name in pvalue_keys:
            if test_name in self.majority_vs_reference:
                pval = self.majority_vs_reference[test_name]
                if isinstance(pval, (int, float)) and np.isfinite(pval):
                    pvalues.append(float(pval))
                    labels.append(
                        f"majority_{test_name.replace('_pvalue', '')}")

        # Conditional tests
        for condition, results in self.conditional_analysis.items():
            for test_name in pvalue_keys:
                if test_name in results:
                    pval = results[test_name]
                    if isinstance(pval, (int, float)) and np.isfinite(pval):
                        pvalues.append(float(pval))
                        labels.append(
                            f"{condition}_{test_name.replace('_pvalue', '')}")

        if not pvalues:
            return []  # No notes if no p-values

        # Apply FDR correction using Benjamini-Hochberg
        try:
            reject, corrected_pvals, _, _ = multipletests(
                pvalues, alpha=alpha, method='fdr_bh'
            )
        except Exception:
            notes.append("Error applying FDR correction")
            return notes

        # Group results by significance
        significant = []
        not_significant = []

        for label, orig_p, adj_p, sig in zip(labels, pvalues, corrected_pvals, reject):
            # Format the result string
            if abs(orig_p - adj_p) > 1e-10:  # Show adjusted p if meaningfully different
                result_str = f"{label}: p={orig_p:.4f} (adj={adj_p:.4f})"
            else:
                result_str = f"{label}: p={orig_p:.4f}"

            if sig:
                significant.append(result_str)
            else:
                not_significant.append(result_str)

        # Build the notes
        notes.append(f"Tests performed: {len(pvalues)}, Alpha: {alpha}")

        if significant:
            notes.append("Significant after FDR correction:")
            for s in significant:
                notes.append(f"  ✓ {s}")
        else:
            notes.append("No tests significant after FDR correction")

        if not_significant:
            notes.append("Not significant after FDR correction:")
            for s in not_significant:
                notes.append(f"  ✗ {s}")

        # Summary interpretation
        sig_pct = len(significant) / len(pvalues) * 100 if pvalues else 0
        notes.append(
            f"\nInterpretation: {len(significant)}/{len(pvalues)} tests remain significant ({sig_pct:.0f}%)")

        return notes


class GraderAnalysis:
    def __init__(
        self,
        df: pd.DataFrame,
        reference_grader: str,
        verbose: bool = True
    ):
        """
        Initialize with DataFrame containing:
        - outcome_id: unique identifier for each outcome
        - agent: agent that performed the task
        - grader: grader identifier
        - weighted_outcome: continuous score (0-1)
        - outcome_success: binary outcome (True/False)
        """
        required_columns = [
            'outcome_id',
            'agent',
            'grader',
            'weighted_outcome',
            'outcome_success'
        ]
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.df = df.copy()

        # Validate data types and ranges
        if not pd.api.types.is_numeric_dtype(self.df['weighted_outcome']):
            raise ValueError("weighted_outcome must be numeric")

        if not self.df['weighted_outcome'].between(-1, 1).all():
            warnings.warn(
                "Some weighted_outcome values are outside [-1, 1] range")

        # Convert success to boolean if needed
        if self.df['outcome_success'].dtype != bool:
            self.df['outcome_success'] = self.df['outcome_success'].astype(
                bool)

        self.graders = sorted(df['grader'].unique())
        self.agents = sorted(df['agent'].unique())
        self.reference_grader = reference_grader
        if self.reference_grader not in self.graders:
            raise ValueError(
                f"Reference grader '{self.reference_grader}' not found in graders: {self.graders}")

        # Validate we have sufficient data
        if len(self.graders) < 2:
            raise ValueError("Need at least 2 graders for analysis")

        # Check for complete data matrix
        pivot_check = self.df.pivot_table(
            index='outcome_id',
            columns='grader',
            values='weighted_outcome',
            aggfunc='count'
        )
        self.complete_outcomes = pivot_check.dropna().index.tolist()

        if len(self.complete_outcomes) == 0:
            raise ValueError("No outcomes have grades from all graders")

        incomplete_pct = (len(pivot_check) -
                          len(self.complete_outcomes)) / len(pivot_check) * 100
        if incomplete_pct > 0:
            warnings.warn(
                f"{incomplete_pct:.1f}% of outcomes have missing grades from some graders")

        if len(self.graders) < 3:
            warnings.warn(
                f"Only {len(self.graders)} graders found. "
                "Results may be unreliable. Consider at least 3 graders for robust analysis."
            )

        self.verbose = verbose

    def compute_icc(self, scores_matrix: np.ndarray) -> float:
        """Compute ICC(2,1) for absolute agreement - Two-way random effects model"""
        # Remove rows with any NaN values
        valid_mask = ~np.isnan(scores_matrix).any(axis=1)
        scores_clean = scores_matrix[valid_mask]

        if len(scores_clean) < 3:  # Need at least 3 subjects for reliable ICC
            return np.nan

        n_subjects, n_raters = scores_clean.shape

        if n_raters < 2:
            return np.nan

        # Calculate means
        subject_means = np.mean(scores_clean, axis=1)
        rater_means = np.mean(scores_clean, axis=0)
        grand_mean = np.mean(scores_clean)

        # Sum of squares
        ss_total = np.sum((scores_clean - grand_mean) ** 2)
        ss_between_subjects = n_raters * \
            np.sum((subject_means - grand_mean) ** 2)
        ss_between_raters = n_subjects * \
            np.sum((rater_means - grand_mean) ** 2)
        ss_residual = ss_total - ss_between_subjects - ss_between_raters

        # Mean squares
        ms_between_subjects = ss_between_subjects / (n_subjects - 1)
        ms_between_raters = ss_between_raters / (n_raters - 1)
        ms_residual = ss_residual / ((n_subjects - 1) * (n_raters - 1))

        # ICC(2,1) calculation
        if ms_between_subjects + ms_residual == 0:
            return np.nan

        icc = (ms_between_subjects - ms_residual) / (
            ms_between_subjects + (n_raters - 1) * ms_residual +
            n_raters * (ms_between_raters - ms_residual) / n_subjects
        )

        return max(0, min(1, icc))  # Clamp ICC between 0 and 1

    def prepare_matrices(self, outcome_subset: Optional[list] = None):
        """Prepare score and success matrices for analysis"""
        df_subset = self.df.copy()

        if outcome_subset is not None:
            df_subset = df_subset[df_subset['outcome_id'].isin(outcome_subset)]

        pivot_scores = df_subset.pivot(
            index='outcome_id',
            columns='grader',
            values='weighted_outcome'
        )
        pivot_success = df_subset.pivot(
            index='outcome_id',
            columns='grader',
            values='outcome_success'
        )

        return pivot_scores.values, pivot_success.values.astype(float)

    def grader_agreement(self, outcome_subset: Optional[list] = None) -> dict[str, Any]:
        scores_matrix, success_matrix = self.prepare_matrices(outcome_subset)

        # Check if we have enough valid data
        valid_scores = scores_matrix[~np.isnan(scores_matrix).any(axis=1)]
        if len(valid_scores) < 3:
            return {
                'icc_weighted': np.nan,
                'fleiss_kappa_binary': np.nan,
                'pairwise_correlations': {},
                'insufficient_data': True,
                'n_valid_outcomes': len(valid_scores)
            }

        # ICC for weighted scores
        icc_score = self.compute_icc(scores_matrix)

        # Fleiss' kappa for binary outcomes
        valid_mask = ~np.isnan(success_matrix).any(axis=1)
        success_clean = success_matrix[valid_mask].astype(int)

        if success_clean.shape[0] > 2 and success_clean.shape[1] > 1:
            try:
                # Convert rater-by-subject binary matrix into counts per category [fail, success]
                # Each row must sum to number of raters for Fleiss' kappa
                counts = np.stack([
                    np.sum(1 - success_clean, axis=1),  # count of failures (0)
                    # count of successes (1)
                    np.sum(success_clean, axis=1)
                ], axis=1)

                # Ensure we have some variability across subjects/raters
                if np.any(counts[:, 0] != counts[:, 1]) or np.var(counts) > 0:
                    kappa_score = fleiss_kappa(counts)
                else:
                    kappa_score = 1.0
            except Exception:
                kappa_score = np.nan
        else:
            kappa_score = np.nan

        # Pairwise correlations for weighted scores
        pairwise_corr = {}
        df_subset = self.df.copy()
        if outcome_subset is not None:
            df_subset = df_subset[df_subset['outcome_id'].isin(outcome_subset)]

        for grader1, grader2 in itertools.combinations(self.graders, 2):
            scores1 = df_subset[df_subset['grader'] == grader1].set_index('outcome_id')[
                'weighted_outcome']
            scores2 = df_subset[df_subset['grader'] == grader2].set_index('outcome_id')[
                'weighted_outcome']

            # Find common outcomes
            common_outcomes = scores1.index.intersection(scores2.index)
            if len(common_outcomes) > 2:  # Need at least 3 points for reliable correlation
                s1_common = scores1[common_outcomes].values
                s2_common = scores2[common_outcomes].values

                # Check for variance
                if np.var(s1_common) > 1e-10 and np.var(s2_common) > 1e-10:
                    corr, _ = pearsonr(s1_common, s2_common)
                    pairwise_corr[(grader1, grader2)] = corr
                else:
                    pairwise_corr[(grader1, grader2)] = 1.0 if np.allclose(
                        s1_common, s2_common) else 0.0
            else:
                pairwise_corr[(grader1, grader2)] = np.nan

        return {
            'icc_weighted': icc_score,
            'fleiss_kappa_binary': kappa_score,
            'pairwise_correlations': pairwise_corr
        }

    def majority_vote_analysis(
            self, outcome_subset: Optional[list] = None) -> dict[str, float | str]:
        """Compare majority vote against reference grader"""
        results = {}

        df_subset = self.df.copy()
        if outcome_subset is not None:
            df_subset = df_subset[df_subset['outcome_id'].isin(outcome_subset)]

        # Get reference grader scores
        ref_data = df_subset[
            df_subset['grader'] == self.reference_grader
        ].set_index('outcome_id')

        if len(ref_data) == 0:
            return {'error': 'No reference grader data found'}

        # Compute majority vote for each outcome
        majority_scores = []
        majority_success = []
        ref_scores = []
        ref_success = []

        for outcome_id in ref_data.index:
            outcome_data = df_subset[df_subset['outcome_id'] == outcome_id]

            if len(outcome_data) >= 2:  # Need at least 2 graders
                # Majority vote for binary outcome (use >= 0.5 threshold)
                success_votes = outcome_data['outcome_success'].sum()
                maj_success = success_votes >= len(outcome_data) / 2

                # Mean score as majority for weighted score
                maj_score = outcome_data['weighted_outcome'].mean()

                majority_scores.append(maj_score)
                majority_success.append(maj_success)
                ref_scores.append(ref_data.loc[outcome_id, 'weighted_outcome'])
                ref_success.append(ref_data.loc[outcome_id, 'outcome_success'])

        if len(majority_scores) < 3:  # Need sufficient data for statistical tests
            return {'error': 'Insufficient data for statistical testing'}

        # Convert to numpy arrays
        ref_scores = np.array(ref_scores)
        majority_scores = np.array(majority_scores)
        ref_success = np.array(ref_success, dtype=bool)
        majority_success = np.array(majority_success, dtype=bool)

        # Test for weighted scores - use Wilcoxon if not normally distributed
        if np.var(ref_scores) > 1e-10 and np.var(majority_scores) > 1e-10:
            try:
                diffs = ref_scores - majority_scores
                n = diffs.size
                # Avoid Shapiro on very large samples; default to t-test via CLT
                if n > 5000:
                    _, p_weighted = ttest_rel(ref_scores, majority_scores)
                else:
                    _, p_normal = stats.shapiro(diffs)
                    if p_normal > 0.05:
                        _, p_weighted = ttest_rel(ref_scores, majority_scores)
                    else:
                        _, p_weighted = wilcoxon(
                            ref_scores, majority_scores, zero_method='zsplit')
                results['weighted_outcome_pvalue'] = p_weighted
            except Exception:
                results['weighted_outcome_pvalue'] = np.nan
        else:
            results['weighted_outcome_pvalue'] = 1.0

        # McNemar's test for binary outcomes
        try:
            # Create contingency table for McNemar's test
            both_success = np.sum(ref_success & majority_success)
            both_failure = np.sum(~ref_success & ~majority_success)
            ref_success_maj_failure = np.sum(ref_success & ~majority_success)
            ref_failure_maj_success = np.sum(~ref_success & majority_success)

            mcnemar_table = np.array([
                [both_failure, ref_failure_maj_success],
                [ref_success_maj_failure, both_success]
            ])

            # Only run test if there are discordant pairs
            discordant_pairs = ref_success_maj_failure + ref_failure_maj_success
            if discordant_pairs > 0:
                if discordant_pairs >= 25:  # Use normal approximation for large samples
                    mcnemar_result = mcnemar(
                        mcnemar_table, exact=False, correction=True)
                else:  # Use exact test for small samples
                    mcnemar_result = mcnemar(mcnemar_table, exact=True)
                # Access p-value safely for typing
                results['binary_outcome_pvalue'] = getattr(
                    mcnemar_result, 'pvalue', np.nan)
            else:
                results['binary_outcome_pvalue'] = 1.0  # Perfect agreement

        except Exception as e:
            results['binary_outcome_pvalue'] = np.nan

        # Add paired effect size (Cohen's dz)
        try:
            diffs = majority_scores - ref_scores
            sd_diff = diffs.std(ddof=1)
            if sd_diff > 1e-10:
                cohens_dz = diffs.mean() / sd_diff
                # Keep the existing key name to avoid breaking consumers
                results['cohens_d_weighted'] = cohens_dz
        except Exception:
            pass

        # Add confidence intervals for the mean difference
        diffs = majority_scores - ref_scores
        mean_diff = diffs.mean()
        se_diff = diffs.std(ddof=1) / np.sqrt(len(diffs))
        ci_95 = (mean_diff - 1.96*se_diff, mean_diff + 1.96*se_diff)

        results['weighted_diff_mean'] = mean_diff
        results['weighted_diff_ci_lower'] = ci_95[0]
        results['weighted_diff_ci_upper'] = ci_95[1]

        return results

    def conditional_analysis(self) -> dict[str, dict[str, float]]:
        """Analyze where disagreements concentrate relative to reference verdict"""
        ref_data = self.df[self.df['grader'] == self.reference_grader]

        # Calculate score variability for each outcome
        outcome_metrics = []
        for outcome_id in self.df['outcome_id'].unique():
            outcome_data = self.df[self.df['outcome_id'] == outcome_id]
            if len(outcome_data) >= 2:
                # Get reference verdict
                ref_verdict = ref_data[
                    ref_data['outcome_id'] == outcome_id
                ]['outcome_success'].iloc[0]

                # Binary disagreement (meaningful)
                binary_values = outcome_data['outcome_success'].values
                has_binary_disagreement = len(set(binary_values)) > 1

                # Continuous score variability (not binary!)
                scores = outcome_data['weighted_outcome'].values
                score_std = np.std(scores)
                score_range = np.max(scores) - np.min(scores)

                outcome_metrics.append({
                    'outcome_id': outcome_id,
                    'ref_verdict': ref_verdict,
                    'has_binary_disagreement': has_binary_disagreement,
                    'score_std': score_std,
                    'score_range': score_range
                })

        metrics_df = pd.DataFrame(outcome_metrics)

        # Split by reference verdict
        success_cases = metrics_df[metrics_df['ref_verdict'] == True]
        failure_cases = metrics_df[metrics_df['ref_verdict'] == False]

        results = {}

        # Store results for success and failure cases
        for condition, cases in [('success_cases', success_cases), ('failure_cases', failure_cases)]:
            if len(cases) > 0:
                # Binary disagreement rate
                disagree_count = cases['has_binary_disagreement'].sum()
                disagree_rate = float(disagree_count / len(cases))

                # Score variability metrics
                mean_std = float(cases['score_std'].mean())
                mean_range = float(cases['score_range'].mean())

                results[condition] = {
                    'n_outcomes': len(cases),
                    'binary_disagreement_rate': disagree_rate,
                    'mean_score_std': mean_std,
                    'mean_score_range': mean_range
                }
            else:
                results[condition] = {
                    'n_outcomes': 0,
                    'binary_disagreement_rate': np.nan,
                    'mean_score_std': np.nan,
                    'mean_score_range': np.nan
                }

        # Statistical tests comparing success vs failure cases
        if len(success_cases) > 0 and len(failure_cases) > 0:
            # 1. Binary disagreement rate comparison (Fisher's exact test)
            success_disagree = success_cases['has_binary_disagreement'].sum()
            failure_disagree = failure_cases['has_binary_disagreement'].sum()

            contingency = np.array([
                [success_disagree, len(success_cases) - success_disagree],
                [failure_disagree, len(failure_cases) - failure_disagree]
            ])

            _, binary_pvalue = fisher_exact(contingency)

            # Store the p-value with the expected key name for FDR correction
            results['success_cases']['binary_outcome_pvalue'] = float(
                binary_pvalue)
            results['failure_cases']['binary_outcome_pvalue'] = float(
                binary_pvalue)

            # 2. Score variability comparison (t-test on std devs)
            from scipy.stats import ttest_ind

            _, variability_pvalue = ttest_ind(
                success_cases['score_std'],
                failure_cases['score_std']
            )

            # Store with expected key name for FDR correction
            results['success_cases']['weighted_outcome_pvalue'] = float(
                variability_pvalue)
            results['failure_cases']['weighted_outcome_pvalue'] = float(
                variability_pvalue)

            # 3. Effect sizes
            # Cohen's d for the variability difference
            success_std = success_cases['score_std'].values
            failure_std = failure_cases['score_std'].values
            pooled_std = np.sqrt((success_std.var() + failure_std.var()) / 2)

            if pooled_std > 0:
                cohens_d = (failure_std.mean() -
                            success_std.mean()) / pooled_std
                # Store as expected by summary
                # Negative for success perspective
                results['success_cases']['cohens_d_weighted'] = float(
                    -cohens_d)
                results['failure_cases']['cohens_d_weighted'] = float(
                    cohens_d)   # Positive for failure perspective

        return results

    def agent_effects(self) -> dict[str, float]:
        """Analyze if grading consistency varies by agent"""
        results = {}

        for agent in self.agents:
            agent_data = self.df[self.df['agent'] == agent]

            # Check if we have sufficient data for this agent
            agent_outcomes = agent_data['outcome_id'].nunique()
            agent_graders = agent_data['grader'].nunique()

            if agent_outcomes >= 3 and agent_graders >= 2:
                try:
                    temp_analysis = GraderAnalysis(
                        agent_data,
                        self.reference_grader,
                        self.verbose
                    )
                    agreement = temp_analysis.grader_agreement()

                    results[f'agent_{agent}_icc'] = agreement['icc_weighted']
                    results[f'agent_{agent}_kappa'] = agreement['fleiss_kappa_binary']

                    # Count of outcomes for this agent
                    results[f'agent_{agent}_n_outcomes'] = agent_outcomes

                except Exception as e:
                    results[f'agent_{agent}_error'] = str(e)
            else:
                results[f'agent_{agent}_insufficient_data'] = True

        return results

    def run_full_analysis(self) -> GraderAnalysisResults:
        """Run complete grader analysis with error handling"""
        if self.verbose:
            print("Running grader agreement analysis...")
        try:
            agreement = self.grader_agreement()
        except Exception as e:
            print(f"Error in agreement analysis: {e}")
            agreement = {
                'icc_weighted': np.nan,
                'fleiss_kappa_binary': np.nan,
                'pairwise_correlations': {}
            }

        if self.verbose:
            print("Running majority vote analysis...")
        try:
            majority = self.majority_vote_analysis()
        except Exception as e:
            print(f"Error in majority vote analysis: {e}")
            majority = {}

        if self.verbose:
            print("Running conditional analysis...")
        try:
            conditional = self.conditional_analysis()
        except Exception as e:
            print(f"Error in conditional analysis: {e}")
            conditional = {}

        if self.verbose:
            print("Running agent effects analysis...")
        try:
            agent_effects = self.agent_effects()
        except Exception as e:
            print(f"Error in agent effects analysis: {e}")
            agent_effects = {}

        # Keep only numeric values in majority for type safety
        majority_clean: dict[str, float] = {}
        if isinstance(majority, dict):
            for k, v in majority.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    majority_clean[k] = float(v)

        return GraderAnalysisResults(
            icc_weighted=agreement.get('icc_weighted', np.nan),
            fleiss_kappa_binary=agreement.get('fleiss_kappa_binary', np.nan),
            pairwise_correlations=agreement.get('pairwise_correlations', {}),
            majority_vs_reference=majority_clean,
            conditional_analysis=conditional,
            agent_effects=agent_effects
        )


def _get_interpretation(
    model_name: str,
    summary: str,
    same_graders: bool
) -> str | None:
    """Generate an interpretation of the grading results using a language model."""
    prompt = UserMessage(
        render_template(
            _GRADING_INTERPRETATION_TEMPLATE,
            user_data={
                "same_graders": same_graders,
                "test_results": summary,
            }
        )
    )
    interpreter = LMFactory(model_name=model_name).create_instance()
    interpretation = interpreter(prompt, return_as_message=True, num_samples=1)

    if isinstance(interpretation[0], AssistantMessage):
        interpretation = interpretation[0].content
    else:
        warnings.warn(
            "Interpretation did not return an AssistantMessage, "
        )
        interpretation = None

    return interpretation


def _perform_grading_ablation(
    sample_df: pd.DataFrame,
    graded_results: list[pd.DataFrame],
    save_path: str,
    same_graders: bool,
    interpretation_model: Optional[str] = None,
    verbose: bool = True
) -> GraderAnalysisResults:
    """
    Run the grader analysis on a grading sample and save the results.

    Args:
        sample_df: DataFrame with sample metadata (should have 'folder_id' column)
        graded_results: list of DataFrames with grading results 
                       (should have 'judge_model_name', 'folder_id', 'weighted_outcome', 'outcome_success' columns)
        save_path: Path to save analysis results

    Returns:
        GraderAnalysisResults object with all analysis results
    """

    # Ensure join column exists in sample_df
    if 'folder_id' not in sample_df.columns:
        raise ValueError("sample_df must have 'folder_id' column")

    # Required columns
    required_cols = [
        'judge_model_name',
        'folder_id',
        'weighted_outcome',
        'outcome_success'
    ]

    # Add the reference grader
    df_reference = sample_df[required_cols].copy()
    graded_results = [df_reference] + graded_results

    # Combine all graded results
    grader_id = 0
    all_grades = []
    for df in graded_results:
        df_copy = df.copy()
        missing_cols = [
            col for col in required_cols if col not in df_copy.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in graded results: {missing_cols}")

        # In the self-consistency case, we might have multiple grades for same judge model
        df_copy['grader'] = df_copy['judge_model_name'].apply(
            lambda x: f"{grader_id}_{x}")
        grader_id += 1

        all_grades.append(df_copy)

    combined_grades = pd.concat(all_grades, ignore_index=True)

    # Create agent column if it doesn't exist
    if 'agent' not in sample_df.columns:
        # Try to infer agent from other columns or use a default
        if 'lm_pair' in sample_df.columns:
            sample_df['agent'] = sample_df['lm_pair']
        elif 'lm_1_name' in sample_df.columns and 'lm_2_name' in sample_df.columns:
            sample_df['agent'] = sample_df['lm_1_name'] + \
                '_' + sample_df['lm_2_name']
        else:
            sample_df['agent'] = 'default_agent'

    # Merge grades with sample data
    analysis_df = combined_grades.merge(
        sample_df[['folder_id', 'agent']],
        on='folder_id',
        how='left'
    )

    # Rename columns to match expected format
    analysis_df = analysis_df.rename(columns={
        'folder_id': 'outcome_id',
    })

    # Ensure boolean success column
    if analysis_df['outcome_success'].dtype != bool:
        analysis_df['outcome_success'] = analysis_df['outcome_success'].astype(
            bool)

    # Run analysis
    analyzer = GraderAnalysis(
        analysis_df,
        reference_grader=f"0_{sample_df['judge_model_name'].iloc[0]}",
        verbose=verbose
    )
    results = analyzer.run_full_analysis()

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save results
    with open(f"{save_path}/grader_analysis_results.txt", 'w') as f:
        f.write(results.summary())

    # Save detailed results as CSV
    results_dict = {
        'metric': [],
        'value': [],
        'category': []
    }

    results_dict['metric'].append('ICC_weighted_outcomes')
    results_dict['value'].append(results.icc_weighted)
    results_dict['category'].append('agreement')

    results_dict['metric'].append('Fleiss_kappa_binary')
    results_dict['value'].append(results.fleiss_kappa_binary)
    results_dict['category'].append('agreement')

    for (g1, g2), corr in results.pairwise_correlations.items():
        results_dict['metric'].append(f'correlation_{g1}_vs_{g2}')
        results_dict['value'].append(corr)
        results_dict['category'].append('pairwise_correlation')

    for test, pval in results.majority_vs_reference.items():
        results_dict['metric'].append(f'majority_vs_ref_{test}')
        results_dict['value'].append(pval)
        results_dict['category'].append('majority_vote_test')

    # Add agent effects
    for key, val in results.agent_effects.items():
        results_dict['metric'].append(f'agent_effect_{key}')
        results_dict['value'].append(val)
        results_dict['category'].append('agent_effect')

    # Add conditional analysis metrics
    for condition, metrics in results.conditional_analysis.items():
        for m_key, m_val in metrics.items():
            if isinstance(m_val, (int, float)) and np.isfinite(m_val):
                results_dict['metric'].append(f'{condition}_{m_key}')
                results_dict['value'].append(m_val)
                results_dict['category'].append(f'conditional_{condition}')

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f"{save_path}/grader_analysis_detailed.csv", index=False)

    # Get interpretation if requested
    interpretation = None
    if interpretation_model:
        if verbose:
            print("Generating interpretation of results...")
        interpretation = _get_interpretation(
            model_name=interpretation_model,
            summary=results.summary(),
            same_graders=same_graders
        )
        if interpretation:
            with open(f"{save_path}/grader_analysis_interpretation.txt", 'w') as f:
                f.write(interpretation)

    if verbose:
        print(f"Analysis complete. Results saved to {save_path}")
        print(results.summary())
        if interpretation:
            print("\n\nInterpretation of results:\n")
            print(interpretation)

    return results


def main(config_file: str, verbose: bool = False) -> None:
    """
    example:
    python -m src.grading_ablation --config_file experiments/grading_configs/example_grading.yaml
    """
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = GradingConfig.from_dict(config_dict)

    if config.config_type == "get_grades":
        _ = get_grades(config, verbose=verbose)
    elif config.config_type == "analyses":
        perform_grade_analysis(config, verbose=verbose)
    elif config.config_type == "ablation":
        grades_folder = get_grades(config, verbose=verbose)
        config.save_folder = grades_folder
        perform_grade_analysis(config, verbose=verbose)

    elif config.config_type == "update_verification":
        pass
    else:
        raise ValueError(
            f"Invalid config_type: {config.config_type}. "
            f"Must be one of ['get_grades', analyses, 'update']"
        )


if __name__ == '__main__':
    os.makedirs(_MAIN_GRADING_FOLDER, exist_ok=True)

    fire.Fire(main)
