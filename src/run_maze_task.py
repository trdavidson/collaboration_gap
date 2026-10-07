import ast
import copy
import random
import os
import yaml
import time
import logging
from uuid import uuid4
from typing import Optional, Any, Literal
from datetime import datetime as dt
from dataclasses import dataclass, asdict
import functools
import time

import fire
import pandas as pd

from .tasks.maze_game import (
    MazeGame, PreMazeGameConfig, MazeTaskConfig, _MazeTypeStr,
    _MazeTypePath, MazeOutcome, SolutionType)
from .models.inference import (LM, LMFactory, _SupportedProviders)
from .models.adapters import (
    lookup_model_details, ProviderAPI, LMConfig, _ReasoningEffort)
from .models.conversation_tools import (
    align_message_roles, annotate_messages_with_prefix, conversation_to_dicts)
from .utils.general_utilities import (print_c, log_c, ColorType, set_random_seed,
                                     normalize_deployment_name,
                                     deployment_names_match)
from .utils.reusable_classes import (
    Conversation, Message, AssistantMessage, UserMessage, SystemMessage,
    RoleType)
from .utils.prompt_helper import render_template


# Set up logging
# Configure logging to show INFO and above
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s:\n%(message)s'
)
logger = logging.getLogger(__name__)

_REAL_USER_AGENT_ID = "real_user"
# The only supported continuation type.
_RELAY_CONTINUATION = "relay_game"
_COMMUNICATION_PROTOCOL_FILE = "communication_protocol.j2"
# The game modes play_maze_task dispatches. A relay is a collab game continuing
# a prior one (a PreMazeGameConfig with an overview file), and the one-shot
# critic is a one-shot game with critic_steps > 0.
_GameModeTypes = Literal["collab", "one-shot"]

_FINISHED_GAME_PHRASE = 'ACTI!'
_FINISHED_PRE_INTERACTION_PHRASE = 'Ready to play!'


@dataclass
class MazeTaskResult:
    lm_1: LM
    lm_2: Optional[LM]
    task_config: MazeTaskConfig
    game: MazeGame
    result: Conversation
    result_to_verify: Conversation
    route: Optional[list[dict[str, Any]]] = None
    processed_route: Optional[list[list[int]]] = None
    continuation_filepath: Optional[str] = None


@dataclass
class ExtractedMazeRoute:
    route: list[dict[str, Any]]
    route_schema: Optional[SolutionType] = None
    error: Optional[str] = None

    @functools.cached_property
    def processed_route(self) -> _MazeTypePath | None:
        if not self.route:
            return None

        if not isinstance(self.route, list) or not self.route:
            return None

        return self._process_route(self.route)

    @staticmethod
    def _process_route(route):
        # Only extract the moves
        raw_solution = [turn.get("coordinates", None)
                        for turn in route if turn.get("turn_type") == "move"]
        # Coordinates are lists of iterables, so we need to unpack to single entries
        # Some times, the grader will extract as [(xi, yi), (xi+1, yi+1)] instead
        # of [(xi, yi)], i.e., the transition and not just the destination
        solution = []
        for coordinates in raw_solution:

            # clean the coordinates to correct for bad formatting
            cleaned_coordinates = ExtractedMazeRoute._clean_coordinates(
                coordinates)

            if cleaned_coordinates is None:
                continue

            if len(cleaned_coordinates) == 1:
                # If there's only one coordinate, this can either be a single coordinate, e.g., (0, 1)
                # OR a coordinate containing a letter, e.g., (A1, A2)
                # OR a coordinate containing directions, e.g., ("up", "down")
                if len(cleaned_coordinates[0]) == 2 and \
                        ExtractedMazeRoute._is_letter_coordinate(
                            cleaned_coordinates[0][0]):
                    solution.append(cleaned_coordinates[0][1])

                elif (
                    len(cleaned_coordinates[0]) == 2 and
                    isinstance(cleaned_coordinates[0][0], str) and
                    isinstance(cleaned_coordinates[0][1], str) and
                    (
                        cleaned_coordinates[0][0].lower() in ["up", "down", "left", "right"] or
                        cleaned_coordinates[0][1].lower(
                        ) in ["up", "down", "left", "right"]
                    )
                ):
                    solution.append(cleaned_coordinates[0][1])

                else:
                    solution.append(cleaned_coordinates[0])

            # We only take the destination, not the transition
            elif len(cleaned_coordinates) == 2:
                # We might have a parsing mistake, e.g., ['(0', '1)'], which gets cleaned to ['0', '1']
                # So we need to check if both are single characters, or -- for
                # mazes with a side of 10 or more -- both bare numbers. Two bare
                # numbers are always the two components of one coordinate: a
                # transition arrives as two complete coordinates instead, either
                # as pairs ([[0, 0], [0, 1]]) or as letter coordinates
                # (['A1', 'B2']), and both of those are handled below.
                if (isinstance(cleaned_coordinates[0], str) and
                    isinstance(cleaned_coordinates[1], str) and
                    (
                        (len(cleaned_coordinates[0]) == 1 and
                         len(cleaned_coordinates[1]) == 1) or
                        (cleaned_coordinates[0].strip().isdigit() and
                         cleaned_coordinates[1].strip().isdigit())
                    )
                    ):
                    solution.append(
                        (cleaned_coordinates[0], cleaned_coordinates[1]))
                elif isinstance(cleaned_coordinates[0], int) and isinstance(cleaned_coordinates[1], int):
                    # If both are integers, we assume they are coordinates
                    solution.append(
                        (cleaned_coordinates[0], cleaned_coordinates[1]))
                else:
                    solution.append(cleaned_coordinates[1])

            # Three or more coordinates in one turn is a whole path segment
            # rather than a transition. One-shot agents propose the entire route
            # in a single message, and a critic revision resubmits all of it, so
            # this is the normal shape there -- the grader template even invites
            # it. Dropping these turns would score a complete valid route as zero
            # progress.
            else:
                for coordinate in cleaned_coordinates:
                    if isinstance(coordinate, (list, tuple)) and len(coordinate) == 2:
                        solution.append(coordinate)
                    elif ExtractedMazeRoute._is_letter_coordinate(coordinate):
                        solution.append(coordinate)
                    elif isinstance(coordinate, str) and coordinate.lower() in (
                            "up", "down", "left", "right"):
                        solution.append(coordinate)
                    # Anything else at this length is a mis-parse (e.g. a single
                    # coordinate split into loose scalars), not a waypoint.
        # Any None coordinates are removed as are dicts
        solution = [coord for coord in solution
                    if coord is not None and not isinstance(coord, dict)]

        return solution

    @staticmethod
    def _clean_coordinates(coordinates) -> None | tuple | list:
        # No coordinates
        if not coordinates or len(coordinates) == 0:
            return None

        if isinstance(coordinates, str):
            coordinates = ExtractedMazeRoute._clean_single_coordinate(
                coordinates)

        if not (isinstance(coordinates, tuple) or isinstance(coordinates, list)):
            return None

        # We iterate through the coordinates and clean them
        coordinates_cleaned = []
        for coordinate in coordinates:
            if isinstance(coordinate, tuple) or isinstance(coordinate, list):
                # If it's a tuple or list, we clean each element
                coordinate = [ExtractedMazeRoute._clean_single_coordinate(
                    coord) for coord in coordinate]
                if (coordinate and len(coordinate) == 1 and
                        isinstance(coordinate, (list, tuple))):
                    coordinate = coordinate[0]
            else:
                coordinate = ExtractedMazeRoute._clean_single_coordinate(
                    coordinate)

            if coordinate is not None:
                coordinates_cleaned.append(coordinate)

        return coordinates_cleaned

    @staticmethod
    def _clean_single_coordinate(coordinate: str | tuple | list):
        if isinstance(coordinate, str):
            coordinate = coordinate.strip()

            # A bracketed, comma-separated pair is one coordinate, so parse it
            # as a literal before the comma split below gets to it. That split
            # turns "(10, 11)" into two components, and the reassembly in
            # _process_route could only rejoin single-character ones, so any
            # coordinate with a value of 10 or more silently lost its row.
            # ast.literal_eval only accepts literals -- it never executes the
            # string. Braces are normalised to
            # parentheses first so "{0, 1}" parses as an ordered tuple rather
            # than a set, whose iteration order would scramble the coordinate.
            if (',' in coordinate and
                    coordinate[:1] in '([{' and coordinate[-1:] in ')]}'):
                try:
                    parsed = ast.literal_eval(
                        coordinate.replace('{', '(').replace('}', ')'))
                    if isinstance(parsed, (tuple, list)):
                        return list(parsed)
                except (ValueError, SyntaxError):
                    # Not a literal (e.g. "(A1, B2)"): fall through to the
                    # string handling below.
                    pass

            if ('(' in coordinate or ')' in coordinate) and ',' not in coordinate:
                coordinate = coordinate.replace('(', '').replace(')', '')
            elif ',' in coordinate:
                coordinate = coordinate.split(',')
            elif ('[' in coordinate or ']' in coordinate) and ',' not in coordinate:
                coordinate = coordinate.replace('[', '').replace(']', '')
            elif ('{' in coordinate or '}' in coordinate) and ',' not in coordinate:
                coordinate = coordinate.replace('{', '').replace('}', '')
            elif '(' in coordinate and ')' in coordinate and ',' in coordinate:
                coordinate = eval(coordinate)
            elif ('[' in coordinate or ']' in coordinate) and ',' in coordinate:
                coordinate = eval(coordinate)
            elif '{' in coordinate and '}' in coordinate and ',' in coordinate:
                coordinate = eval(coordinate.replace(
                    '{', '(').replace('}', ')'))

        return coordinate

    @staticmethod
    def _is_letter_coordinate(coord) -> bool:
        if isinstance(coord, str) and len(coord) == 2:
            letter_digit = coord[0].isalpha() and coord[1].isdigit()
            digit_letter = coord[0].isdigit() and coord[1].isalpha()
            letter_letter = coord[0].isalpha() and coord[1].isalpha()

            return letter_digit or digit_letter or letter_letter

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return {
            "route": self.route,
            "route_schema": self.route_schema,
            "processed_route": self.processed_route,
            "error": self.error
        }


@dataclass
class MazeTaskSaveResult:
    task_path: str
    maze_str: str
    maze_coordinates: _MazeTypeStr
    user_intermediate: bool
    split_view: bool
    split_1: _MazeTypeStr
    split_2: _MazeTypeStr
    messages: list[dict[str, Any]]
    max_messages: int
    num_messages: int
    extracted_route_schema: Optional[dict]
    extracted_route_full: list[dict[str, Any]]
    extracted_route_clean: Optional[_MazeTypePath]
    outcome: MazeOutcome
    lm_1_name: str
    lm_1_provider: _SupportedProviders
    lm_2_name: Optional[str] = None
    lm_2_provider: Optional[_SupportedProviders] = None
    judge_model_name: Optional[str] = None
    judge_model_provider: Optional[str] = None
    # Reasoning effort actually in force for each model, read back off the LM
    # rather than from the CLI argument: LMFactory clears the requested value
    # for models that do not declare ReasoningSupport, so these record the
    # condition that ran, not the one that was asked for.
    lm_1_reasoning_effort: Optional[_ReasoningEffort] = None
    lm_2_reasoning_effort: Optional[_ReasoningEffort] = None
    judge_model_reasoning_effort: Optional[_ReasoningEffort] = None
    show_same_agent_value: Optional[bool] = None
    time_to_complete: float = 0.0
    critic_steps: Optional[int] = None
    pre_critic_outcome: Optional[MazeOutcome] = None
    pre_critic_extracted_route_schema: Optional[dict] = None
    pre_critic_extracted_route_full: Optional[list[dict[str, Any]]] = None
    continuation_filepath: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'MazeTaskSaveResult':
        """Load the dataclass from a dictionary."""
        return cls(
            task_path=data.get('task_path', ''),
            maze_str=data.get('maze_str', ''),
            maze_coordinates=data.get('maze_coordinates', []),
            user_intermediate=data.get('user_intermediate', True),
            split_view=data.get('split_view', False),
            split_1=data.get('split_1', []),
            split_2=data.get('split_2', []),
            messages=data.get('messages', []),
            max_messages=data.get('max_messages', 0),
            num_messages=data.get('num_messages', 0),
            extracted_route_schema=data.get('extracted_route_schema', None),
            extracted_route_full=data.get('extracted_route_full', []),
            extracted_route_clean=data.get('extracted_route_clean', None),
            outcome=MazeOutcome.from_dict(data.get('outcome', {})),
            lm_1_name=data.get('lm_1_name', ''),
            lm_1_provider=data.get('lm_1_provider', ''),
            lm_2_name=data.get('lm_2_name', None),
            lm_2_provider=data.get('lm_2_provider', None),
            judge_model_name=data.get('judge_model_name', None),
            judge_model_provider=data.get('judge_model_provider', None),
            lm_1_reasoning_effort=data.get('lm_1_reasoning_effort', None),
            lm_2_reasoning_effort=data.get('lm_2_reasoning_effort', None),
            judge_model_reasoning_effort=data.get(
                'judge_model_reasoning_effort', None),
            show_same_agent_value=data.get('show_same_agent_value', None),
            time_to_complete=data.get('time_to_complete', 0.0),
            critic_steps=data.get('critic_steps', None),
            pre_critic_extracted_route_schema=data.get(
                'pre_critic_extracted_route_schema', None),
            pre_critic_extracted_route_full=data.get(
                'pre_critic_extracted_route_full', None),
            pre_critic_outcome=data.get('pre_critic_outcome', None),
            continuation_filepath=data.get('continuation_filepath', None)
        )


@dataclass
class LoadedMazeGameResult:
    """Object representing the result of a loaded maze game.

    """
    maze_task_config: MazeTaskConfig
    maze_game: MazeGame
    messages: list[Message]
    outcome: MazeOutcome
    filepath: str


@dataclass
class MazeContinuation:
    """Object used to continue a maze game from a pre-existing interaction."""
    maze_game: MazeGame
    agent_1_messages: list[Message]
    agent_2_messages: list[Message]
    outcome: MazeOutcome
    filepath: str


###############################################################################
# Post-process Data
###############################################################################
def extract_maze_route(
        lm_grader: LM,
        conversation: Conversation,
        lm_1_id: str,
        lm_2_id: Optional[str] = None,
) -> ExtractedMazeRoute:
    """Extract the route taken by the agent in the maze.
    Args:
        lm (LM): Language model instance.
        messages (list[Message]): List of messages in the conversation.
        maze (MazeGame): Instance of the MazeGame.
    Returns:
        list[tuple[int, int]]: List of coordinates representing the route.
    """
    mapping = {
        lm_1_id: "agent_1",
        _REAL_USER_AGENT_ID: "user",
    }
    if lm_2_id:
        mapping[lm_2_id] = "agent_2"
    dialogue = conversation.as_transcript(
        mapping=mapping,
        show_real_user_message=False,
        show_system_message=False
    )
    user_message = UserMessage(content=render_template(
        template_name="outcome_evaluation_collab.j2",
        user_data={
            "dialogue": dialogue.strip()
        }
    ))

    x_solution = lm_grader(
        [user_message], num_samples=1, return_as_message=True)

    if isinstance(x_solution, list) and len(x_solution) == 1:
        x_solution = x_solution[0]
    if not isinstance(x_solution, AssistantMessage):
        raise ValueError(
            f"Expected an AssistantMessage, but got {type(x_solution)}")

    extracted_maze_route = _extract_maze_route_from_yaml_response(x_solution)

    return extracted_maze_route


def extract_maze_route_one_shot(
    lm_grader: LM,
    data: str
) -> ExtractedMazeRoute:
    """Extract the route taken by the agent in the maze from a one-shot example.

    Args:
        lm (LM): Language model instance.
        data (str): The one-shot example containing the maze and conversation.

    Returns:
        list[tuple[int, int]]: List of coordinates representing the route.
    """

    user_message = UserMessage(content=render_template(
        template_name="outcome_evaluation_one_shot.j2",
        user_data={
            "proposed_solution": data.strip()
        }
    ))
    x_solution = lm_grader(
        [user_message], num_samples=1, return_as_message=True)
    if isinstance(x_solution, list) and len(x_solution) == 1:
        x_solution = x_solution[0]
    if not isinstance(x_solution, AssistantMessage):
        raise ValueError(
            f"Expected an AssistantMessage, but got {type(x_solution)}")

    extracted_maze_route = _extract_maze_route_from_yaml_response(x_solution)

    return extracted_maze_route


def _extract_maze_route_from_yaml_response(
        response: AssistantMessage) -> ExtractedMazeRoute:
    extracted_yaml = ExtractedMazeRoute(
        route=[],
        route_schema=None,
    )

    try:
        solution = yaml.safe_load(response.content.strip().replace(
            "```yaml", "").replace("```", ""))
        extracted_yaml.route = solution.get('route', [])
        route_schema = solution.get('route_schema', {})
        if route_schema:
            try:
                extracted_yaml.route_schema = SolutionType.from_dict(
                    route_schema)
            except Exception as e:
                extracted_yaml.error = f"Failed to extract route schema: {e}"
    except Exception as e:
        extracted_yaml.error = f"Failed to parse YAML: {e}"

    return extracted_yaml


###############################################################################
# Load / Parse Existing Data
###############################################################################


def _get_agent_messages(messages: list[Message]) -> list[Message]:
    """
    Extract messages that have a model_name set, indicating they are from agents.
    """
    return [msg for msg in messages if msg.model_name is not None]


def _annotate_pre_maze_game_messages(
    messages: list[Message],
    lm_1: LM,
    lm_2: LM,
    force_model_match: bool = True,
    force_model_match_lm_1: Optional[bool] = None,
    force_model_match_lm_2: Optional[bool] = None
):
    """
    Add agent_id to messages to align them with the new interaction.
    Either:
    - Use the turn number to assign agent IDs (0 -> lm_1, 1 -> lm_2)
    - Or match the model_name of the messages

    Args:
        messages (list[Message]): List of messages to annotate.
        lm_1 (LM): First language model instance.
        lm_2 (LM): Second language model instance.
        force_model_match (bool): If True, ensure that both model names match the provided LMs.
        force_model_match_lm_1 (Optional[bool]): If True, force model match for lm_1.
        force_model_match_lm_2 (Optional[bool]): If True, force model match for lm_2.

    Returns:
        list[Message]: List of messages with updated agent IDs.
    """

    if force_model_match and (force_model_match_lm_1 or force_model_match_lm_2):
        raise ValueError(
            "Cannot use force_model_match with force_model_match_lm_1 or force_model_match_lm_2. "
            "Please choose one of the options."
        )

    # Check if the provided LMs are the same model
    same_model = lm_1.adapter.model_name == lm_2.adapter.model_name
    # Retrieve the messages generated by the agents
    agent_messages = _get_agent_messages(messages)
    msg_id_to_lm_map = dict()
    # We keep track of the "turn" to assign to lm_1 or lm_2 (if needed)
    for turn, msg in enumerate(agent_messages):

        msg_model_name = msg.model_name
        msg_agent_id = msg.agent_id
        # Only assign if not already mapped
        if msg_agent_id and msg_agent_id not in msg_id_to_lm_map and msg_agent_id != _REAL_USER_AGENT_ID:

            if force_model_match:
                if same_model:
                    msg_id_to_lm_map[msg_agent_id] = lm_1.unique_id if turn == 0 else lm_2.unique_id
                else:
                    if deployment_names_match(
                            msg_model_name, lm_1.adapter.model_name):
                        msg_id_to_lm_map[msg_agent_id] = lm_1.unique_id
                    elif deployment_names_match(
                            msg_model_name, lm_2.adapter.model_name):
                        msg_id_to_lm_map[msg_agent_id] = lm_2.unique_id
                    else:
                        raise ValueError(
                            f"Message model name {msg_model_name} does not match either LM's model name.")
            elif force_model_match_lm_1:
                if deployment_names_match(
                        msg_model_name, lm_1.adapter.model_name):
                    msg_id_to_lm_map[msg_agent_id] = lm_1.unique_id
                else:
                    msg_id_to_lm_map[msg_agent_id] = lm_2.unique_id
            elif force_model_match_lm_2:
                if deployment_names_match(
                        msg_model_name, lm_2.adapter.model_name):
                    msg_id_to_lm_map[msg_agent_id] = lm_2.unique_id
                else:
                    msg_id_to_lm_map[msg_agent_id] = lm_1.unique_id
            else:
                msg_id_to_lm_map[msg_agent_id] = lm_1.unique_id if turn == 0 else lm_2.unique_id

    if len(msg_id_to_lm_map) != 2:
        raise ValueError("Expected exactly two unique agent IDs.")

    for msg in agent_messages:
        if msg.role == RoleType.ASSISTANT:
            msg.agent_id = msg_id_to_lm_map[msg.agent_id]
        else:
            # User/System messages should always have the real user agent ID
            msg.agent_id = _REAL_USER_AGENT_ID

    return agent_messages


def _convert_dicts_to_messages(
    messages: list[dict[str, Any]],
) -> list[Message]:
    message_objects = []
    for msg in messages:
        if msg['model_name'] is not None:
            msg_type = AssistantMessage
        else:
            msg_type = UserMessage
        message_objects.append(
            msg_type(
                content=msg['message'],
                model_name=msg['model_name'],
                model_provider=msg['model_provider'],
                agent_id=msg['agent_id'],
                # Absent from transcripts saved before reasoning was recorded,
                # and from non-reasoning turns.
                reasoning_content=msg.get('reasoning_content'),
                reasoning_tokens=msg.get('reasoning_tokens'),
            )
        )

    return message_objects


def _load_messages_and_game_from_file(filepath) -> LoadedMazeGameResult:

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"The specified file does not exist: {filepath}"
        )

    with open(filepath, 'r') as f:
        pre_maze_result = MazeTaskSaveResult.from_dict(yaml.safe_load(f))
    with open(filepath.replace('result.yaml', 'task_config.yaml'), 'r') as f:
        pre_maze_task_config = MazeTaskConfig.from_dict(yaml.safe_load(f))

    pre_maze_config = pre_maze_task_config.config
    maze_game = MazeGame(
        pre_maze_config,
        maze=pre_maze_result.maze_coordinates,
        maze_split_1=pre_maze_result.split_1,
        maze_split_2=pre_maze_result.split_2
    )

    messages = _convert_dicts_to_messages(pre_maze_result.messages)

    return LoadedMazeGameResult(
        maze_task_config=pre_maze_task_config,
        maze_game=maze_game,
        messages=messages,
        outcome=pre_maze_result.outcome,
        filepath=filepath
    )


def _load_existing_maze_game_messages_from_file(
    lm_1: LM,
    lm_2: LM,
    pre_maze_game_config: PreMazeGameConfig
) -> LoadedMazeGameResult:

    if pre_maze_game_config.specific_filepath is None:

        if pre_maze_game_config.overview_filepath is None:
            raise ValueError(
                "Either overview_filepath or specific_filepath must"
                "be provided in PreMazeGameConfig."
            )
        df_overview = pd.read_csv(pre_maze_game_config.overview_filepath)

        # Ensure has the necessary models
        pre_lm_1_name, pre_lm_1_provider = (pre_maze_game_config.lm_1_name,
                                            pre_maze_game_config.lm_1_provider)
        pre_lm_2_name, pre_lm_2_provider = (pre_maze_game_config.lm_2_name,
                                            pre_maze_game_config.lm_2_provider)

        if pre_lm_1_name is not None and pre_lm_1_provider is not None:
            pre_lm_1_details = lookup_model_details(
                pre_lm_1_name, ProviderAPI(pre_lm_1_provider))
            pre_lm_1_name = pre_lm_1_details.model_deployment_name
        if pre_lm_2_name is not None and pre_lm_2_provider is not None:
            pre_lm_2_details = lookup_model_details(
                pre_lm_2_name, ProviderAPI(pre_lm_2_provider))
            pre_lm_2_name = pre_lm_2_details.model_deployment_name

        # Compare on the canonical spelling: an overview may record the
        # deployment as gpt-4.1-mini_2025-04-14 while lookup_model_details
        # returns gpt-4.1-mini-2025-04-14, and an exact comparison would
        # silently match nothing.
        overview_lm_1 = df_overview['lm_1_name'].map(normalize_deployment_name)
        overview_lm_2 = df_overview['lm_2_name'].map(normalize_deployment_name)

        if pre_lm_1_name is not None:
            df_overview = df_overview[(
                overview_lm_1 == normalize_deployment_name(pre_lm_1_name)
            )].copy()
            overview_lm_1 = df_overview['lm_1_name'].map(normalize_deployment_name)
            overview_lm_2 = df_overview['lm_2_name'].map(normalize_deployment_name)
        if pre_lm_2_name is not None:
            df_overview = df_overview[(
                overview_lm_2 == normalize_deployment_name(pre_lm_2_name)
            )].copy()
            overview_lm_1 = df_overview['lm_1_name'].map(normalize_deployment_name)
            overview_lm_2 = df_overview['lm_2_name'].map(normalize_deployment_name)

        if pre_maze_game_config.force_model_match:
            df_overview = df_overview[(
                (overview_lm_1 == normalize_deployment_name(lm_1.adapter.model_name)) &
                (overview_lm_2 == normalize_deployment_name(lm_2.adapter.model_name))
            )].copy()
        elif pre_maze_game_config.force_model_match_lm_1:
            df_overview = df_overview[(
                overview_lm_1 == normalize_deployment_name(lm_1.adapter.model_name)
            )].copy()
        elif pre_maze_game_config.force_model_match_lm_2:
            df_overview = df_overview[(
                overview_lm_2 == normalize_deployment_name(lm_2.adapter.model_name)
            )].copy()
        else:
            pass

        # Filter by maze criteria
        if pre_maze_game_config.split_1_proportion is not None:
            df_overview = df_overview[(
                df_overview['split_1_proportion'] == pre_maze_game_config.split_1_proportion
            )].copy()
        if pre_maze_game_config.split_2_proportion is not None:
            df_overview = df_overview[(
                df_overview['split_2_proportion'] == pre_maze_game_config.split_2_proportion
            )].copy()
        if pre_maze_game_config.maze_size is not None:
            mask = df_overview['maze_size'].apply(
                lambda size: list(ast.literal_eval(size)) == pre_maze_game_config.maze_size)
            df_overview = df_overview[mask].copy()

        # Filter by outcome criteria
        # NOTE: we only apply min/max outcome if success is False
        num_valid_files = len(df_overview)
        if pre_maze_game_config.success is not None:
            if "success" in df_overview.columns:
                df_overview = df_overview[(
                    df_overview['success'] == pre_maze_game_config.success)].copy()
            elif "outcome_success" in df_overview.columns:
                df_overview = df_overview[(
                    df_overview['outcome_success'] == pre_maze_game_config.success)].copy()
            else:
                raise ValueError(
                    "The overview file does not contain a 'success' or 'outcome_success' column."
                )
        if pre_maze_game_config.min_outcome is not None and not pre_maze_game_config.success:
            df_overview = df_overview[(
                df_overview['weighted_outcome'] >= pre_maze_game_config.min_outcome)].copy()
        if pre_maze_game_config.max_outcome is not None and not pre_maze_game_config.success:
            df_overview = df_overview[(
                df_overview['weighted_outcome'] <= pre_maze_game_config.max_outcome)].copy()

        if df_overview.empty:
            raise ValueError(
                f"No pre-maze interactions found for {lm_1.adapter.model_name}"
                f" and {lm_2.adapter.model_name} with the specified criteria."
                f"--> {num_valid_files} total interactions found"
            )

        # Sort before sampling so the seed alone decides which game is drawn,
        # whatever order the overview CSV happens to list them in. Relays at
        # different relay points then resume the same game for the same seed.
        df_overview = df_overview.sort_values('filepath').reset_index(drop=True)
        sample_row = df_overview.sample(n=1)
        print_c(
            f"Sampled pre_maze_game file: {sample_row.iloc[0]['filepath']}\n"
            f"Details: {sample_row.iloc[0].to_dict()}",
            ColorType.GREEN
        )
        sampled_filepath = sample_row.iloc[0]['filepath']
    else:
        sampled_filepath = pre_maze_game_config.specific_filepath
    sampled_filepath = sampled_filepath.replace('../../data', './data')

    loaded_result = _load_messages_and_game_from_file(sampled_filepath)

    return loaded_result


def _load_relay_game_interactions(
    lm_1: LM,
    lm_2: LM,
    show_same_agent_value: Optional[bool] = None,
    pre_maze_game_config: Optional[PreMazeGameConfig] = None
) -> MazeContinuation:

    if pre_maze_game_config is None:
        raise ValueError("pre_maze_game_config must be provided.")

    loaded_result = _load_existing_maze_game_messages_from_file(
        lm_1, lm_2, pre_maze_game_config)

    if pre_maze_game_config.relay_point is None:
        raise ValueError(
            "relay_point must be specified in PreMazeGameConfig for relay games."
        )

    annotated_messages = _annotate_pre_maze_game_messages(
        loaded_result.messages,
        lm_1,
        lm_2,
        force_model_match=pre_maze_game_config.force_model_match
    )

    annotated_messages = annotated_messages[:pre_maze_game_config.relay_point]

    game_instruction_message_1, game_instruction_message_2 = _get_game_instruction_messages_collab(
        loaded_result.maze_game,
        loaded_result.maze_task_config,
        user_intermediate=True,
        show_same_agent_value=show_same_agent_value
    )

    relay_game_1: list[Message] = [game_instruction_message_1] + \
        annotated_messages  # type: ignore
    relay_game_2: list[Message] = [game_instruction_message_2] + \
        annotated_messages  # type: ignore

    return MazeContinuation(
        maze_game=loaded_result.maze_game,
        agent_1_messages=relay_game_1,
        agent_2_messages=relay_game_2,
        outcome=loaded_result.outcome,
        filepath=loaded_result.filepath
    )


def _get_game_instruction_messages_collab(
    maze: MazeGame,
    task_config: MazeTaskConfig,
    user_intermediate: bool,
    show_same_agent_value: Optional[bool]
) -> tuple[UserMessage, UserMessage]:
    maze_1_str, maze_2_str = maze.get_maze_splits(
        return_as_string=True, add_legend=True)

    game_instruction = task_config.prompts["system_instruction_collab"]
    game_instruction_1 = render_template(
        template_name=game_instruction,
        user_data={
            "map_str": maze_1_str,
            "hidden_coordinates": maze.config.split_1_proportion < 1.0,
            "traps": maze.config.trap_density > 0.,
            "user_intermediate": user_intermediate,
            "language": task_config.agent_1_language,
            "same_agent": show_same_agent_value
        }
    ).strip()
    game_instruction_2 = render_template(
        template_name=game_instruction,
        user_data={
            "map_str": maze_2_str,
            "hidden_coordinates": maze.config.split_2_proportion < 1.0,
            "traps": maze.config.trap_density > 0.,
            "user_intermediate": user_intermediate,
            "language": task_config.agent_2_language,
            "same_agent": show_same_agent_value
        }
    ).strip()

    game_instruction_message_1 = UserMessage(
        content=game_instruction_1,
        agent_id=_REAL_USER_AGENT_ID,
    )
    game_instruction_message_2 = UserMessage(
        content=game_instruction_2,
        agent_id=_REAL_USER_AGENT_ID,)

    return game_instruction_message_1, game_instruction_message_2


###############################################################################
# Generate Data
###############################################################################


def run_maze_task_collab(
    task_config: MazeTaskConfig,
    lm_1: LM,
    lm_2: LM,
    max_turns: int = 50,
    user_intermediate: bool = False,
    show_same_agent_value: Optional[bool] = None
) -> MazeTaskResult:
    """
    Run a specified task using two language models.

    Args:
        task_path (str): Path to the task module.
        lm_1 (LM): First language model instance.
        lm_2 (LM): Second language model instance.
        max_turns (int): Maximum number of turns for the task. Defaults to 50.
    """

    maze = MazeGame(task_config.config)
    game_instruction_message_1, game_instruction_message_2 = _get_game_instruction_messages_collab(
        maze,
        task_config,
        user_intermediate,
        show_same_agent_value
    )

    communication_protocol_message_1 = SystemMessage(content=render_template(
        template_name=_COMMUNICATION_PROTOCOL_FILE,
        user_data={"language": task_config.agent_1_language, }
    ))
    communication_protocol_message_2 = SystemMessage(content=render_template(
        template_name=_COMMUNICATION_PROTOCOL_FILE,
        user_data={"language": task_config.agent_2_language, }
    ))

    pre_game_config = task_config.pre_maze_game_config
    # For a relay game, we do not want the instructions to be added again
    is_relay_game = False
    if (
        pre_game_config and
        pre_game_config.has_file and
        isinstance(pre_game_config.relay_point, int) and
        pre_game_config.relay_point > 0
    ):
        is_relay_game = True

    add_instructions = True
    game_instruction_preamble = ""
    continuation_filepath = None
    if pre_game_config and pre_game_config.has_file:
        if pre_game_config.continuation_type != _RELAY_CONTINUATION:
            raise NotImplementedError(
                f"Unsupported continuation type: "
                f"{pre_game_config.continuation_type!r}. Only "
                f"{_RELAY_CONTINUATION!r} is supported."
            )
        # Relay: resume a prior collaborative game from its relay point. The
        # MazeGame instance is replaced by the one the prior game was played on,
        # and instructions are not re-added because the loaded transcript
        # already opens with them.
        add_instructions = False
        maze_continuation = _load_relay_game_interactions(
            lm_1=lm_1,
            lm_2=lm_2,
            show_same_agent_value=show_same_agent_value,
            pre_maze_game_config=pre_game_config
        )
        continuation_filepath = maze_continuation.filepath
        maze = maze_continuation.maze_game
        pre_interactions_1 = maze_continuation.agent_1_messages
        pre_interactions_2 = maze_continuation.agent_2_messages
    else:
        pre_interactions_1, pre_interactions_2 = [], []

    if pre_interactions_1 or pre_interactions_2:
        context_1: list[Message] = [communication_protocol_message_1] + \
            pre_interactions_1  # type: ignore
        context_2: list[Message] = [communication_protocol_message_2] + \
            pre_interactions_2  # type: ignore

        if add_instructions:
            game_instruction_message_1.content = (
                game_instruction_preamble +
                "\n\n" + game_instruction_message_1.content
            )
            game_instruction_message_2.content = (
                game_instruction_preamble +
                "\n\n" + game_instruction_message_2.content
            )
            context_1.append(game_instruction_message_1)
            context_2.append(game_instruction_message_2)

    else:
        context_1: list[Message] = [
            communication_protocol_message_1,
            game_instruction_message_1
        ]
        context_2: list[Message] = [
            communication_protocol_message_2,
            game_instruction_message_2
        ]

    if add_instructions:
        log_c(
            "STARTING DIALOGUE\n\n"
            f"[system message 1]\n{communication_protocol_message_1.content}\n\n--"
            f"[game message 1]\n{game_instruction_message_1.content}\n\n--"
            f"[system message 2]\n{communication_protocol_message_2.content}\n\n--"
            f"[game message 2]\n{game_instruction_message_2.content}",
            logger, ColorType.YELLOW)
    else:
        log_c(
            "STARTING DIALOGUE\n\n"
            f"[system message 1]\n{communication_protocol_message_1.content}\n\n--"
            f"[game message 1]\n{context_1[1].content}\n\n--"
            f"[system message 2]\n{communication_protocol_message_2.content}\n\n--"
            f"[game message 2]\n{context_2[1].content}",
            logger, ColorType.YELLOW)

    messages_1: list[Message] = [msg for msg in context_1]
    messages_2: list[Message] = [msg for msg in context_2]
    messages_to_verify: list[Message] = []
    for turn in range(max_turns):
        log_c(
            f"\nTURN: {turn + 1}/{max_turns} - history length: {len(messages_1)}\n",
            logger, ColorType.YELLOW)

        if turn % 2 == 0:
            current_lm = lm_1
            current_messages = messages_1
            print_color = ColorType.BLUE
        else:
            current_lm = lm_2
            current_messages = messages_2
            print_color = ColorType.RED

        messages_in_loop = align_message_roles(
            current_messages,
            current_lm.unique_id,
            real_user_agent_id=_REAL_USER_AGENT_ID
        )
        if user_intermediate:
            messages_in_loop = annotate_messages_with_prefix(
                messages_in_loop,
                map_by_role={
                    RoleType.USER: "other_agent",
                    RoleType.ASSISTANT: None
                },
                agent_id_overrides={_REAL_USER_AGENT_ID: "user"}
            )

        next_message = current_lm(
            messages_in_loop, num_samples=1, return_as_message=True)

        if isinstance(next_message, list) and len(next_message) == 1:
            next_message = next_message[0]

        if not isinstance(next_message, AssistantMessage):
            raise ValueError(
                f"Expected an AssistantMessage, but got {type(next_message)}")

        log_c(
            f"[agent_{current_lm.unique_id[:4]} | "
            f"{current_lm.adapter.model_name}]:\n\n{next_message.content}",
            logger,
            print_color
        )

        messages_1.append(next_message)
        messages_2.append(next_message)
        messages_to_verify.append(next_message)

        # We finish the game when the target phrase is uttered, but we check to
        # make sure this was not an accident, e.g., on turn <= 4.
        if (
            _FINISHED_GAME_PHRASE.lower() in next_message.content.lower() or
            _FINISHED_GAME_PHRASE.replace("!", "") in next_message.content
        ):
            finish_game = False
            if is_relay_game:
                if pre_game_config and pre_game_config.relay_point is not None:
                    real_turn = turn + pre_game_config.relay_point
                    if real_turn > 4:
                        finish_game = True
            else:
                if turn > 4:
                    finish_game = True

            if finish_game:
                log_c(
                    "Task completed within max turns!",
                    logger, ColorType.GREEN)
                break

    result = Conversation(messages=messages_1)
    result_to_verify = Conversation(messages=messages_to_verify)
    if pre_game_config and pre_game_config.has_file:
        result_to_verify = Conversation(
            messages=_get_agent_messages(messages_1))

    return MazeTaskResult(
        lm_1=lm_1,
        lm_2=lm_2,
        task_config=task_config,
        game=maze,
        result=result,
        result_to_verify=result_to_verify,
        continuation_filepath=continuation_filepath
    )


def run_maze_one_shot(
    task_config: MazeTaskConfig,
    lm_1: LM,
    split_view: bool = False,
    critic_steps: int = 0
) -> MazeTaskResult:
    """
    Run a maze task with a one-shot example.

    Args:
        task_config (MazeTaskConfig): Configuration for the maze task.
        lm_1 (LM): Language model instance for the first agent.
        split_view (bool): Whether to display the maze in split view. Defaults to False.
        critic_steps (int): Number of steps for the critic model. Defaults to 0.

    Returns:
        MazeTaskResult: Result of the maze task
    """

    maze = MazeGame(task_config.config)
    if split_view:
        maze_str = maze.display_split_view(add_legend=True)
    else:
        maze_str = maze.display_maze(add_legend=True)

    communication_protocol_message = SystemMessage(content=render_template(
        template_name=_COMMUNICATION_PROTOCOL_FILE,
        user_data={"language": task_config.agent_1_language, }
    ))
    print_c(
        f"[system message]\n{communication_protocol_message.content}", ColorType.YELLOW)

    game_instruction_file = task_config.prompts["system_instruction_one_shot"]
    game_instruction_message = UserMessage(
        content=render_template(
            template_name=game_instruction_file,
            user_data={
                "map_str": maze_str,
                "split_view": split_view,
                "user_intermediate": False,  # One-shot mode does not use user interaction
                "language": task_config.agent_1_language,
                "traps": maze.config.trap_density > 0.,
            }
        ).strip(),
        agent_id=_REAL_USER_AGENT_ID,
    )
    print_c(
        f"[game message]\n{game_instruction_message.content}", ColorType.YELLOW)

    messages: list[Message] = [
        communication_protocol_message,
        game_instruction_message
    ]

    messages_to_verify: list[Message] = []
    completion = lm_1(messages, num_samples=1, return_as_message=True)
    if isinstance(completion, list) and len(completion) == 1:
        completion = completion[0]
    if not isinstance(completion, AssistantMessage):
        raise ValueError(
            f"Expected an AssistantMessage, but got {type(completion)}")

    messages_to_verify.append(completion)
    print_c(f"[agent_1]: {completion.content}", ColorType.BLUE)

    if isinstance(critic_steps, int) and critic_steps > 0:
        critic_message = UserMessage(
            content=render_template(
                template_name="game_mode_one_shot_critic_step.j2"),
            agent_id=_REAL_USER_AGENT_ID,
        )
        messages.append(completion)
        for critic_step in range(critic_steps):
            log_c(
                f"\nCRITIC STEP: {critic_step + 1}/{critic_steps}\n",
                logger, ColorType.YELLOW)
            messages.append(critic_message)
            messages_in_loop = align_message_roles(
                messages,
                next_agent_id=lm_1.unique_id,
                real_user_agent_id=_REAL_USER_AGENT_ID
            )
            next_message = lm_1(
                messages_in_loop, num_samples=1, return_as_message=True)

            if isinstance(next_message, list) and len(next_message) == 1:
                next_message = next_message[0]

            if not isinstance(next_message, AssistantMessage):
                raise ValueError(
                    f"Expected an AssistantMessage, but got {type(next_message)}")

            messages_to_verify.append(next_message)
            log_c(
                f"[agent_1 | {lm_1.adapter.model_name}]:\n\n{next_message.content}",
                logger, ColorType.BLUE)

            messages.append(next_message)
    else:
        messages.append(completion)

    result = Conversation(messages=messages)
    result_to_verify = Conversation(messages=messages_to_verify)

    return MazeTaskResult(
        lm_1=lm_1,
        lm_2=None,  # No second model in one-shot mode
        task_config=task_config,
        game=maze,
        result=result,
        result_to_verify=result_to_verify
    )


def save_result_maze(
    save_object: MazeTaskSaveResult,
    task_config: MazeTaskConfig,
    folder_name: str
) -> None:

    save_dict = None
    try:
        save_dict = asdict(save_object)
    except TypeError as e:
        print_c(f"Error converting save_object to dict: {e}", ColorType.RED)
        raise
    log_c("Saving result...", logger, ColorType.GREEN)

    save_object = copy.deepcopy(save_object)
    with open(os.path.join(folder_name, "result.yaml"), 'w') as file:
        yaml.safe_dump(
            save_dict,
            file,
            default_flow_style=False,
            indent=4
        )
    log_c(
        f"Result saved to {folder_name}/result.yaml", logger, ColorType.GREEN)

    # Save the task configuration
    task_config_path = os.path.join(folder_name, "task_config.yaml")
    with open(task_config_path, 'w') as file:
        yaml.safe_dump(
            task_config.to_dict(),
            file,
            default_flow_style=False,
            indent=4
        )


def play_maze_task(
    task_config: MazeTaskConfig,
    lm_1: LM,
    lm_2: Optional[LM],
    lm_grader: LM,
    max_turns: int = 20,
    game_mode: _GameModeTypes = "collab",
    split_view: bool = False,
    critic_steps: int = 0,
    user_intermediate: bool = False,
    job_folder="maze_games",
    job_id: Optional[str] = None,
    show_same_agent: bool = False,
    show_same_is_model_based: bool = True,
    show_same_override_value: bool = False
) -> None:
    """
    Play a maze task with specified language models and configurations.
    Args:
        task_config (MazeTaskConfig): Configuration for the maze task.
        lm_1 (LM): Language model instance for the first agent.
        lm_2 (Optional[LM]): Language model instance for the second agent (if applicable).
        lm_grader (LM): Language model instance for grading the solution.
        max_turns (int): Maximum number of turns for the task. Defaults to 20.
        game_mode (_GameModeTypes): Mode of the game, 'collab' or 'one-shot'.
        split_view (bool): Whether to display the maze in split view. Defaults to False.
        critic_steps (int): Number of steps for the critic model in one-shot mode. Defaults to 0.
        user_intermediate (bool): Whether to allow user intermediate messages. Defaults to False.
        job_folder (str): Folder name to save results. Defaults to "maze_games".
        job_id (Optional[str]): Unique identifier for the job. If None, a random ID is generated.
        show_same_agent (bool): Whether to tell the agents the identity of the other agent.
        show_same_is_model_based (bool): Whether the identity is based on the actual models used, or overridden by the user.
        show_same_override_value (bool): What to override the same_agent identity value to.
    """
    start_str = (
        MazeGame.get_maze_game_ascii_art() + "\n\n" +
        f"Starting maze task with:\n"
        f"Game mode: \t\t {game_mode}\n"
        f"Grid size: \t\t {task_config.config.size}\n"
        f"Max turns: \t\t {max_turns}\n"
        f"Split view: \t\t {split_view}\n"
        f"Critic steps: \t\t {critic_steps}\n"
        f"User intermediate: \t {user_intermediate}\n"
        f"Language 1: \t\t {task_config.agent_1_language}\n"
        f"Language 2: \t\t {task_config.agent_2_language}\n"
        f"Players:\n  - {lm_1.adapter.model_name} ({lm_1.adapter.provider.value})\n"
    )
    if lm_2 is not None:
        start_str += f"  - {lm_2.adapter.model_name} ({lm_2.adapter.provider.value})\n"
    start_str += f"Grader: \t\t {lm_grader.adapter.model_name} ({lm_grader.adapter.provider.value})\n"
    start_str += (
        f"Show Model ID: \t\t {show_same_agent}\n"
        f"Model ID = Model-Based: \t\t {show_same_is_model_based}\n"
        f"Mdel ID Override Value: \t\t {show_same_override_value}\n"
    )

    # Create folder to save results:
    if job_id is None:
        job_id = str(uuid4())[:5]
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"./data/task_results/{job_folder}/" + f"{timestamp}_{job_id}"
    os.makedirs(save_path, exist_ok=True)

    start_str += f"Results will be saved to: {save_path}\n"
    log_c(start_str, logger, ColorType.BLUE)

    start_time = time.time()
    show_same_agent_value = None
    pre_critic_outcome = None
    pre_critic_solution = None
    if game_mode == "collab":
        assert lm_2 is not None, "lm_2 must be provided for collaborative mode"

        if show_same_agent:
            if show_same_is_model_based:
                show_same_agent_value = lm_1.adapter.model_name == lm_2.adapter.model_name
            else:
                show_same_agent_value = show_same_override_value

        task_result = run_maze_task_collab(
            lm_1=lm_1,
            lm_2=lm_2,
            task_config=task_config,
            max_turns=max_turns,
            user_intermediate=user_intermediate,
            show_same_agent_value=show_same_agent_value
        )
        solution = extract_maze_route(
            lm_1_id=lm_1.unique_id,
            lm_2_id=lm_2.unique_id,
            lm_grader=lm_grader,
            conversation=task_result.result_to_verify
        )
    elif game_mode == "one-shot":
        task_result = run_maze_one_shot(
            lm_1=lm_1,
            task_config=task_config,
            split_view=split_view,
            critic_steps=critic_steps,
        )
        agent_messages = []
        if task_result.result_to_verify.messages:
            agent_messages = [
                msg for msg in task_result.result_to_verify.messages
                if msg.agent_id == lm_1.unique_id]
        solution = extract_maze_route_one_shot(
            lm_grader=lm_grader,
            data=agent_messages[-1].content if agent_messages else ""
        )
        if critic_steps > 0 and len(agent_messages) > 1:
            first_attempt = agent_messages[0].content
            pre_critic_solution = extract_maze_route_one_shot(
                lm_grader=lm_grader,
                data=first_attempt
            )
            pre_critic_outcome = task_result.game.verify_solution(
                pre_critic_solution.processed_route,  # type: ignore
                pre_critic_solution.route_schema
            )
    else:
        raise ValueError(
            f"Invalid play mode: {game_mode}. Choose from 'collab' or 'one-shot'.")

    # Verify the solution using the maze game
    maze_outcome = task_result.game.verify_solution(
        solution.processed_route,  # type: ignore
        solution.route_schema
    )
    # Annotate messages with model names and providers
    annotated_messages = conversation_to_dicts(
        result=task_result.result,
        skip_system=True
    )

    log_c(f'\n\nReal Map:\n\n{task_result.game.display_maze(
        add_space=True, border_style="simple", add_legend=False)}\n\n',
        logger, ColorType.BLUE)
    optimal_route = task_result.game.find_optimal_solution()
    if optimal_route is not None:
        try:
            log_c(
                f"Optimal Route:\n\n"
                f"{task_result.game.display_solution(optimal_route)}\n\n",
                logger, ColorType.BLUE)
        except Exception as e:
            log_c(
                f'Error displaying optimal route: {e}', logger, ColorType.RED)
    if maze_outcome.outcome_path is not None:
        try:
            log_c(
                f"Proposed Solution\n\n"
                f"{task_result.game.display_solution(maze_outcome.outcome_path)}\n\n"
                f"Proposed Solution raw:\n\n"
                f"{solution}\n\n",
                logger,
                ColorType.GREEN)
        except Exception as e:
            log_c(f'Error displaying solution: {e}', logger, ColorType.RED)
            log_c(
                f'Proposed Solution: {maze_outcome.outcome_path}',
                logger, ColorType.YELLOW)
    else:
        log_c(
            f'Extracted Route: {solution.route}', logger, ColorType.YELLOW)
    log_c(f'Outcome: {maze_outcome}', logger, ColorType.GREEN)

    if pre_critic_outcome is not None and pre_critic_outcome.outcome_path is not None:
        log_c(
            f"Pre-Critic Proposed Solution\n\n"
            f"{task_result.game.display_solution(pre_critic_outcome.outcome_path)}\n\n",
            logger,
            ColorType.GREEN)

    total_time = float(time.time() - start_time)
    save_object = MazeTaskSaveResult(
        task_path=task_config.fname,
        maze_str=task_result.game.display_maze(add_legend=False),
        maze_coordinates=task_result.game.maze,
        user_intermediate=user_intermediate,
        split_view=split_view,
        split_1=task_result.game.get_maze_splits(return_as_string=False)[0],
        split_2=task_result.game.get_maze_splits(return_as_string=False)[1],
        messages=annotated_messages,
        max_messages=max_turns,
        num_messages=len(annotated_messages),
        extracted_route_schema=solution.route_schema.to_dict(
        ) if solution.route_schema else None,
        extracted_route_full=solution.route,
        extracted_route_clean=solution.processed_route,
        outcome=maze_outcome,
        lm_1_name=lm_1.adapter.model_name,
        lm_1_provider=lm_1.adapter.provider.value,  # type: ignore
        lm_2_name=lm_2.adapter.model_name if lm_2 else None,
        lm_2_provider=lm_2.adapter.provider.value if lm_2 else None,  # type: ignore
        judge_model_name=lm_grader.adapter.model_name,
        judge_model_provider=lm_grader.adapter.provider.value,
        lm_1_reasoning_effort=lm_1.config.reasoning_effort,
        lm_2_reasoning_effort=lm_2.config.reasoning_effort if lm_2 else None,
        judge_model_reasoning_effort=lm_grader.config.reasoning_effort,
        show_same_agent_value=show_same_agent_value,
        time_to_complete=total_time,
        critic_steps=critic_steps,
        pre_critic_extracted_route_schema=pre_critic_solution.route_schema.to_dict(
        ) if (pre_critic_solution is not None and pre_critic_solution.route_schema) else None,
        pre_critic_extracted_route_full=pre_critic_solution.route if (
            pre_critic_solution is not None and pre_critic_solution.route) else None,
        pre_critic_outcome=pre_critic_outcome,
        continuation_filepath=task_result.continuation_filepath,
    )

    save_result_maze(
        save_object=save_object,
        task_config=task_config,
        folder_name=save_path)


###############################################################################
# Initialize Run
###############################################################################
def apply_maze_config_overrides(
    task_path: str,
    maze_size: Optional[tuple[int, int]] = None,
    min_path_length: Optional[int] = None,
    trap_density: Optional[float] = None,
    wall_density: Optional[float] = None,
    split_1_proportion: Optional[float] = None,
    split_2_proportion: Optional[float] = None,
    random_seed: Optional[int] = None,
    agent_1_language: Optional[str] = None,
    agent_2_language: Optional[str] = None,
    pre_maze_game_overview_filepath: Optional[str] = None,
    pre_maze_game_specific_filepath: Optional[str] = None,
    pre_maze_game_filter_maze_size: Optional[tuple[int, int]] = None,
    pre_maze_game_filter_success: Optional[bool] = None,
    pre_maze_game_filter_min_outcome: Optional[float] = None,
    pre_maze_game_filter_max_outcome: Optional[float] = None,
    pre_maze_game_filter_lm_1_name: Optional[str] = None,
    pre_maze_game_filter_lm_2_name: Optional[str] = None,
    pre_maze_game_filter_lm_1_provider: Optional[_SupportedProviders] = None,
    pre_maze_game_filter_lm_2_provider: Optional[_SupportedProviders] = None,
    pre_maze_game_force_model_match: bool = True,
    pre_maze_game_filter_split_1_proportion: Optional[float] = None,
    pre_maze_game_filter_split_2_proportion: Optional[float] = None,
    pre_maze_game_continuation_type: Literal["relay_game"] = _RELAY_CONTINUATION,
    pre_maze_game_relay_point: Optional[int] = None,
    pre_maze_game_force_model_match_lm_1: Optional[bool] = None,
    pre_maze_game_force_model_match_lm_2: Optional[bool] = None,
) -> MazeTaskConfig:

    log_c(f"Running task: {task_path}", logger, ColorType.BLUE)
    task_config = MazeTaskConfig.from_config(fname=task_path)

    if maze_size:
        task_config.config.size = maze_size
        log_c(f"Updated maze size to: {maze_size}", logger, ColorType.GREEN)
    if min_path_length is not None:
        config_maze_size = task_config.config.size
        # min_path_length is given as a multiple of the maze side.
        min_path_length = int(config_maze_size[0] * min_path_length)
        task_config.config._min_path_length = min_path_length
        log_c(
            f"Updated minimum path length to: {min_path_length}", logger, ColorType.GREEN)
    if trap_density is not None:
        task_config.config.trap_density = trap_density
        log_c(
            f"Updated trap density to: {trap_density}", logger, ColorType.GREEN)
    if wall_density is not None:
        task_config.config.wall_density = wall_density
        log_c(
            f"Updated wall density to: {wall_density}", logger, ColorType.GREEN)
    if split_1_proportion is not None:
        task_config.config.split_1_proportion = split_1_proportion
        log_c(
            f"Updated split 1 proportion to: {split_1_proportion}", logger, ColorType.GREEN)
    if split_2_proportion is not None:
        task_config.config.split_2_proportion = split_2_proportion
        log_c(
            f"Updated split 2 proportion to: {split_2_proportion}", logger, ColorType.GREEN)
    if random_seed is not None:
        task_config.random_seed = random_seed
        task_config.config.random_seed = random_seed
        set_random_seed(random_seed)
        log_c(
            f"Updated random seed to: {random_seed}", logger, ColorType.GREEN)

    if agent_1_language is not None:
        task_config.agent_1_language = agent_1_language
        log_c(
            f"Updated agent 1 language to: {agent_1_language}", logger, ColorType.GREEN)
    if agent_2_language is not None:
        task_config.agent_2_language = agent_2_language
        log_c(
            f"Updated agent 2 language to: {agent_2_language}", logger, ColorType.GREEN)

    # Create the PreMazeGameConfig object
    pre_maze_game_config = PreMazeGameConfig(
        overview_filepath=pre_maze_game_overview_filepath,
        specific_filepath=pre_maze_game_specific_filepath,
        maze_size=pre_maze_game_filter_maze_size,
        success=pre_maze_game_filter_success,
        min_outcome=pre_maze_game_filter_min_outcome,
        max_outcome=pre_maze_game_filter_max_outcome,
        force_model_match=pre_maze_game_force_model_match,
        force_model_match_lm_1=pre_maze_game_force_model_match_lm_1,
        force_model_match_lm_2=pre_maze_game_force_model_match_lm_2,
        lm_1_name=pre_maze_game_filter_lm_1_name,
        lm_1_provider=pre_maze_game_filter_lm_1_provider,
        lm_2_name=pre_maze_game_filter_lm_2_name,
        lm_2_provider=pre_maze_game_filter_lm_2_provider,
        split_1_proportion=pre_maze_game_filter_split_1_proportion,
        split_2_proportion=pre_maze_game_filter_split_2_proportion,
        continuation_type=pre_maze_game_continuation_type,
        relay_point=pre_maze_game_relay_point,
    )

    task_config.pre_maze_game_config = pre_maze_game_config
    log_c(
        f'Updated the pre-maze game config: {pre_maze_game_config}',
        logger, ColorType.GREEN)

    return task_config


def main(
    lm_1_name: str = "gpt-5.4-mini",
    lm_1_provider: Optional[_SupportedProviders] = None,
    lm_1_reasoning_effort: Optional[_ReasoningEffort] = None,
    lm_2_name: Optional[str] = "gpt-5.4-mini",
    lm_2_provider: Optional[_SupportedProviders] = None,
    lm_2_reasoning_effort: Optional[_ReasoningEffort] = None,
    lm_grader_name: str = "gpt-4.1",
    lm_grader_provider: Optional[_SupportedProviders] = None,
    lm_grader_reasoning_effort: Optional[_ReasoningEffort] = None,
    task_path: str = "./data/tasks/collaborative/maze_game.yaml",
    max_turns: int = 50,
    maze_size: Optional[tuple[int, int]] = None,
    min_path_length: Optional[int] = None,
    trap_density: Optional[float] = 0.0,
    wall_density: Optional[float] = None,
    split_1_proportion: Optional[float] = None,
    split_2_proportion: Optional[float] = None,
    random_seed: Optional[int] = None,
    game_mode: Literal["collab", "one-shot"] = "collab",
    split_view: bool = False,
    critic_steps: int = 0,
    user_intermediate: bool = True,
    agent_1_language: Optional[str] = None,
    agent_2_language: Optional[str] = None,
    job_folder: str = "maze_games",
    job_id: Optional[str] = None,
    show_same_agent: bool = False,
    show_same_is_model_based: bool = True,
    show_same_override_value: bool = False,
    sleep_time: int = 0,
    pre_maze_game_overview_filepath: Optional[str] = None,
    pre_maze_game_specific_filepath: Optional[str] = None,
    pre_maze_game_filter_maze_size: Optional[tuple[int, int]] = None,
    pre_maze_game_filter_lm_1_name: Optional[str] = None,
    pre_maze_game_filter_lm_1_provider: Optional[_SupportedProviders] = None,
    pre_maze_game_filter_lm_2_name: Optional[str] = None,
    pre_maze_game_filter_lm_2_provider: Optional[_SupportedProviders] = None,
    pre_maze_game_filter_success: Optional[bool] = None,
    pre_maze_game_filter_min_outcome: Optional[float] = None,
    pre_maze_game_filter_max_outcome: Optional[float] = None,
    pre_maze_game_force_model_match: bool = True,
    pre_maze_game_filter_split_1_proportion: Optional[float] = None,
    pre_maze_game_filter_split_2_proportion: Optional[float] = None,
    pre_maze_game_continuation_type: Literal["relay_game"] = _RELAY_CONTINUATION,
    pre_maze_game_relay_point: Optional[int] = None,
    pre_maze_game_force_model_match_lm_1: Optional[bool] = None,
    pre_maze_game_force_model_match_lm_2: Optional[bool] = None,
    debug_mode: bool = False
):
    """Run a maze task with specified language models and configurations.

    Example, a collaborative game:
        python -m src.run_maze_task \
        --job_folder=scratch --job_id=demo \
        --game_mode=collab \
        --lm_1_name=gpt-4.1-mini --lm_2_name=gpt-4.1-mini \
        --maze_size=[6,6] --min_path_length=1.5 \
        --wall_density=0.3 --trap_density=0.0 \
        --random_seed=42

    Example, a relay game resuming a prior game at turn 4:
        python -m src.run_maze_task \
        --job_folder=scratch --job_id=relay \
        --game_mode=collab \
        --lm_1_name=gpt-4.1-mini --lm_2_name=gpt-4.1-mini \
        --maze_size=[6,6] \
        --pre_maze_game_overview_filepath=./data/task_results/example_relay_source/relay_overview.csv \
        --pre_maze_game_force_model_match=False \
        --pre_maze_game_relay_point=4 \
        --random_seed=42

    Args:
        lm_1_name (str): Name of the first language model.
        lm_1_provider (_SupportedProviders): Provider of the first language model.
        lm_2_name (Optional[str]): Name of the second language model (if applicable).
        lm_2_provider (_SupportedProviders): Provider of the second language model (if applicable).
        lm_grader_name (str): Name of the language model used for grading.
        lm_grader_provider (_SupportedProviders): Provider of the language model used for grading.
        lm_1_reasoning_effort (Optional[_ReasoningEffort]): Reasoning effort for
            the first model, one of 'none', 'minimal', 'low', 'medium', 'high'.
            Ignored unless the model declares ReasoningSupport in
            data/models.yaml and is served by the OpenAI or Google adapter.
        lm_2_reasoning_effort (Optional[_ReasoningEffort]): As above, for the
            second model.
        lm_grader_reasoning_effort (Optional[_ReasoningEffort]): As above, for
            the grading model.
        task_path (str): Path to the maze task configuration file.
        max_turns (int): Maximum number of turns for the task. Defaults to 50.
        maze_size (Optional[tuple[int, int]]): Size of the maze as a tuple of (rows, columns).
        min_path_length (Optional[int]): Minimum length of the path in the maze.
        trap_density (Optional[float]): Density of traps in the maze. Defaults to 0.0.
        wall_density (Optional[float]): Density of walls in the maze. Defaults to None.
        split_1_proportion (Optional[float]): Proportion of the first split in the maze.
        split_2_proportion (Optional[float]): Proportion of the second split in the maze.
        random_seed (Optional[int]): Random seed for reproducibility.
        game_mode (Literal["collab", "one-shot"]): Mode of the game, 'collab' or 'one-shot'.
        split_view (bool): Whether to display the maze in split view. Defaults to False.
        critic_steps (int): Number of steps for the critic model in one-shot mode. Defaults to 0.
        user_intermediate (bool): Whether to allow user intermediate messages. Defaults to True.
        agent_1_language (Optional[str]): Language for the first agent.
        agent_2_language (Optional[str]): Language for the second agent.
        job_folder (str): Folder name to save results. Defaults to "maze_games".
        job_id (Optional[str]): Unique identifier for the job. If None, a random ID is generated.
        show_same_agent (bool): Whether to tell the agents the identity of the other agent.
        show_same_is_model_based (bool): Whether the identity is based on the actual models used, or overridden by the user.
        show_same_override_value (bool): What to override the same_agent identity value to.
        sleep_time (int): Time in minutes to sleep before starting the task. Defaults to 0.
        pre_maze_game_overview_filepath (Optional[str]): Path to the overview file for the pre-maze game.
        pre_maze_game_specific_filepath (Optional[str]): Path to the specific file for the pre-maze game.
        pre_maze_game_filter_success (Optional[bool]): Whether to filter successful outcomes in the pre-maze game.
        pre_maze_game_filter_min_outcome (Optional[float]): Minimum outcome for filtering in the pre-maze game.
        pre_maze_game_filter_max_outcome (Optional[float]): Maximum outcome for filtering in the pre-maze game.
        pre_maze_game_force_model_match (bool): Whether to force model match in the pre-maze game. Defaults to True.
        pre_maze_game_relay_point (Optional[int]): Relay point for the pre-maze game.
        pre_maze_game_force_model_match_lm_1 (Optional[bool]): Whether to force model match for the first language model in the pre-maze game.
        pre_maze_game_force_model_match_lm_2 (Optional[bool]): Whether to force model match for the second language model in the pre-maze game.
        debug_mode (bool): Whether to enable debug mode. Defaults to False.
    """

    if sleep_time > 0:
        log_c(f"Sleeping for {sleep_time} minutes...",
              logger, ColorType.YELLOW)
        time.sleep(sleep_time * 60)

    log_c("Starting maze task...", logger, ColorType.BLUE)
    # NOTE: reasoning_effort is silently dropped by LMFactory for any model
    # whose registry entry does not declare ReasoningSupport, so it is safe to
    # pass unconditionally. Only the OpenAI and Google adapters act on it.
    lm_1 = LMFactory(
        model_name=lm_1_name,
        provider=lm_1_provider,
        config=LMConfig(reasoning_effort=lm_1_reasoning_effort),
    ).create_instance()
    if game_mode == "collab":
        assert lm_2_name is not None, "lm_2_name must be provided for collaborative mode"
        lm_2 = LMFactory(
            model_name=lm_2_name,
            provider=lm_2_provider,
            config=LMConfig(reasoning_effort=lm_2_reasoning_effort),
        ).create_instance()
    else:
        lm_2 = None
    lm_grader = LMFactory(
        model_name=lm_grader_name,
        provider=lm_grader_provider,
        config=LMConfig(reasoning_effort=lm_grader_reasoning_effort),
    ).create_instance()

    if debug_mode:
        log_c("WARNING DEBUG MODE ON!", logger, ColorType.RED)
        lm_1.debug_mode = True
        lm_grader.debug_mode = True
        if lm_2 is not None:
            lm_2.debug_mode = True

    task_config = apply_maze_config_overrides(
        task_path=task_path,
        maze_size=maze_size,
        min_path_length=min_path_length,
        trap_density=trap_density,
        wall_density=wall_density,
        split_1_proportion=split_1_proportion,
        split_2_proportion=split_2_proportion,
        random_seed=random_seed,
        agent_1_language=agent_1_language,
        agent_2_language=agent_2_language,
        pre_maze_game_overview_filepath=pre_maze_game_overview_filepath,
        pre_maze_game_specific_filepath=pre_maze_game_specific_filepath,
        pre_maze_game_filter_maze_size=pre_maze_game_filter_maze_size,
        pre_maze_game_filter_success=pre_maze_game_filter_success,
        pre_maze_game_filter_min_outcome=pre_maze_game_filter_min_outcome,
        pre_maze_game_filter_max_outcome=pre_maze_game_filter_max_outcome,
        pre_maze_game_force_model_match=pre_maze_game_force_model_match,
        pre_maze_game_filter_lm_1_name=pre_maze_game_filter_lm_1_name,
        pre_maze_game_filter_lm_1_provider=pre_maze_game_filter_lm_1_provider,
        pre_maze_game_filter_lm_2_name=pre_maze_game_filter_lm_2_name,
        pre_maze_game_filter_lm_2_provider=pre_maze_game_filter_lm_2_provider,
        pre_maze_game_filter_split_1_proportion=pre_maze_game_filter_split_1_proportion,
        pre_maze_game_filter_split_2_proportion=pre_maze_game_filter_split_2_proportion,
        pre_maze_game_continuation_type=pre_maze_game_continuation_type,
        pre_maze_game_relay_point=pre_maze_game_relay_point,
        pre_maze_game_force_model_match_lm_1=pre_maze_game_force_model_match_lm_1,
        pre_maze_game_force_model_match_lm_2=pre_maze_game_force_model_match_lm_2,
    )

    play_maze_task(
        task_config=task_config,
        lm_1=lm_1,
        lm_2=lm_2,
        lm_grader=lm_grader,
        max_turns=max_turns,
        game_mode=game_mode,
        split_view=split_view,
        critic_steps=critic_steps,
        user_intermediate=user_intermediate,
        job_folder=job_folder,
        job_id=job_id,
        show_same_agent=show_same_agent,
        show_same_is_model_based=show_same_is_model_based,
        show_same_override_value=show_same_override_value,
    )
    print_c("Finished task!", ColorType.GREEN)


if __name__ == "__main__":
    fire.Fire(main)
