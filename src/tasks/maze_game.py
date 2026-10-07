"""A class to handle the maze game task.
"""
import yaml
from dataclasses import dataclass, asdict
from typing import Optional, Literal, overload
import random
from collections import deque
from dataclasses import dataclass, field
import functools
import logging
import math
import enum

import fire

from src.utils.general_utilities import set_random_seed
from src.models.adapters import _SupportedProviders


logger = logging.getLogger(__name__)

_CoordinateType = tuple[int, int]
_MazeTypeStr = list[list[str]]
_MazeTypePath = list[_CoordinateType]


def _manhattan_distance(pos1: _CoordinateType, pos2: _CoordinateType) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


@dataclass
class MazeSymbols:
    """Class to hold maze symbols."""
    current: str = '@'
    goal: str = '*'
    trap: str = 'X'
    wall: str = '#'
    hidden: str = '?'
    path: str = '.'
    solution: str = '>'

    def to_dict(self) -> dict:
        """Convert the MazeSymbols to a dictionary."""
        return {
            'current': self.current,
            'goal': self.goal,
            'trap': self.trap,
            'wall': self.wall,
            'hidden': self.hidden,
            'path': self.path,
            'solution': self.solution
        }


@dataclass
class PreMazeGameConfig:
    """Configuration for a relay game: which prior game to resume, and where.

    Attributes:
        overview_filepath: Overview CSV listing candidate source games.
        specific_filepath: A single source game's result.yaml, instead.
        success: Keep only source games that were (not) solved.
        min_outcome: Minimum outcome value.
        max_outcome: Maximum outcome value.
        force_model_match: Whether to force model matching.
        force_model_match_lm_1: Whether to force model matching for LM 1.
        force_model_match_lm_2: Whether to force model matching for LM 2.
        relay_point: Relay point for the game.

    Functions:
        from_dict: Class method to create an instance from a dictionary.
        to_dict: Convert the dataclass to a dictionary.
        has_file: Cached property to check if either overview or specific file path is set.
    """
    overview_filepath: Optional[str] = None
    specific_filepath: Optional[str] = None
    maze_size: Optional[tuple[int, int]] = None
    success: Optional[bool] = None
    min_outcome: Optional[float] = None
    max_outcome: Optional[float] = None
    force_model_match: bool = True
    force_model_match_lm_1: Optional[bool] = None
    force_model_match_lm_2: Optional[bool] = None
    lm_1_name: Optional[str] = None
    lm_2_name: Optional[str] = None
    lm_1_provider: Optional[_SupportedProviders] = None
    lm_2_provider: Optional[_SupportedProviders] = None
    split_1_proportion: Optional[float] = 0.5
    split_2_proportion: Optional[float] = 0.5
    # Only "relay_game" is supported.
    continuation_type: Literal["relay_game"] = "relay_game"
    relay_point: Optional[int] = None

    @functools.cached_property
    def has_file(self) -> bool:
        """Check if either overview or specific file path is set."""
        return bool(self.overview_filepath or self.specific_filepath)

    @classmethod
    def from_dict(cls, data: dict) -> 'PreMazeGameConfig':
        """Create a PreMazeGameConfig instance from a dictionary."""
        return cls(
            overview_filepath=data.get('overview_filepath'),
            specific_filepath=data.get('specific_filepath'),
            maze_size=tuple(data.get('maze_size', (6, 6))
                            ) if data.get('maze_size') else None,
            success=data.get('success'),
            min_outcome=data.get('min_outcome'),
            max_outcome=data.get('max_outcome'),
            force_model_match=data.get('force_model_match', True),
            force_model_match_lm_1=data.get('force_model_match_lm_1'),
            force_model_match_lm_2=data.get('force_model_match_lm_2'),
            lm_1_name=data.get('lm_1_name'),
            lm_2_name=data.get('lm_2_name'),
            lm_1_provider=data.get('lm_1_provider'),
            lm_2_provider=data.get('lm_2_provider'),
            split_1_proportion=data.get('split_1_proportion', 0.5),
            split_2_proportion=data.get('split_2_proportion', 0.5),
            continuation_type=data.get('continuation_type', 'relay_game'),
            relay_point=data.get('relay_point'),
        )

    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary."""
        return {
            "overview_filepath": self.overview_filepath,
            "specific_filepath": self.specific_filepath,
            "maze_size": self.maze_size,
            "success": self.success,
            "min_outcome": self.min_outcome,
            "max_outcome": self.max_outcome,
            "force_model_match": self.force_model_match,
            "force_model_match_lm_1": self.force_model_match_lm_1,
            "force_model_match_lm_2": self.force_model_match_lm_2,
            "lm_1_name": self.lm_1_name,
            "lm_2_name": self.lm_2_name,
            "lm_1_provider": self.lm_1_provider,
            "lm_2_provider": self.lm_2_provider,
            "split_1_proportion": self.split_1_proportion,
            "split_2_proportion": self.split_2_proportion,
            "continuation_type": self.continuation_type,
            "relay_point": self.relay_point,
            "has_file": self.has_file
        }


@dataclass
class MazeConfig:
    size: tuple[int, int]
    wall_density: float
    trap_density: float
    goal_position: Optional[_CoordinateType] = None
    start_position: Optional[_CoordinateType] = None
    base_symbols: MazeSymbols = field(default_factory=MazeSymbols)
    _min_path_length: Optional[int] = None
    _max_path_length: Optional[int] = None
    split_1_proportion: float = 0.5
    split_2_proportion: float = 0.5
    random_seed: Optional[int] = None

    # Plain properties, not cached_property: apply_maze_config_overrides assigns
    # _min_path_length after the config is constructed, and a cached value would
    # make whether that override takes effect depend on whether anything had
    # read the property first.
    @property
    def min_path_length(self) -> int:
        if self._min_path_length is not None:
            return self._min_path_length
        return max(self.size[0], self.size[1]) - 1

    @property
    def max_path_length(self) -> int:
        if self._max_path_length is not None:
            return self._max_path_length
        return (max(self.size) - 1) * (min(self.size) - 1)

    @classmethod
    def from_config(
        cls,
        config_dict: dict
    ) -> 'MazeConfig':
        """Create a MazeConfig instance from a dictionary."""
        return cls(
            size=tuple(config_dict.get('size', (6, 6))),
            wall_density=config_dict.get('wall_density', 0.2),
            trap_density=config_dict.get('trap_density', 0.0),
            goal_position=config_dict.get('goal_position'),
            start_position=config_dict.get('start_position', (0, 0)),
            base_symbols=MazeSymbols(
                current=config_dict.get('current_symbol', '@'),
                goal=config_dict.get('goal_symbol', '*'),
                trap=config_dict.get('trap_symbol', 'T'),
                wall=config_dict.get('wall_symbol', '#'),
                hidden=config_dict.get('hidden_symbol', '?'),
                path=config_dict.get('path_symbol', '.')
            ),
            _min_path_length=config_dict.get('min_path_length'),
            _max_path_length=config_dict.get('max_path_length'),
            split_1_proportion=config_dict.get('split_1_proportion', 0.5),
            split_2_proportion=config_dict.get('split_2_proportion', 0.5),
            random_seed=config_dict.get('random_seed', None)
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MazeConfig':
        """Create a MazeConfig instance from a dictionary."""

        return cls(
            size=tuple(config_dict.get('size', (6, 6))),
            wall_density=config_dict.get('wall_density', 0.2),
            trap_density=config_dict.get('trap_density', 0.0),
            goal_position=config_dict.get('goal_position'),
            start_position=config_dict.get('start_position'),
            base_symbols=MazeSymbols(**config_dict.get('base_symbols', {})),
            _min_path_length=config_dict.get('min_path_length'),
            _max_path_length=config_dict.get('max_path_length'),
            split_1_proportion=config_dict.get(
                'split_1_proportion', 0.5),
            split_2_proportion=config_dict.get(
                'split_2_proportion', 0.5),
            random_seed=config_dict.get('random_seed', None)
        )

    def to_dict(self) -> dict:
        """Convert the MazeConfig to a dictionary."""
        return {
            'size': self.size,
            'wall_density': self.wall_density,
            'trap_density': self.trap_density,
            'goal_position': self.goal_position,
            'start_position': self.start_position,
            'base_symbols': self.base_symbols.to_dict(),
            'min_path_length': self.min_path_length,
            'max_path_length': self.max_path_length,
            'split_1_proportion': self.split_1_proportion,
            'split_2_proportion': self.split_2_proportion,
            'random_seed': self.random_seed
        }

    def validate(self):
        """Validate configuration parameters."""
        rows, cols = self.size
        if rows <= 0 or cols <= 0:
            raise ValueError("Maze size must be positive")

        if self.start_position and not (0 <= self.start_position[0] < rows and
                                        0 <= self.start_position[1] < cols):
            raise ValueError("Current position out of bounds")

        if self.goal_position and not (0 <= self.goal_position[0] < rows and
                                       0 <= self.goal_position[1] < cols):
            raise ValueError("Goal position out of bounds")

        if self.start_position == self.goal_position:
            raise ValueError("Start and goal positions cannot be the same")

        if not (0 <= self.wall_density <= 1):
            raise ValueError("Wall density must be between 0 and 1")

        if not (0 <= self.trap_density <= 1):
            raise ValueError("Trap density must be between 0 and 1")


@dataclass
class MazeTaskConfig:
    """Class to represent the configuration for a maze task."""
    fname: str
    config: MazeConfig
    prompts: dict[str, str] = field(default_factory=dict)
    noise_level: float = 0.0
    random_seed: Optional[int] = None
    agent_1_language: Optional[str] = None
    agent_2_language: Optional[str] = None
    enable_agent_symbols: bool = False
    agent_1_symbols: MazeSymbols = field(default_factory=MazeSymbols)
    agent_2_symbols: MazeSymbols = field(default_factory=MazeSymbols)
    pre_maze_game_config: Optional[PreMazeGameConfig] = None

    @classmethod
    def from_config(cls, fname: str) -> 'MazeTaskConfig':
        """Create a MazeTaskConfig instance from a dictionary."""

        with open(fname, 'r') as f:
            config_dict = yaml.safe_load(f)

        agent_symbols = config_dict.get("agent_symbols", {})
        enable_agent_symbols = agent_symbols.get("enable", False)
        agent_1_symbols = MazeSymbols(
            **agent_symbols.get('agent_1_symbols', {}))
        agent_2_symbols = MazeSymbols(
            **agent_symbols.get('agent_2_symbols', {}))

        return cls(
            fname=fname,
            config=MazeConfig.from_config(
                config_dict.get('maze_config', {})
            ),
            prompts=config_dict.get('prompts', {}),
            noise_level=config_dict.get('noise_level', 0.0),
            random_seed=config_dict.get('random_seed', None),
            agent_1_language=config_dict.get('agent_1_language', None),
            agent_2_language=config_dict.get('agent_2_language', None),
            enable_agent_symbols=enable_agent_symbols,
            agent_1_symbols=agent_1_symbols,
            agent_2_symbols=agent_2_symbols,
            pre_maze_game_config=PreMazeGameConfig.from_dict(
                config_dict.get('pre_maze_game_config', {})
            ) if config_dict.get('pre_maze_game_config') else None
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MazeTaskConfig':
        """Create a MazeTaskConfig instance from a dictionary."""

        agent_symbols = config_dict.get("agent_symbols", {})
        enable_agent_symbols = agent_symbols.get("enable", False)
        agent_1_symbols = MazeSymbols(
            **agent_symbols.get('agent_1_symbols', {}))
        agent_2_symbols = MazeSymbols(
            **agent_symbols.get('agent_2_symbols', {}))

        return cls(
            fname=config_dict.get('fname', 'maze_task'),
            config=MazeConfig.from_dict(config_dict.get('maze_config', {})),
            prompts=config_dict.get('prompts', {}),
            noise_level=config_dict.get('noise_level', 0.0),
            random_seed=config_dict.get('random_seed', None),
            agent_1_language=config_dict.get('agent_1_language', None),
            agent_2_language=config_dict.get('agent_2_language', None),
            enable_agent_symbols=enable_agent_symbols,
            agent_1_symbols=agent_1_symbols,
            agent_2_symbols=agent_2_symbols,
            pre_maze_game_config=PreMazeGameConfig.from_dict(
                config_dict.get('pre_maze_game_config', {})
            ) if config_dict.get('pre_maze_game_config') else None
        )

    def to_dict(self) -> dict:
        """Convert the MazeTaskConfig to a dictionary."""
        return {
            'fname': self.fname,
            'maze_config': self.config.to_dict(),
            'prompts': self.prompts,
            'noise_level': self.noise_level,
            'random_seed': self.random_seed,
            'agent_1_language': self.agent_1_language,
            'agent_2_language': self.agent_2_language,
            'agent_symbols': {
                'enable': self.enable_agent_symbols,
                'agent_1_symbols': self.agent_1_symbols.to_dict(),
                'agent_2_symbols': self.agent_2_symbols.to_dict()
            },
            'pre_maze_game_config': self.pre_maze_game_config.to_dict() if self.pre_maze_game_config else None
        }


class SolutionTypeOrientation(enum.Enum):
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class SolutionTypeCoordinates(enum.Enum):
    ROW_COL = "row_col"
    COL_ROW = "col_row"


class SolutionTypeSymbols(enum.Enum):
    number_number = "number_number"
    directions = "directions"
    number_letter = "number_letter"
    letter_number = "letter_number"
    letter_letter = "letter_letter"


@dataclass
class SolutionType:
    maze_orientation: SolutionTypeOrientation
    coordinates_orientation: SolutionTypeCoordinates
    coordinates_symbols: SolutionTypeSymbols
    maze_origin: Literal["0", "1"] = "0"

    def __str__(self) -> str:
        return f"{self.maze_orientation.value}|{self.coordinates_orientation.value}|{self.coordinates_symbols.value}|{self.maze_origin}"

    @classmethod
    def infer_type(cls, coordinates: _MazeTypePath) -> 'SolutionType':
        symbol_type = cls._infer_symbol_type(coordinates)

        if symbol_type == SolutionTypeSymbols.letter_letter:
            maze_origin = "1"
            coordinate_type = SolutionTypeCoordinates.COL_ROW
        elif symbol_type == SolutionTypeSymbols.letter_number:
            maze_origin = "1"
            coordinate_type = SolutionTypeCoordinates.COL_ROW
        elif symbol_type == SolutionTypeSymbols.number_letter:
            maze_origin = "1"
            coordinate_type = SolutionTypeCoordinates.ROW_COL
        elif symbol_type == SolutionTypeSymbols.number_number:
            maze_origin = "0"
            coordinate_type = SolutionTypeCoordinates.ROW_COL
        elif symbol_type == SolutionTypeSymbols.directions:
            maze_origin = "0"
            coordinate_type = SolutionTypeCoordinates.ROW_COL
        else:
            raise ValueError(f"Invalid symbol type: {symbol_type}")

        return cls(
            maze_orientation=SolutionTypeOrientation.TOP_LEFT,
            coordinates_orientation=coordinate_type,
            coordinates_symbols=symbol_type,
            maze_origin=maze_origin
        )

    @staticmethod
    def _infer_symbol_type(coordinates: _MazeTypePath) -> SolutionTypeSymbols:
        if not isinstance(coordinates, list) or not coordinates:
            return SolutionTypeSymbols.number_number

        coordinate = coordinates[0]
        if (isinstance(coordinate, tuple) or isinstance(coordinate, list)) and coordinate:
            coordinate = coordinate[0]

        if isinstance(coordinate, str) and len(coordinate) == 2:
            if coordinate[0].isalpha() and coordinate[1].isdigit():  # type: ignore
                return SolutionTypeSymbols.letter_number
            elif coordinate[0].isdigit() and coordinate[1].isalpha():  # type: ignore
                return SolutionTypeSymbols.number_letter
            elif coordinate[0].isalpha() and coordinate[1].isalpha():  # type: ignore
                return SolutionTypeSymbols.letter_letter
        elif isinstance(coordinate, str) and coordinate.lower() in ["up", "down", "left", "right"]:
            return SolutionTypeSymbols.directions

        return SolutionTypeSymbols.number_number

    @classmethod
    def from_string(cls, s: str) -> 'SolutionType':
        """Create a SolutionType instance from a string."""
        parts = s.split('|')
        if len(parts) != 4:
            raise ValueError(f"Invalid solution type string: {s}")

        origin = parts[3]

        return cls(
            maze_orientation=SolutionTypeOrientation(parts[0]),
            coordinates_orientation=SolutionTypeCoordinates(parts[1]),
            coordinates_symbols=SolutionTypeSymbols(parts[2]),
            maze_origin=origin if origin in ["0", "1"] else "0"  # type: ignore
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'SolutionType':
        """Create a SolutionType instance from a dictionary."""
        return cls(
            maze_orientation=SolutionTypeOrientation(data['maze_orientation']),
            coordinates_orientation=SolutionTypeCoordinates(
                data['coordinates_orientation']),
            coordinates_symbols=SolutionTypeSymbols(
                data['coordinates_symbols']),
            maze_origin=str(data['maze_origin']
                            ) if 'maze_origin' in data else "0"  # type: ignore
        )

    def to_dict(self) -> dict:
        return {
            "maze_orientation": self.maze_orientation.value,
            "coordinates_orientation": self.coordinates_orientation.value,
            "coordinates_symbols": self.coordinates_symbols.value,
            "maze_origin": self.maze_origin
        }

    @staticmethod
    def get_all_types():
        solution_types = []
        for maze_orientation in SolutionTypeOrientation:
            for coordinates_orientation in SolutionTypeCoordinates:
                for coordinates_symbols in SolutionTypeSymbols:
                    for maze_origin in ["0", "1"]:
                        solution_types.append(
                            SolutionType(
                                maze_orientation=maze_orientation,
                                coordinates_orientation=coordinates_orientation,
                                coordinates_symbols=coordinates_symbols,
                                maze_origin=maze_origin  # type: ignore
                            )
                        )

        return solution_types


@dataclass
class MazeOutcome:
    """Class to represent the outcome of a maze game."""
    outcome_optimal_num_steps: int
    outcome_success: bool
    outcome_best_position: Optional[_CoordinateType] = None
    outcome_num_steps: int = 0
    outcome_path: _MazeTypePath = field(default_factory=list)
    outcome_invalid_reason: Optional[str] = None
    outcome_steps_from_goal: Optional[int] = None
    outcome_solution_type: Optional[str] = None
    outcome_parsing_error: Optional[str] = None

    @functools.cached_property
    def weighted_outcome(self) -> float:
        """Calculate a weighted outcome based on success and number of steps."""

        if self.outcome_success:
            return 1.

        optimal_steps = self.outcome_optimal_num_steps
        steps = self.outcome_steps_from_goal
        if steps is not None and optimal_steps > 0:
            per_outcome = (optimal_steps - steps) / optimal_steps
        else:
            per_outcome = 0

        return per_outcome

    def to_dict(self) -> dict:
        """Convert the MazeOutcome to a dictionary."""
        return {
            'outcome_optimal_num_steps': self.outcome_optimal_num_steps,
            'outcome_success': self.outcome_success,
            'outcome_best_position': self.outcome_best_position,
            'outcome_num_steps': self.outcome_num_steps,
            'outcome_path': self.outcome_path,
            'outcome_invalid_reason': self.outcome_invalid_reason,
            'outcome_steps_from_goal': self.outcome_steps_from_goal,
            'outcome_solution_type': self.outcome_solution_type,
            'outcome_parsing_error': self.outcome_parsing_error,
            'weighted_outcome': self.weighted_outcome
        }

    @staticmethod
    def from_dict(data: dict) -> 'MazeOutcome':
        """Create a MazeOutcome instance from a dictionary."""
        return MazeOutcome(
            outcome_optimal_num_steps=data.get('outcome_optimal_num_steps', 0),
            outcome_success=data.get('outcome_success', False),
            outcome_best_position=data.get('outcome_best_position'),
            outcome_num_steps=data.get('outcome_num_steps', 0),
            outcome_path=data.get('outcome_path', []),
            outcome_invalid_reason=data.get('outcome_invalid_reason'),
            outcome_steps_from_goal=data.get('outcome_steps_from_goal'),
            outcome_solution_type=data.get('outcome_solution_type'),
            outcome_parsing_error=data.get('outcome_parsing_error')
        )


class MazeGame:
    def __init__(
        self,
        config: MazeConfig,
        maze: Optional[_MazeTypeStr] = None,
        maze_split_1: Optional[_MazeTypeStr] = None,
        maze_split_2: Optional[_MazeTypeStr] = None
    ):
        self.config = config
        set_random_seed(self.config.random_seed)

        self.rows, self.cols = config.size
        self.start_position: _CoordinateType = config.start_position or (0, 0)
        self.goal_position: _CoordinateType = config.goal_position or self._generate_goal_position()
        self.config.start_position = self.start_position
        self.config.goal_position = self.goal_position
        self.symbols = config.base_symbols
        self.sampled_path = None

        if not (isinstance(self.start_position, tuple) and len(self.start_position) == 2):
            self.start_position = tuple(self.start_position[:2])
        if not (isinstance(self.goal_position, tuple) and len(self.goal_position) == 2):
            self.goal_position = tuple(self.goal_position[:2])

        self.config.validate()

        if maze is None:
            self.maze: _MazeTypeStr = self.generate_maze()
        else:
            self.maze: _MazeTypeStr = maze
            self.start_position = self._find_symbol_positions(
                self.symbols.current)[0]
            self.goal_position = self._find_symbol_positions(
                self.symbols.goal)[0]

        if maze_split_1 is None or maze_split_2 is None:
            self.maze_split_1, self.maze_split_2 = self._split_maze()
        else:
            self.maze_split_1 = maze_split_1
            self.maze_split_2 = maze_split_2
            self._assert_valid_splits(self.maze_split_1, self.maze_split_2)

    @staticmethod
    def get_maze_game_ascii_art() -> str:
        art = """
    ┌───┬───┬───┬───┐
    │ @ │   │ M │ a │  @ = player
    ├───┘ ┌─┴───┤ z │  * = goal
    │ ┌───┘ ┌───┤ e │
    │ │ G a m e │ * │
    └─┴─────────┴───┘
"""
        return art

    def _find_symbol_positions(self, symbol: str) -> list[_CoordinateType]:
        """Find all positions of a given symbol in the maze."""
        positions = set()
        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                if cell == symbol:
                    positions.add((i, j))
        return list(positions)

    def generate_maze(self) -> _MazeTypeStr:
        """Generate a maze following the simple approach."""
        # Step 1: Start with empty maze
        self.maze = [['' for _ in range(self.cols)] for _ in range(self.rows)]

        # Step 2: Fill in start and goal positions
        self.maze[self.start_position[0]][
            self.start_position[1]] = self.symbols.current
        self.maze[self.goal_position[0]][
            self.goal_position[1]] = self.symbols.goal

        # Step 3: Sample valid paths
        valid_paths = self._sample_paths()
        if not valid_paths:
            raise RuntimeError(
                "Could not find a valid path meeting requirements")

        # Step 4: Pick one path and mark it
        chosen_path = random.choice(valid_paths)
        self.sampled_path = chosen_path

        # Ensure chosen_path is a list of (row, col) tuples
        if not all(isinstance(cell, tuple) and len(cell) == 2 for cell in chosen_path):
            raise RuntimeError(
                "Chosen path is not a list of (row, col) tuples")
        path_cells = set(chosen_path)

        for row, col in path_cells:  # type: ignore
            if self.maze[row][col] == '':  # Don't overwrite start/goal
                self.maze[row][col] = self.symbols.path

        # Step 5: Place walls and traps on non-path coordinates
        non_path_cells = []
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) not in path_cells:
                    non_path_cells.append((row, col))

        random.shuffle(non_path_cells)

        # Calculate how many walls and traps to place
        total_cells = self.rows * self.cols
        num_walls = int(total_cells * self.config.wall_density)
        num_traps = int(total_cells * self.config.trap_density)

        # Walls and traps may only occupy cells off the sampled solution path,
        # so the achievable density is capped by how many such cells exist.
        # Warn rather than truncate silently (e.g. wall_density 0.75 on a 6x6
        # cannot be met).
        capacity = len(non_path_cells)
        if num_walls + num_traps > capacity:
            logger.warning(
                "Cannot place %d walls and %d traps: only %d of %d cells lie "
                "off the solution path, so the combined density is capped at "
                "%.3f (requested %.3f).",
                num_walls, num_traps, capacity, total_cells,
                capacity / total_cells,
                self.config.wall_density + self.config.trap_density)

        # Place walls
        for i in range(min(num_walls, len(non_path_cells))):
            row, col = non_path_cells[i]
            self.maze[row][col] = self.symbols.wall

        # Place traps
        trap_start = num_walls
        for i in range(trap_start, min(trap_start + num_traps, len(non_path_cells))):
            row, col = non_path_cells[i]
            self.maze[row][col] = self.symbols.trap

        # Step 6: Fill remaining cells with path symbol
        for row in range(self.rows):
            for col in range(self.cols):
                if self.maze[row][col] == '':
                    self.maze[row][col] = self.symbols.path

        # Walls and traps are only placed off the sampled path, so the maze is
        # solvable by construction. Assert it rather than trust it, so any
        # future change to placement fails loudly here instead of producing
        # silently unsolvable games.
        if not self.is_solvable():
            raise RuntimeError(
                "Generated maze is unsolvable despite walls and traps being "
                "placed off the solution path")

        return self.maze

    def _generate_goal_position(self) -> _CoordinateType:
        """
        Pick the goal uniformly among cells at least min_path_length away from
        the start (Manhattan distance), capped at the grid's maximum distance.

        Returns:
            (row, col) tuple for goal position
        """
        rows, cols = self.config.size

        # Cells far enough from the start
        valid_positions = []
        # The goal can be at most rows + cols - 2 steps from the start.
        min_goal_distance = min(rows + cols - 2, self.config.min_path_length)
        for row in range(rows):
            for col in range(cols):
                distance = _manhattan_distance(self.start_position, (row, col))
                if distance >= min_goal_distance:
                    valid_positions.append((row, col))

        if not valid_positions:
            raise ValueError(
                f"No valid goal position with min_distance={self.config.min_path_length} "
                f"in maze of size {self.config.size}"
            )

        # Return a random valid position
        return random.choice(valid_positions)

    def _sample_paths(self, num_samples: int = 500) -> _MazeTypePath:
        """Sample multiple paths from start to goal.

        Draws from the stream seeded once from config.random_seed in __init__,
        so different seeds yield different candidate paths.
        """
        valid_paths = []
        min_length = self.config.min_path_length or 0
        max_length = self.config.max_path_length or (self.rows * self.cols)

        for _ in range(num_samples):
            path = self._generate_random_path(max_length)
            if path and min_length <= len(path) <= max_length:
                valid_paths.append(path)

                # Early exit if we have enough paths
                if len(valid_paths) >= 10:
                    break

        return valid_paths

    def _generate_random_path(self, max_length: int) -> _MazeTypePath:
        """Generate a random path from start to goal without adjacent parallel routes."""
        path = [self.start_position]
        current = self.start_position

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while current != self.goal_position and len(path) < max_length:
            valid_moves = []

            for dr, dc in directions:
                new_row, new_col = current[0] + dr, current[1] + dc
                new_pos = (new_row, new_col)

                # Basic bounds check and not already in path
                if (0 <= new_row < self.rows and
                    0 <= new_col < self.cols and
                        new_pos not in path):

                    # Check if this would create adjacent parallel segments
                    # A move is invalid if the new position touches any previous path segment
                    # (except the current position we're moving from)
                    valid = True
                    for i in range(len(path) - 1):  # Don't check against current position
                        if _manhattan_distance(new_pos, path[i]) == 1:
                            valid = False
                            break

                    if valid:
                        valid_moves.append(new_pos)

            if not valid_moves:
                return []  # Dead end

            # Random step among the valid moves
            next_pos = random.choice(valid_moves)
            path.append(next_pos)
            current = next_pos

        return path if current == self.goal_position else []

    def is_solvable(self) -> bool:
        """Check if the maze can be solved."""
        if not self.maze:
            return False

        visited = set()
        queue = deque([self.start_position])
        visited.add(self.start_position)

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while queue:
            row, col = queue.popleft()

            if (row, col) == self.goal_position:
                return True

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                new_pos = (new_row, new_col)

                if (0 <= new_row < self.rows and
                    0 <= new_col < self.cols and
                    new_pos not in visited and
                        self.maze[new_row][new_col] not in [self.symbols.wall, self.symbols.trap]):
                    visited.add(new_pos)
                    queue.append(new_pos)

        return False

    def _convert_letter_to_number(self, letter: str | int) -> int:
        """Convert a letter to a number (A=1, B=2, ..., Z=26)."""
        if isinstance(letter, int):
            return letter
        if letter.isalpha():
            return ord(letter.upper()) - ord('A') + 1
        raise ValueError(f"Invalid letter: {letter}")

    def _convert_solution_symbols(
        self,
        solution: _MazeTypePath,
        symbols: SolutionTypeSymbols
    ) -> _MazeTypePath:
        """Convert solution symbols based on the specified symbols type."""
        converted_solution = None
        if symbols == SolutionTypeSymbols.letter_letter:
            converted_solution = [(
                self._convert_letter_to_number(c[0][0]),
                self._convert_letter_to_number(c[0][1])
            )
                if len(c) == 1 else (
                self._convert_letter_to_number(c[0]),
                self._convert_letter_to_number(c[1])
            ) for c in solution]
        elif symbols == SolutionTypeSymbols.number_letter:
            converted_solution = [(
                c[0][0], self._convert_letter_to_number(c[0][1])
            )
                if len(c) == 1 else (
                    c[0], self._convert_letter_to_number(c[1])
            ) for c in solution]
        elif symbols == SolutionTypeSymbols.letter_number:
            converted_solution = [(
                self._convert_letter_to_number(c[0][0]), c[0][1]
            )
                if len(c) == 1 else (
                    self._convert_letter_to_number(c[0]), c[1]
            ) for c in solution]
        elif symbols == SolutionTypeSymbols.number_number:
            converted_solution = solution
        elif symbols == SolutionTypeSymbols.directions:
            direction_map = {
                'up': [-1, 0],
                'down': [1, 0],
                'left': [0, -1],
                'right': [0, 1]
            }
            state = [0, 0]
            converted_solution = [state.copy()]
            for move in solution:

                move_clean = move
                if (isinstance(move, list) or isinstance(move, tuple)) \
                        and len(move) > 0:
                    move_clean = move[0]
                if isinstance(move_clean, str):
                    move_clean = move_clean.lower()
                else:
                    continue

                if move_clean not in direction_map:
                    continue

                state[0] += direction_map[move_clean][0]
                state[1] += direction_map[move_clean][1]

                converted_solution.append(state.copy())
        else:
            raise ValueError(f"Invalid solution type symbols: {symbols}")

        converted_solution = [
            (int(c[0]), int(c[1])) for c in converted_solution]

        return converted_solution

    def _reorientate_solution(self, solution: _MazeTypePath, orientation: SolutionTypeOrientation) -> _MazeTypePath:
        """Reorientate the solution path based on the specified orientation."""
        if orientation == SolutionTypeOrientation.TOP_LEFT:
            return solution
        elif orientation == SolutionTypeOrientation.BOTTOM_LEFT:
            return [((c[0] + self.rows - 1) % self.rows, c[1]) for c in solution]
        elif orientation == SolutionTypeOrientation.TOP_RIGHT:
            return [(c[1], c[0]) for c in solution]
        elif orientation == SolutionTypeOrientation.BOTTOM_RIGHT:
            return [((c[1] + self.cols - 1) % self.cols, c[0]) for c in solution]
        else:
            raise ValueError(f"Invalid orientation: {orientation}")

    def _convert_x_y_to_row_col(self, solution: _MazeTypePath) -> _MazeTypePath:
        """Convert (x, y) coordinates to (row, col) coordinates."""
        return [(c[1], c[0]) for c in solution]

    def _offset_solution_by_one(self, solution: _MazeTypePath, offset: str) -> _MazeTypePath:
        if offset not in ['0', '1']:
            return solution

        if offset == '0':
            return solution
        else:
            # Offset by 1 in both dimensions
            return [(max(0, c[0] - 1), max(0, c[1] - 1)) for c in solution]

    def _solution_path_contains_directions(self, solution: _MazeTypePath) -> bool:
        """Check if the solution path contains direction moves."""
        direction_moves = {'up', 'down', 'left', 'right'}
        has_directions = False
        for move in solution:
            if isinstance(move, str) and move.lower() in direction_moves:
                has_directions = True
                break
            if isinstance(move, (list, tuple)) and len(move) > 0:
                first_elem = move[0]
                if isinstance(first_elem, str) and first_elem.lower() in direction_moves:
                    has_directions = True
                    break
        return has_directions

    def normalize_solution(
        self,
        solution: _MazeTypePath,
        solution_type: Optional[SolutionType] = None
    ) -> tuple[_MazeTypePath, SolutionType] | tuple[None, None]:

        if not solution:
            return None, None

        if not solution_type:
            solution_type = SolutionType.infer_type(solution)
        else:
            # Sanity checks:
            # (1) if the solution contains directions, we cannot
            # have a number_number solution.
            # (2) if the solution does **not** contain directions, we cannot
            # have a directions solution.
            solution_contains_directions = self._solution_path_contains_directions(
                solution)
            if solution_contains_directions and solution_type.coordinates_symbols == SolutionTypeSymbols.number_number:
                solution_type.coordinates_symbols = SolutionTypeSymbols.directions
            elif not solution_contains_directions and solution_type.coordinates_symbols == SolutionTypeSymbols.directions:
                solution_type.coordinates_symbols = SolutionTypeSymbols.number_number
            else:
                pass

        # Make sure the solution is a list of int tuples
        solution = self._convert_solution_symbols(
            solution, solution_type.coordinates_symbols)

        # Convert coordinates if needed
        if solution_type.coordinates_orientation == SolutionTypeCoordinates.COL_ROW:
            solution = self._convert_x_y_to_row_col(solution)

        # Reorientate the solution if needed
        solution = self._reorientate_solution(
            solution, solution_type.maze_orientation)

        # Offset solution if needed
        solution = self._offset_solution_by_one(
            solution, solution_type.maze_origin)

        # Remove sequence duplicates while preserving order
        _solution_deduped = []
        for i, pos in enumerate(solution):
            if i == 0 or pos != solution[i - 1]:
                _solution_deduped.append(pos)
        solution = _solution_deduped

        return solution, solution_type

    def _find_most_favored_outcome(
        self,
        solution: _MazeTypePath
    ) -> MazeOutcome:

        all_solution_types = SolutionType.get_all_types()
        best_outcome = None
        e = "No valid solution found"
        for solution_type in all_solution_types:
            try:
                normalized_solution, _ = self.normalize_solution(
                    solution, solution_type=solution_type
                )
                outcome = self._verify_solution(normalized_solution)

                # If the outcome is a success, solution can not be improved
                if outcome.outcome_success:
                    best_outcome = outcome
                    best_outcome.outcome_solution_type = str(solution_type)
                    break

                # check if the current solution is better than the old one
                if best_outcome is None or outcome.weighted_outcome > best_outcome.weighted_outcome:
                    best_outcome = outcome
                    best_outcome.outcome_solution_type = str(solution_type)

            except Exception:
                continue

        if best_outcome is None:
            best_outcome = self._verify_solution(None)
            best_outcome.outcome_parsing_error = str(e)

        return best_outcome

    def verify_solution(
            self,
            solution: _MazeTypePath,
            solution_type: Optional[SolutionType] = None) -> MazeOutcome:
        """
        Verify if the proposed solution is valid.

        Args:
            solution: A list of (int, int) tuples representing the path.
        Returns:
            MazeOutcome: An object containing the outcome of the verification.
        """

        # First try to use the schema found by the grader
        outcome = None
        normalized_solution = None
        normalized_solution_type = None

        try:
            normalized_solution, normalized_solution_type = self.normalize_solution(
                solution, solution_type=solution_type
            )
        except Exception as e:
            pass

        if normalized_solution is None:
            outcome = self._find_most_favored_outcome(solution)
        else:
            try:
                outcome = self._verify_solution(normalized_solution)
                outcome.outcome_solution_type = str(normalized_solution_type)
            except:
                outcome = self._verify_solution(None)
                outcome.outcome_parsing_error = "Error verifying solution"
                outcome.outcome_invalid_reason = "Invalid solution format"
                outcome.outcome_solution_type = str(
                    normalized_solution_type) if normalized_solution_type else None

            if not outcome.outcome_success:
                outcome = self._find_most_favored_outcome(solution)

        return outcome

    def _verify_solution(self, solution: _MazeTypePath | None) -> MazeOutcome:
        """Check if a proposed solution is correct."""
        optimal_solution = self.find_optimal_solution()
        optimal_num_steps = len(optimal_solution) - \
            1 if optimal_solution else -1
        default_outcome = MazeOutcome(
            outcome_optimal_num_steps=optimal_num_steps,
            outcome_num_steps=0,
            outcome_success=False,
            outcome_best_position=self.start_position,
            outcome_steps_from_goal=optimal_num_steps
        )
        if not solution or not isinstance(solution, list):
            return default_outcome

        try:
            _solution = [tuple(c) for c in solution]
        except TypeError:
            default_outcome.outcome_invalid_reason = "Solution contains non tuple/list elements"
            return default_outcome

        if not all(isinstance(pos, tuple) and len(pos) == 2 for pos in _solution):
            default_outcome.outcome_invalid_reason = "Solution contains invalid positions"
            return default_outcome
        if not all(isinstance(pos[0], int) and isinstance(pos[1], int) for pos in _solution):
            default_outcome.outcome_invalid_reason = "Solution contains non-integer coordinates"
            return default_outcome

        num_steps = 0
        best_position: Optional[_CoordinateType] = None
        is_valid = True
        invalid_reason = None

        if not _solution or not self.maze:
            is_valid = False
            invalid_reason = "Solution is empty or maze is not generated"

        # Check start and end
        if _solution[-1] != self.goal_position:
            is_valid = False
            invalid_reason = "Solution does not end at goal position"

        if _solution[0] != self.start_position:
            _solution.insert(0, self.start_position)

        # Check each move
        invalid_move = False
        for i in range(len(_solution)-1):
            current = _solution[i]
            next_pos = _solution[i + 1]

            # Check if move is horizontal or vertical (no diagonal)
            manhattan_dist = _manhattan_distance(
                current, next_pos)  # type: ignore
            if manhattan_dist != 1:
                invalid_move = True
                is_valid = False
                invalid_reason = "Move is diagonal or not adjacent"

            # Check bounds
            if not (0 <= next_pos[0] < self.rows and 0 <= next_pos[1] < self.cols):
                invalid_move = True
                is_valid = False
                invalid_reason = "Move out of bounds"

            # Check no walls and not traps
            if not invalid_move and self.maze[next_pos[0]][next_pos[1]] in [self.symbols.wall, self.symbols.trap]:
                invalid_move = True
                is_valid = False
                invalid_reason = "Move into wall or trap"

            if invalid_move:
                break
            else:
                num_steps += 1
                best_position = next_pos  # type: ignore

        if best_position is None:
            best_position = self.start_position

        distance_from_goal = self.find_optimal_solution(best_position)
        if distance_from_goal is not None:
            distance_from_goal = len(distance_from_goal) - 1

        return MazeOutcome(
            outcome_optimal_num_steps=optimal_num_steps,
            outcome_success=is_valid,
            outcome_best_position=best_position,
            outcome_steps_from_goal=distance_from_goal,
            outcome_num_steps=num_steps,
            outcome_path=_solution,  # type: ignore
            outcome_invalid_reason=invalid_reason
        )

    def find_optimal_solution(self, position: Optional[_CoordinateType] = None) -> Optional[_MazeTypePath]:
        """Find the shortest path from start to goal."""
        if not self.maze:
            return None
        if position is None:
            position = self.start_position  # type: ignore

        if position is None:
            return None

        position = tuple(position) if not isinstance(
            position, tuple) else position  # type: ignore

        if not isinstance(position, tuple) or len(position) != 2:
            raise ValueError("Position must be a tuple of (row, col)")

        queue = deque([(position, [position])])
        visited = {position}

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while queue:
            (row, col), path = queue.popleft()

            if (row, col) == self.goal_position:
                return path

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                new_pos = (new_row, new_col)

                if (0 <= new_row < self.rows and
                    0 <= new_col < self.cols and
                    new_pos not in visited and
                        self.maze[new_row][new_col] not in [self.symbols.wall, self.symbols.trap]):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))

        return None

    def get_maze_legend(self) -> str:
        """Get a string representation of the maze legend."""
        legend_str = (
            f"{self.symbols.current} - Current Position\n"
            f"{self.symbols.goal} - Goal Position\n"
            f"{self.symbols.path} - Path\n"
        )
        if self.config.wall_density > 0:
            legend_str += f"{self.symbols.wall} - Wall\n"
        if self.config.trap_density > 0:
            legend_str += f"{self.symbols.trap} - Trap\n"
        legend_str += f"{self.symbols.hidden} - Hidden Cell\n"
        return legend_str

    def display_maze(
            self,
            maze_to_display: Optional[_MazeTypeStr] = None,
            add_legend=False,
            border_style: Optional[Literal['simple',
                                           'thick', 'double']] = None,
            add_space: bool = False
    ) -> str:
        """Display the maze as a string with border walls."""
        if maze_to_display is None:
            maze_to_display = self.maze
        if maze_to_display is None or not maze_to_display:
            return "Maze not generated yet."

        if border_style is None:
            join_char = ' ' if add_space else ''
            maze_str = '\n'.join([join_char.join(row)
                                 for row in maze_to_display])
        else:
            # Get maze dimensions
            rows = len(maze_to_display)
            cols = len(maze_to_display[0]) if rows > 0 else 0

            # Border styles
            if border_style == 'double':
                borders = {'tl': '╔', 'tr': '╗', 'bl': '╚', 'br': '╝',
                           'h': '═', 'v': '║'}
            elif border_style == 'thick':
                borders = {'tl': '┏', 'tr': '┓', 'bl': '┗', 'br': '┛',
                           'h': '━', 'v': '┃'}
            elif border_style == 'simple':  # simple
                borders = {'tl': '┌', 'tr': '┐', 'bl': '└', 'br': '┘',
                           'h': '─', 'v': '│'}
            else:
                raise NotImplementedError(
                    f"Border style '{border_style}' is not implemented. "
                    "Available styles: 'double', 'thick', 'simple'.")

            # Build the maze with borders
            result = []

            if add_space:
                # Calculate border width (cells + spaces between)
                border_width = cols * 2 - 1 if cols > 0 else 0
            else:
                # No spaces, just the number of columns
                border_width = cols
            # Top border
            result.append(borders['tl'] + borders['h']
                          * border_width + borders['tr'])

            # Maze rows with side borders and spacing
            for row in maze_to_display:
                spaced_row = ' '.join(row)
                result.append(borders['v'] + spaced_row + borders['v'])

            # Bottom border
            result.append(borders['bl'] + borders['h']
                          * border_width + borders['br'])

            maze_str = '\n'.join(result)

        if add_legend:
            legend = self.get_maze_legend()
            maze_str += '\n\nLegend:\n\n' + legend

        return maze_str.strip()

    def display_solution(
            self,
            solution: _MazeTypePath,
            maze: Optional[_MazeTypeStr] = None) -> str:
        """Display the maze with the solution path marked."""
        if maze is None:
            maze = self.maze
        if not maze:
            raise RuntimeError("Maze not generated yet")

        if not solution or not isinstance(solution, list):
            return self.display_maze(
                add_legend=False, add_space=True, border_style='simple')

        display = [row[:] for row in maze]

        _solution = [c for c in solution]
        if len(_solution) > 0:
            if _solution[0] == self.start_position:
                _solution = _solution[1:]
        if len(_solution) > 0:
            if solution[-1] == self.goal_position:
                _solution = _solution[:-1]

        # A rejected solution may leave the grid -- "Move out of bounds" is a
        # recorded invalid_reason -- and those coordinates cannot be drawn.
        # Skipping them renders the part of the route that is on the maze; not
        # skipping them either raises IndexError or, for a negative index,
        # silently marks a cell at the opposite edge.
        num_rows, num_cols = len(display), len(display[0])
        skipped = 0
        for i in range(0, len(_solution)):
            row, col = _solution[i]
            if not (0 <= row < num_rows and 0 <= col < num_cols):
                skipped += 1
                continue
            if display[row][col] in [self.symbols.path]:
                display[row][col] = self.symbols.solution
            if display[row][col] in [self.symbols.wall, self.symbols.trap]:
                display[row][col] = '!'

        if skipped:
            logger.debug(
                "display_solution: skipped %d of %d coordinates outside the "
                "%dx%d maze", skipped, len(_solution), num_rows, num_cols)

        return self.display_maze(
            maze_to_display=display,
            add_legend=False,
            add_space=True,
            border_style='simple')

    def display_split_view(
        self,
        add_legend: bool = True,
        add_space: bool = False,
        border_style: Optional[Literal['simple', 'thick', 'double']] = None
    ) -> str:
        """Display the two splits of the maze side by side."""
        split_1_str, split_2_str = self.get_maze_splits(
            return_as_string=True,
            add_legend=False,
            add_space=add_space,
            border_style=border_style
        )

        display_str = (
            f"Map view 1:\n\n{split_1_str}\n\n"
            f"Map view 2:\n\n{split_2_str}"
        )
        if add_legend:
            display_str += f"\n\nLegend:\n\n{self.get_maze_legend()}\n"

        return display_str.strip()

    def get_maze_stats(self) -> dict:
        """Get statistics about the generated maze."""
        if not self.maze:
            return {"error": "Maze not generated"}

        stats = {
            "size": f"{self.rows}x{self.cols}",
            "total_cells": self.rows * self.cols,
            "walls": sum(row.count(self.symbols.wall) for row in self.maze),
            "traps": sum(row.count(self.symbols.trap) for row in self.maze),
            "empty_paths": sum(row.count(self.symbols.path) for row in self.maze),
            "solvable": self.is_solvable()
        }

        optimal = self.find_optimal_solution()
        if optimal:
            stats["shortest_path_length"] = len(optimal)

        return stats

    def _assert_valid_splits(self, split_1: _MazeTypeStr, split_2: _MazeTypeStr) -> bool:
        """Ensure that the two splits of the maze are valid."""
        if not split_1 or not split_2:
            return False

        # Ensure that all visible fields in both splits are the same
        for row in range(self.rows):
            for col in range(self.cols):
                if (split_1[row][col] != self.symbols.hidden and
                        split_2[row][col] != self.symbols.hidden):
                    if split_1[row][col] != split_2[row][col]:
                        raise RuntimeError(
                            f"Splits have different visible fields at the "
                            f"same position: {split_1[row][col]} !="
                            f" {split_2[row][col]} at ({row}, {col})")

        return True

    def _obfuscate_maze(self, coordinates_to_keep: set[_CoordinateType]) -> _MazeTypeStr:
        """Obfuscate the maze by hiding all cells except those in coordinates_to_keep."""
        if not self.maze:
            raise RuntimeError("Maze not generated")

        obfuscated_maze = []
        for row in range(self.rows):
            obfuscated_row = []
            for col in range(self.cols):
                if (row, col) in coordinates_to_keep:
                    obfuscated_row.append(self.maze[row][col])
                else:
                    obfuscated_row.append(self.symbols.hidden)
            obfuscated_maze.append(obfuscated_row)

        return obfuscated_maze

    def _split_maze(self) -> tuple[_MazeTypeStr, _MazeTypeStr]:
        """Split the maze into two halves."""
        if not self.maze:
            raise RuntimeError("Maze not generated")

        # Each split should contain the current and goal positions
        # The two splits can be partially overlapping
        # Combined, the splits should cover the entire maze
        total_cells = self.rows * self.cols
        split_1_size = math.ceil(total_cells * self.config.split_1_proportion)
        split_2_size = math.ceil(total_cells * self.config.split_2_proportion)

        start_and_goal_coordinate = set(
            [self.start_position, self.goal_position])
        shared_size = len(start_and_goal_coordinate)

        # The two splits must jointly cover the maze while both showing start
        # and goal, so |split_1| + |split_2| - |overlap| == total_cells with an
        # overlap of at least start+goal. Requested proportions summing below
        # (total + 2) / total cannot satisfy that. Grow both splits evenly to
        # the smallest sizes that do, so a symmetric request stays symmetric.
        deficit = (total_cells + shared_size) - (split_1_size + split_2_size)
        if deficit > 0:
            split_1_size += math.ceil(deficit / 2)
            split_2_size += deficit // 2

        # A split can never exceed the maze, nor show less than start+goal.
        split_1_size = min(max(split_1_size, shared_size), total_cells)
        split_2_size = min(max(split_2_size, shared_size), total_cells)

        for label, size, requested in (
            ("split_1", split_1_size, self.config.split_1_proportion),
            ("split_2", split_2_size, self.config.split_2_proportion),
        ):
            achieved = size / total_cells
            # Tolerate the sub-cell rounding that math.ceil introduces, plus each
            # split's share of the start and goal cells both agents must see:
            # that much is added to every evenly split maze (19/36 rather than
            # 18/36 for 0.5 on a 6x6) and is expected, not worth a warning.
            tolerance = (math.ceil(shared_size / 2) + 1) / total_cells
            if abs(achieved - requested) > tolerance:
                logger.warning(
                    "%s proportion cannot be honoured: requested %.3f, using "
                    "%.3f (%d/%d cells). Full coverage with start and goal "
                    "visible to both agents requires the two proportions to "
                    "sum to at least %.3f.",
                    label, requested, achieved, size, total_cells,
                    (total_cells + shared_size) / total_cells)

        # Generate all possible coordinates in the maze
        all_other_coordinates = sorted(set((row, col) for row in range(
            # type: ignore
            self.rows) for col in range(self.cols)) - start_and_goal_coordinate)

        # Start with start and goal positions, then randomly select additional cells
        split_1_cells = set(start_and_goal_coordinate)
        split_1_cells.update(random.sample(
            all_other_coordinates, split_1_size - shared_size))

        # Leftover cells go into split 2
        non_covered_cells = sorted(set(all_other_coordinates) - split_1_cells)
        split_2_cells = set(start_and_goal_coordinate)
        split_2_cells.update(non_covered_cells)
        remaining_capacity = split_2_size - len(split_2_cells)
        if remaining_capacity > 0:
            # If split 2 is smaller than required, fill it with random cells from split 1
            remaining_cells = sorted(
                set(all_other_coordinates) - split_2_cells)
            split_2_cells.update(random.sample(
                remaining_cells, remaining_capacity))

        # We check to ensure that the union of the splits covers the entire maze
        if len(split_1_cells.union(split_2_cells)) != total_cells:
            raise RuntimeError("Splits do not cover the entire maze")

        # Finally, we create the two splits
        split_1 = self._obfuscate_maze(split_1_cells)
        split_2 = self._obfuscate_maze(split_2_cells)

        # Ensure that all visible fields in both splits are the same
        self._assert_valid_splits(split_1, split_2)

        return split_1, split_2

    @overload
    def get_maze_splits(
        self, return_as_string: Literal[True],
        add_legend: bool = ...,
        add_space: bool = ...,
        border_style: Optional[Literal['simple', 'thick', 'double']] = ...
    ) -> tuple[str, str]: ...

    @overload
    def get_maze_splits(
        self, return_as_string: Literal[False],
        add_legend: bool = ...,
        add_space: bool = ...,
        border_style: Optional[Literal['simple', 'thick', 'double']] = ...
    ) -> tuple[_MazeTypeStr, _MazeTypeStr]: ...

    def get_maze_splits(
        self,
        return_as_string: bool = True,
        add_legend: bool = False,
        add_space: bool = False,
        border_style: Optional[Literal['simple', 'thick', 'double']] = None
    ) -> tuple[_MazeTypeStr, _MazeTypeStr] | tuple[str, str]:
        """Get the two splits of the maze."""

        if return_as_string:
            split_1_str = self.display_maze(
                self.maze_split_1,
                add_legend=add_legend,
                add_space=add_space,
                border_style=border_style
            )
            split_2_str = self.display_maze(
                self.maze_split_2,
                add_legend=add_legend,
                add_space=add_space,
                border_style=border_style
            )
            return split_1_str, split_2_str
        return self.maze_split_1, self.maze_split_2

    def __str__(self) -> str:
        return self.display_maze(add_legend=True, border_style='simple', add_space=True)

    def __repr__(self) -> str:
        maze_str = self.display_maze(
            add_legend=False, border_style='simple', add_space=True)
        maze_str_split = self.display_split_view(
            add_legend=True, add_space=True, border_style='simple')
        config_dict = self.config.to_dict()
        config_str = ""
        for k, v in config_dict.items():
            config_str += f"{k}:".ljust(25) + str(v) + "\n"

        rep = f"Full:\n{maze_str}\n\n{maze_str_split}\n\nConfig:\n{'-'*30}\n{config_str}"

        return rep


def _example_maze(
    size: tuple[int, int] = (5, 5),
    wall_density: float = 0.3,
    trap_density: float = 0.0,
    min_path_length: int = 2,
    max_path_length: int = 15,
    random_seed: int = 42
):
    # Build one maze and print it with its optimal and sampled solutions.
    config = MazeConfig(
        size=size,
        wall_density=wall_density,
        trap_density=trap_density,
        _min_path_length=min_path_length,
        _max_path_length=max_path_length,
        random_seed=random_seed
    )

    game = MazeGame(config=config)

    print("Generated Maze:")
    print(game.display_maze(
        add_legend=True,
        border_style='simple',
        add_space=True))
    print()

    stats = game.get_maze_stats()
    print("Maze Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    solution = game.find_optimal_solution()
    if solution:
        print(f"\nOptimal solution: {len(solution)} steps")
        print("Is valid:", game.verify_solution(solution))
        print(game.display_solution(solution))
        assert game.verify_solution(
            solution).outcome_success, "Optimal solution is not valid!"
    if isinstance(game.sampled_path, list):
        print(f"\nSampled solution: {len(game.sampled_path)} steps")
        print("Is valid:", game.verify_solution(game.sampled_path))
        print(game.display_solution(game.sampled_path))
        assert game.verify_solution(
            game.sampled_path).outcome_success, "Sampled path is not valid!"

    print(game.display_split_view(
        add_legend=True, add_space=True, border_style='simple'))


# python -m src.tasks.maze_game --random_seed=1
if __name__ == "__main__":
    fire.Fire(_example_maze)
