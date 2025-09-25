from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Set, Callable, cast, Any
from datasets import Dataset
import json
from .position_finder import Position
import os


@dataclass
class CharacterPositions:
    """Container for character positions of interest."""

    positions: Dict[str, Union[Position, List[Position]]]

    def __getitem__(self, key: str) -> Union[Position, List[Position]]:
        return self.positions[key]

    def keys(self) -> Set[str]:
        return set(self.positions.keys())


@dataclass
class ProbingExample:
    """Single example for probing."""

    text: str
    label: Union[int, float]  # Numeric label
    label_text: str  # Original text label
    character_positions: Optional[CharacterPositions] = None
    group_id: Optional[str] = None  # For cross-validation/grouping
    attributes: Optional[Dict] = (
        None  # Additional attributes about this example (previously called metadata)
    )


class ProbingDataset:
    """Base dataset for probing experiments."""

    def __init__(
        self,
        examples: List[ProbingExample],
        task_type: str = "classification",  # or "regression"
        valid_layers: Optional[List[str]] = None,
        label_mapping: Optional[Dict[str, int]] = None,
        dataset_attributes: Optional[
            Dict
        ] = None,  # Dataset-level attributes (previously metadata)
    ):
        self.task_type = task_type
        self.valid_layers = valid_layers
        self.label_mapping = label_mapping
        self.dataset_attributes = dataset_attributes or {}
        self.examples = examples
        self.dataset = self._to_hf_dataset()

        # Infer position types from examples if available
        self.position_types = set()
        for example in self.examples:
            if example.character_positions:
                self.position_types.update(example.character_positions.keys())

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def add_target_positions(
        self, key: str, finder: Callable[[str], Union[Position, List[Position]]]
    ) -> None:
        """Add target positions using a finding strategy.

        Args:
            key: Name for this set of positions
            finder: Function that finds positions in text
        """
        for example in self.examples:
            positions = finder(example.text)

            # Initialize character_positions if needed
            if example.character_positions is None:
                example.character_positions = CharacterPositions({})

            # Add new positions
            example.character_positions.positions[key] = positions

        # Update the position_types attribute with the new key
        self.position_types.add(key)

    def _to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace dataset format."""
        data_dict: Dict[str, List[Optional[Union[str, int, float, List]]]] = {
            "text": [],
            "label": [],
            "label_text": [],
            "group_id": [],
            "attributes_json": [],  # Add column for attributes JSON string
        }

        # Add position data if available
        position_keys = set()
        for ex in self.examples:
            if ex.character_positions:
                position_keys.update(ex.character_positions.keys())

        for key in position_keys:
            data_dict[f"char_pos_{key}_start"] = []
            data_dict[f"char_pos_{key}_end"] = []
            data_dict[f"char_pos_{key}_multi"] = []  # For multiple matches

        # Populate data
        for ex in self.examples:
            data_dict["text"].append(ex.text)
            data_dict["label"].append(ex.label)
            data_dict["label_text"].append(ex.label_text)
            data_dict["group_id"].append(ex.group_id)
            # Save attributes as JSON string
            attributes_str = json.dumps(ex.attributes) if ex.attributes else None
            data_dict["attributes_json"].append(attributes_str)

            # Add position data
            if ex.character_positions:
                for key in position_keys:
                    pos = ex.character_positions.positions.get(key)
                    if pos is None:
                        data_dict[f"char_pos_{key}_start"].append(None)
                        data_dict[f"char_pos_{key}_end"].append(None)
                        data_dict[f"char_pos_{key}_multi"].append([])
                    elif isinstance(pos, Position):
                        data_dict[f"char_pos_{key}_start"].append(pos.start)
                        data_dict[f"char_pos_{key}_end"].append(pos.end)
                        data_dict[f"char_pos_{key}_multi"].append([])
                    else:  # List[Position]
                        # Store first match in main columns
                        data_dict[f"char_pos_{key}_start"].append(
                            pos[0].start if pos else None
                        )
                        data_dict[f"char_pos_{key}_end"].append(
                            pos[0].end if pos else None
                        )
                        # Store all matches in multi column
                        data_dict[f"char_pos_{key}_multi"].append(
                            [(p.start, p.end) for p in pos]
                        )
            else:
                for key in position_keys:
                    data_dict[f"char_pos_{key}_start"].append(None)
                    data_dict[f"char_pos_{key}_end"].append(None)
                    data_dict[f"char_pos_{key}_multi"].append([])

        return Dataset.from_dict(data_dict)

    @classmethod
    def from_hf_dataset(
        cls,
        dataset: Dataset,
        position_types: List[str],
        label_mapping: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> "ProbingDataset":
        """Create ProbingDataset from a HuggingFace dataset."""
        examples = []

        for item_raw in dataset:
            item = cast(Dict[str, Any], item_raw)  # Cast to Dict with Any values
            # Explicitly type the positions dictionary
            positions: Dict[str, Union[Position, List[Position]]] = {}
            for pt in position_types:
                # Check for single position
                start = item.get(f"char_pos_{pt}_start")
                end = item.get(f"char_pos_{pt}_end")
                multi = item.get(f"char_pos_{pt}_multi", [])

                if multi:
                    # Handle multiple positions
                    positions[pt] = [Position(start=s, end=e) for s, e in multi]
                elif start is not None and end is not None:
                    # Handle single position
                    positions[pt] = Position(start=start, end=end)
                # Only add key if multi or start/end were found
                elif multi or (start is not None and end is not None):
                    pass  # Already handled above
                else:
                    # If no position data found for this key, don't add it
                    pass

            # Only create CharacterPositions if positions dict is not empty
            char_positions_obj = CharacterPositions(positions) if positions else None

            # Load attributes from JSON string
            attributes = None
            attributes_str = item.get("attributes_json")
            if isinstance(attributes_str, str):
                try:
                    attributes = json.loads(attributes_str)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Failed to decode attributes JSON: {attributes_str}"
                    )
                    attributes = None  # Or handle error as appropriate

            examples.append(
                ProbingExample(
                    text=str(item["text"]),
                    label=float(item["label"]),
                    label_text=str(item["label_text"]),
                    character_positions=char_positions_obj,  # Use potentially None object
                    group_id=item.get("group_id"),
                    attributes=attributes,  # Add loaded attributes
                )
            )

        return cls(
            examples=examples,
            label_mapping=label_mapping,  # Remove position_types from kwargs
            **kwargs,
        )

    def save(self, path: str) -> None:
        """Save dataset to disk."""
        # Save HF dataset
        self.dataset.save_to_disk(f"{path}/hf_dataset")

        # Get position types from the examples
        position_types = set()
        for example in self.examples:
            if example.character_positions:
                position_types.update(example.character_positions.keys())

        # Save attributes (previously metadata)
        attributes = {
            "task_type": self.task_type,
            "valid_layers": self.valid_layers,
            "label_mapping": self.label_mapping,
            "dataset_attributes": self.dataset_attributes,  # Renamed from metadata
            "position_types": list(position_types),  # Add position_types to attributes
        }

        with open(
            f"{path}/dataset_attributes.json", "w"
        ) as f:  # Renamed from metadata.json
            json.dump(attributes, f)

    @classmethod
    def load(cls, path: str) -> "ProbingDataset":
        """Load dataset from disk."""
        # Load HF dataset
        dataset = Dataset.load_from_disk(f"{path}/hf_dataset")

        # Try the new attributes filename first, fall back to old metadata.json for backward compatibility
        attributes_file = (
            f"{path}/dataset_attributes.json"
            if os.path.exists(f"{path}/dataset_attributes.json")
            else f"{path}/metadata.json"
        )

        # Load attributes
        with open(attributes_file, "r") as f:
            attributes = json.load(f)

        # Handle backward compatibility - if loading old file format
        dataset_attributes_key = (
            "dataset_attributes" if "dataset_attributes" in attributes else "metadata"
        )

        return cls.from_hf_dataset(
            dataset=dataset,
            position_types=attributes["position_types"],
            label_mapping=attributes["label_mapping"],
            task_type=attributes["task_type"],
            valid_layers=attributes["valid_layers"],
            dataset_attributes=attributes[dataset_attributes_key],
        )

    def train_test_split(
        self, test_size: float = 0.2, shuffle: bool = True, seed: Optional[int] = None
    ) -> tuple["ProbingDataset", "ProbingDataset"]:
        """Split into train and test datasets."""
        split = self.dataset.train_test_split(
            test_size=test_size, shuffle=shuffle, seed=seed
        )

        # Get position types from the original dataset
        original_position_types = list(self.position_types)

        # Pass original position types when creating new datasets
        train = self.from_hf_dataset(
            dataset=split["train"],
            position_types=original_position_types,
            label_mapping=self.label_mapping,
            task_type=self.task_type,
            valid_layers=self.valid_layers,
            dataset_attributes=self.dataset_attributes,  # Renamed metadata
        )

        test = self.from_hf_dataset(
            dataset=split["test"],
            position_types=original_position_types,
            label_mapping=self.label_mapping,
            task_type=self.task_type,
            valid_layers=self.valid_layers,
            dataset_attributes=self.dataset_attributes,  # Renamed metadata
        )

        return train, test
