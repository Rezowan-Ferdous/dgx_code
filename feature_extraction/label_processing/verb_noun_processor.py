"""
Verb-Noun Processor
Merges verb and noun labels to create action descriptions
Commonly used in action recognition datasets (EPIC-KITCHENS, etc.)
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import pandas as pd


class VerbNounProcessor:
    """
    Process and merge verb-noun pairs into action descriptions

    Supports:
    - Verb-noun pair combination
    - Multi-class to single-class conversion
    - Description generation from indices
    - Label smoothing for verb-noun combinations
    """

    def __init__(
        self,
        verb_classes: List[str],
        noun_classes: List[str],
        separator: str = "_",
        use_templates: bool = True
    ):
        """
        Initialize verb-noun processor

        Args:
            verb_classes: List of verb class names
            noun_classes: List of noun class names
            separator: Separator for combining verb-noun
            use_templates: Use natural language templates
        """
        self.verb_classes = verb_classes
        self.noun_classes = noun_classes
        self.num_verbs = len(verb_classes)
        self.num_nouns = len(noun_classes)
        self.separator = separator
        self.use_templates = use_templates

        # Create action class mappings
        self._create_action_classes()

        # Natural language templates
        self.templates = [
            "{verb} {noun}",
            "{verb} the {noun}",
            "person {verb} {noun}",
            "{verb}ing {noun}",
            "{verb}s {noun}",
        ]

    def _create_action_classes(self):
        """Create combined action classes from verb-noun pairs"""
        self.action_classes = []
        self.verb_noun_to_action = {}

        action_id = 0
        for v_idx, verb in enumerate(self.verb_classes):
            for n_idx, noun in enumerate(self.noun_classes):
                action_name = f"{verb}{self.separator}{noun}"
                self.action_classes.append(action_name)
                self.verb_noun_to_action[(v_idx, n_idx)] = action_id
                action_id += 1

        self.num_actions = len(self.action_classes)

    def merge_verb_noun(
        self,
        verb_idx: Union[int, np.ndarray, torch.Tensor],
        noun_idx: Union[int, np.ndarray, torch.Tensor]
    ) -> Union[int, np.ndarray, torch.Tensor]:
        """
        Merge verb and noun indices into action index

        Args:
            verb_idx: Verb class index/indices
            noun_idx: Noun class index/indices

        Returns:
            Action class index/indices
        """
        if isinstance(verb_idx, int) and isinstance(noun_idx, int):
            return self.verb_noun_to_action[(verb_idx, noun_idx)]

        elif isinstance(verb_idx, np.ndarray):
            action_idx = np.array([
                self.verb_noun_to_action[(v, n)]
                for v, n in zip(verb_idx, noun_idx)
            ])
            return action_idx

        elif isinstance(verb_idx, torch.Tensor):
            verb_idx_np = verb_idx.cpu().numpy()
            noun_idx_np = noun_idx.cpu().numpy()
            action_idx = np.array([
                self.verb_noun_to_action[(v, n)]
                for v, n in zip(verb_idx_np, noun_idx_np)
            ])
            return torch.from_numpy(action_idx).to(verb_idx.device)

        else:
            raise TypeError(f"Unsupported type: {type(verb_idx)}")

    def split_action(
        self,
        action_idx: Union[int, np.ndarray, torch.Tensor]
    ) -> Tuple[Union[int, np.ndarray, torch.Tensor], Union[int, np.ndarray, torch.Tensor]]:
        """
        Split action index into verb and noun indices

        Args:
            action_idx: Action class index/indices

        Returns:
            Tuple of (verb_idx, noun_idx)
        """
        if isinstance(action_idx, int):
            verb_idx = action_idx // self.num_nouns
            noun_idx = action_idx % self.num_nouns
            return verb_idx, noun_idx

        elif isinstance(action_idx, np.ndarray):
            verb_idx = action_idx // self.num_nouns
            noun_idx = action_idx % self.num_nouns
            return verb_idx, noun_idx

        elif isinstance(action_idx, torch.Tensor):
            verb_idx = action_idx // self.num_nouns
            noun_idx = action_idx % self.num_nouns
            return verb_idx, noun_idx

        else:
            raise TypeError(f"Unsupported type: {type(action_idx)}")

    def get_description(
        self,
        verb_idx: int,
        noun_idx: int,
        template_id: int = 0,
        capitalize: bool = True
    ) -> str:
        """
        Generate natural language description from verb-noun pair

        Args:
            verb_idx: Verb class index
            noun_idx: Noun class index
            template_id: Template to use (0-4)
            capitalize: Capitalize first letter

        Returns:
            Natural language description
        """
        verb = self.verb_classes[verb_idx]
        noun = self.noun_classes[noun_idx]

        if self.use_templates and template_id < len(self.templates):
            description = self.templates[template_id].format(verb=verb, noun=noun)
        else:
            description = f"{verb} {noun}"

        if capitalize:
            description = description[0].upper() + description[1:]

        return description

    def get_all_descriptions(
        self,
        template_id: int = 0,
        capitalize: bool = True
    ) -> List[str]:
        """
        Get descriptions for all action classes

        Args:
            template_id: Template to use
            capitalize: Capitalize first letter

        Returns:
            List of descriptions for all actions
        """
        descriptions = []
        for v_idx in range(self.num_verbs):
            for n_idx in range(self.num_nouns):
                desc = self.get_description(v_idx, n_idx, template_id, capitalize)
                descriptions.append(desc)

        return descriptions

    def create_joint_labels(
        self,
        verb_labels: torch.Tensor,
        noun_labels: torch.Tensor,
        label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """
        Create joint verb-noun labels with optional label smoothing

        Args:
            verb_labels: Verb class labels [B] or [B, num_verbs]
            noun_labels: Noun class labels [B] or [B, num_nouns]
            label_smoothing: Label smoothing factor

        Returns:
            Joint action labels [B, num_actions]
        """
        batch_size = verb_labels.shape[0]
        device = verb_labels.device

        # Initialize action labels
        action_labels = torch.zeros(batch_size, self.num_actions, device=device)

        # Check if labels are one-hot or indices
        if verb_labels.dim() == 1:  # Indices
            # Convert to one-hot
            verb_onehot = torch.zeros(batch_size, self.num_verbs, device=device)
            verb_onehot.scatter_(1, verb_labels.unsqueeze(1), 1)
        else:  # Already one-hot
            verb_onehot = verb_labels

        if noun_labels.dim() == 1:  # Indices
            noun_onehot = torch.zeros(batch_size, self.num_nouns, device=device)
            noun_onehot.scatter_(1, noun_labels.unsqueeze(1), 1)
        else:
            noun_onehot = noun_labels

        # Create joint distribution (outer product)
        for i in range(batch_size):
            joint = torch.outer(verb_onehot[i], noun_onehot[i]).flatten()
            action_labels[i] = joint

        # Apply label smoothing if specified
        if label_smoothing > 0:
            action_labels = action_labels * (1 - label_smoothing) + label_smoothing / self.num_actions

        return action_labels

    def to_dataframe(
        self,
        verb_indices: List[int],
        noun_indices: List[int],
        include_descriptions: bool = True
    ) -> pd.DataFrame:
        """
        Create pandas DataFrame with verb-noun mappings

        Args:
            verb_indices: List of verb indices
            noun_indices: List of noun indices
            include_descriptions: Include natural language descriptions

        Returns:
            DataFrame with columns: [verb_idx, noun_idx, verb, noun, action_idx, action_name, description]
        """
        data = {
            'verb_idx': verb_indices,
            'noun_idx': noun_indices,
            'verb': [self.verb_classes[v] for v in verb_indices],
            'noun': [self.noun_classes[n] for n in noun_indices],
            'action_idx': [self.merge_verb_noun(v, n) for v, n in zip(verb_indices, noun_indices)],
            'action_name': [f"{self.verb_classes[v]}{self.separator}{self.noun_classes[n]}"
                           for v, n in zip(verb_indices, noun_indices)]
        }

        if include_descriptions:
            data['description'] = [self.get_description(v, n) for v, n in zip(verb_indices, noun_indices)]

        return pd.DataFrame(data)

    def save_mapping(self, filepath: str):
        """Save verb-noun to action mapping"""
        import json

        mapping = {
            'verb_classes': self.verb_classes,
            'noun_classes': self.noun_classes,
            'action_classes': self.action_classes,
            'num_verbs': self.num_verbs,
            'num_nouns': self.num_nouns,
            'num_actions': self.num_actions,
        }

        with open(filepath, 'w') as f:
            json.dump(mapping, f, indent=2)

    @classmethod
    def load_mapping(cls, filepath: str):
        """Load verb-noun processor from saved mapping"""
        import json

        with open(filepath, 'r') as f:
            mapping = json.load(f)

        return cls(
            verb_classes=mapping['verb_classes'],
            noun_classes=mapping['noun_classes']
        )


class VerbNounDataset:
    """
    Helper class for datasets with verb-noun annotations
    Common in action recognition (EPIC-KITCHENS, Charades, etc.)
    """

    def __init__(
        self,
        annotations_df: pd.DataFrame,
        verb_column: str = 'verb',
        noun_column: str = 'noun',
        processor: Optional[VerbNounProcessor] = None
    ):
        """
        Initialize verb-noun dataset helper

        Args:
            annotations_df: DataFrame with verb-noun annotations
            verb_column: Column name for verb labels
            noun_column: Column name for noun labels
            processor: VerbNounProcessor instance
        """
        self.annotations_df = annotations_df
        self.verb_column = verb_column
        self.noun_column = noun_column

        # Create processor if not provided
        if processor is None:
            unique_verbs = sorted(annotations_df[verb_column].unique())
            unique_nouns = sorted(annotations_df[noun_column].unique())
            self.processor = VerbNounProcessor(unique_verbs, unique_nouns)
        else:
            self.processor = processor

    def get_action_labels(self) -> np.ndarray:
        """Get action labels from verb-noun pairs"""
        verb_indices = self.annotations_df[self.verb_column].values
        noun_indices = self.annotations_df[self.noun_column].values

        action_labels = self.processor.merge_verb_noun(verb_indices, noun_indices)
        return action_labels

    def get_descriptions(self) -> List[str]:
        """Get natural language descriptions"""
        verb_indices = self.annotations_df[self.verb_column].values
        noun_indices = self.annotations_df[self.noun_column].values

        descriptions = [
            self.processor.get_description(v, n)
            for v, n in zip(verb_indices, noun_indices)
        ]
        return descriptions
