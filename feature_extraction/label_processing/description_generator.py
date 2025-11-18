"""
Description Generator
Generate natural language descriptions from labels using templates or language models
"""

import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class DescriptionGenerator:
    """
    Generate natural language descriptions from action labels

    Supports:
    - Template-based generation
    - T5-based generation (learned)
    - Multiple description styles
    """

    def __init__(
        self,
        method: str = "template",  # 'template' or 't5'
        model_name: str = "t5-small",
        templates: Optional[List[str]] = None
    ):
        """
        Initialize description generator

        Args:
            method: Generation method ('template' or 't5')
            model_name: HuggingFace model name for T5
            templates: Custom templates for template-based generation
        """
        self.method = method

        if method == "t5":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.eval()

        # Default templates
        self.templates = templates or [
            "A person is {action}",
            "Someone is {action}",
            "{action} is being performed",
            "The video shows {action}",
            "Action: {action}",
        ]

    def generate_from_label(
        self,
        label: str,
        style: str = "descriptive",  # 'simple', 'descriptive', 'detailed'
        max_length: int = 50
    ) -> str:
        """
        Generate description from action label

        Args:
            label: Action label (e.g., "cutting_apple")
            style: Description style
            max_length: Maximum description length

        Returns:
            Natural language description
        """
        if self.method == "template":
            return self._generate_template(label, style)
        elif self.method == "t5":
            return self._generate_t5(label, style, max_length)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _generate_template(self, label: str, style: str) -> str:
        """Generate description using templates"""
        # Clean label
        action = label.replace("_", " ").replace("-", " ")

        if style == "simple":
            return action.capitalize()
        elif style == "descriptive":
            return f"A person is {action}"
        elif style == "detailed":
            return f"The video shows a person {action} in the scene"
        else:
            # Use random template
            import random
            template = random.choice(self.templates)
            return template.format(action=action)

    def _generate_t5(self, label: str, style: str, max_length: int) -> str:
        """Generate description using T5"""
        # Create prompt
        prompt = f"generate description for action: {label.replace('_', ' ')}"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return description

    def generate_batch(
        self,
        labels: List[str],
        style: str = "descriptive"
    ) -> List[str]:
        """
        Generate descriptions for multiple labels

        Args:
            labels: List of action labels
            style: Description style

        Returns:
            List of descriptions
        """
        return [self.generate_from_label(label, style) for label in labels]

    def generate_from_verb_noun(
        self,
        verb: str,
        noun: str,
        include_article: bool = True,
        tense: str = "present"  # 'present', 'past', 'future'
    ) -> str:
        """
        Generate description from verb-noun pair

        Args:
            verb: Verb (e.g., "cutting")
            noun: Noun (e.g., "apple")
            include_article: Include article (a/an/the)
            tense: Verb tense

        Returns:
            Natural language description
        """
        # Determine article
        article = ""
        if include_article:
            if noun[0].lower() in 'aeiou':
                article = "an "
            else:
                article = "a "

        # Adjust verb tense
        if tense == "present":
            verb_form = f"{verb}ing"
        elif tense == "past":
            if verb.endswith('e'):
                verb_form = f"{verb}d"
            else:
                verb_form = f"{verb}ed"
        elif tense == "future":
            verb_form = f"will {verb}"
        else:
            verb_form = verb

        return f"A person is {verb_form} {article}{noun}"

    def add_context(
        self,
        description: str,
        context: Dict[str, str]
    ) -> str:
        """
        Add contextual information to description

        Args:
            description: Base description
            context: Dictionary with context info (location, time, etc.)

        Returns:
            Enhanced description
        """
        enhanced = description

        if 'location' in context:
            enhanced += f" in the {context['location']}"

        if 'time' in context:
            enhanced += f" during {context['time']}"

        if 'tool' in context:
            enhanced += f" using a {context['tool']}"

        return enhanced

    def create_caption_dataset(
        self,
        labels: List[str],
        generate_multiple: bool = True,
        variations_per_label: int = 3
    ) -> Dict[str, List[str]]:
        """
        Create caption dataset from labels

        Args:
            labels: List of action labels
            generate_multiple: Generate multiple captions per label
            variations_per_label: Number of variations

        Returns:
            Dictionary mapping labels to list of captions
        """
        caption_dataset = {}

        styles = ["simple", "descriptive", "detailed"]

        for label in labels:
            if generate_multiple:
                captions = []
                for i in range(variations_per_label):
                    style = styles[i % len(styles)]
                    caption = self.generate_from_label(label, style)
                    captions.append(caption)
                caption_dataset[label] = captions
            else:
                caption_dataset[label] = [self.generate_from_label(label)]

        return caption_dataset


class LabelEncoder:
    """
    Encode labels for different tasks (classification, retrieval, etc.)
    """

    def __init__(self, num_classes: int, embedding_dim: int = 512):
        """
        Initialize label encoder

        Args:
            num_classes: Number of classes
            embedding_dim: Dimension of label embeddings
        """
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Learnable label embeddings
        self.label_embeddings = torch.nn.Embedding(num_classes, embedding_dim)

    def encode_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Encode labels to embeddings

        Args:
            labels: Label indices [B] or [B, num_classes] (one-hot)

        Returns:
            Label embeddings [B, embedding_dim]
        """
        if labels.dim() == 1:  # Indices
            return self.label_embeddings(labels)
        else:  # One-hot
            # Weighted average of embeddings
            return torch.matmul(labels.float(), self.label_embeddings.weight)

    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get embeddings for all classes

        Returns:
            All label embeddings [num_classes, embedding_dim]
        """
        return self.label_embeddings.weight
