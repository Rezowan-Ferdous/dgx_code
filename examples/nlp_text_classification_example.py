"""
Example: Text Classification with BERT

This script demonstrates how to:
1. Create a text classification task
2. Load a dataset (IMDb sentiment)
3. Train BERT for sentiment analysis
4. Evaluate and make predictions
"""

import torch
from torch.utils.data import DataLoader
from domains.nlp.text_classification import (
    TextClassificationTask,
    get_text_classification_dataset
)


def main():
    # Configuration
    CONFIG = {
        'architecture': 'bert',
        'variant': 'base',
        'num_classes': 2,
        'batch_size': 16,
        'epochs': 3,
        'learning_rate': 2e-5,
        'max_length': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 50)
    print("Text Classification Example: BERT on IMDb")
    print("=" * 50)

    # Create task
    print("\n1. Creating BERT model...")
    task = TextClassificationTask(
        architecture=CONFIG['architecture'],
        variant=CONFIG['variant'],
        num_classes=CONFIG['num_classes'],
        device=CONFIG['device']
    )
    print(f"✓ Model created: {CONFIG['architecture']}-{CONFIG['variant']}")
    print(f"  Model has {sum(p.numel() for p in task.model.parameters()):,} parameters")

    # Prepare datasets
    print("\n2. Loading IMDb dataset...")
    print("  (This may take a few minutes on first run)")

    train_dataset = get_text_classification_dataset(
        'imdb',
        split='train',
        tokenizer=task.model.tokenizer,
        max_length=CONFIG['max_length']
    )

    # Use a smaller subset for this example
    print(f"✓ Dataset loaded: {len(train_dataset)} samples")
    print("  Using first 1000 samples for quick training...")

    # Create a smaller subset
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, range(1000))

    train_loader = DataLoader(
        train_subset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=2
    )

    print(f"✓ DataLoader created with batch size {CONFIG['batch_size']}")

    # Train
    print("\n3. Starting training...")
    print(f"  Training for {CONFIG['epochs']} epochs...")
    task.train(
        train_loader=train_loader,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate']
    )
    print("✓ Training completed")

    # Make predictions
    print("\n4. Making predictions on sample texts...")

    test_texts = [
        "This movie was absolutely fantastic! Great acting and amazing plot.",
        "Terrible film. Waste of time and money. Very disappointed.",
        "An okay movie, nothing special but not terrible either.",
        "One of the best films I've ever seen! Highly recommend!",
        "Boring and predictable. Would not watch again.",
    ]

    predictions = task.predict(test_texts)

    print("\nSample predictions:")
    sentiment_labels = ['Negative', 'Positive']
    for i, (text, pred) in enumerate(zip(test_texts, predictions)):
        print(f"\n  Text {i+1}:")
        print(f"    \"{text[:60]}...\"" if len(text) > 60 else f"    \"{text}\"")
        print(f"    Predicted sentiment: {sentiment_labels[pred]}")

    # Test with custom text
    print("\n5. Interactive prediction:")
    custom_text = "The acting was superb and the storyline kept me engaged throughout!"
    custom_pred = task.predict([custom_text])[0]
    print(f"  Input: \"{custom_text}\"")
    print(f"  Prediction: {sentiment_labels[custom_pred]}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)
    print("\nTips:")
    print("  - To train on full dataset, remove the Subset wrapper")
    print("  - Adjust batch_size based on your GPU memory")
    print("  - Try different models: 'roberta', 'deberta', 'distilbert'")
    print("  - Use different datasets: 'sst2', 'ag_news', 'yelp'")


if __name__ == '__main__':
    main()
