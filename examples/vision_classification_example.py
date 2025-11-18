"""
Example: Image Classification with Vision Transformers

This script demonstrates how to:
1. Create a vision classification task
2. Load a dataset (CIFAR-10)
3. Train a Vision Transformer
4. Evaluate and make predictions
"""

import torch
from domains.computer_vision.classification import (
    ImageClassificationTask,
    get_classification_dataset,
    create_dataloader
)


def main():
    # Configuration
    CONFIG = {
        'architecture': 'vit',
        'variant': 'small',
        'num_classes': 10,
        'batch_size': 128,
        'epochs': 50,
        'learning_rate': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 50)
    print("Vision Classification Example: ViT on CIFAR-10")
    print("=" * 50)

    # Create task
    print("\n1. Creating Vision Transformer model...")
    task = ImageClassificationTask(
        architecture=CONFIG['architecture'],
        variant=CONFIG['variant'],
        num_classes=CONFIG['num_classes'],
        pretrained=True,
        device=CONFIG['device']
    )
    print(f"✓ Model created: {CONFIG['architecture']}-{CONFIG['variant']}")

    # Prepare datasets
    print("\n2. Loading CIFAR-10 dataset...")
    train_dataset = get_classification_dataset(
        'cifar10',
        split='train',
        img_size=224,
        augmentation='basic'
    )
    val_dataset = get_classification_dataset(
        'cifar10',
        split='val',
        img_size=224,
        augmentation='none'
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4
    )
    print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Setup training
    print("\n3. Configuring training...")
    task.prepare_training(
        optimizer='adamw',
        lr=CONFIG['learning_rate'],
        scheduler='cosine',
        weight_decay=0.01
    )
    print("✓ Training configured")

    # Train
    print("\n4. Starting training...")
    task.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=CONFIG['epochs'],
        use_amp=True,
        save_dir='./checkpoints/vit_cifar10',
        early_stopping_patience=10
    )
    print("✓ Training completed")

    # Evaluate
    print("\n5. Final evaluation...")
    metrics = task.evaluate(val_loader)
    print(f"✓ Final Accuracy: {metrics['accuracy']:.2f}%")
    print(f"✓ Final Loss: {metrics['loss']:.4f}")

    # Make predictions on sample images
    print("\n6. Making predictions on sample images...")
    sample_batch = next(iter(val_loader))
    images, labels = sample_batch
    predictions = task.predict(images[:5])

    print("\nSample predictions:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    for i, (pred, true) in enumerate(zip(predictions, labels[:5])):
        print(f"  Image {i+1}: Predicted={class_names[pred]}, True={class_names[true]}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()
