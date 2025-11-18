"""
Main entry point for running experiments
Supports training, testing, evaluation, and reporting
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add framework to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework import ExperimentConfig, ExperimentManager
from framework.visualization import TrainingVisualizer, ResultsVisualizer
from framework.reporting import ReportGenerator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run surgical action recognition experiments')

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'eval', 'full'],
        default='full',
        help='Experiment mode: train, test, eval, or full (default: full)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for resuming training or testing'
    )

    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU device ID(s) to use (default: 0)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )

    return parser.parse_args()


def setup_model_and_data(config, experiment_manager):
    """
    Setup model and data loaders based on configuration

    This is a placeholder - you should customize this based on your specific models and datasets
    """
    # Import your models and datasets
    from models.mymodel import MyAsformer
    from datasets.rarp import RARPDataset
    from torch.utils.data import DataLoader

    # Create model
    model = MyAsformer(
        num_classes=config.model.num_classes,
        in_channel=config.model.in_channel,
        n_features=config.model.n_features,
        n_layers=config.model.n_layers,
    )

    experiment_manager.setup_model(model)

    # Create datasets (customize based on your data structure)
    if config.data.name == "RARP":
        # Example for RARP dataset
        train_dataset = RARPDataset(
            root=config.data.data_root,
            split='train',
            video_list=config.data.train_split,
        )

        val_dataset = RARPDataset(
            root=config.data.data_root,
            split='val',
            video_list=config.data.val_split,
        )

        test_dataset = RARPDataset(
            root=config.data.data_root,
            split='test',
            video_list=config.data.test_split,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        experiment_manager.setup_data(train_loader, val_loader, test_loader)

    # Setup optimizer and criterion
    experiment_manager.setup_optimizer()

    # Setup loss criterion (customize based on your loss configuration)
    from losses.focal_tmse import ActionSegmentationLoss

    criterion = ActionSegmentationLoss(
        ce=config.loss.ce,
        focal=config.loss.focal,
        tmse=config.loss.tmse,
        gstmse=config.loss.gstmse,
        ce_weight=config.loss.ce_weight,
        focal_weight=config.loss.focal_weight,
        tmse_weight=config.loss.tmse_weight,
        gstmse_weight=config.loss.gstmse_weight,
    )

    experiment_manager.setup_criterion(criterion)


def main():
    """Main function"""
    args = parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = ExperimentConfig.from_yaml(args.config)

    # Override config with command line arguments
    if args.seed is not None:
        config.seed = args.seed
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    print(f"\nExperiment: {config.experiment_name}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {config.output_dir}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Create experiment manager
    experiment_manager = ExperimentManager(config)

    # Setup model and data
    print("\nSetting up model and data...")
    setup_model_and_data(config, experiment_manager)

    # Create visualizers
    viz_dir = os.path.join(config.output_dir, config.experiment_name, "visualizations")
    training_viz = TrainingVisualizer(save_dir=viz_dir)
    results_viz = ResultsVisualizer(save_dir=viz_dir)

    # Create report generator
    report_dir = os.path.join(config.output_dir, config.experiment_name, "reports")
    report_gen = ReportGenerator(save_dir=report_dir, experiment_name=config.experiment_name)

    # Run experiment based on mode
    if args.mode in ['train', 'full']:
        print("\n" + "=" * 80)
        print("TRAINING")
        print("=" * 80)

        # Resume from checkpoint if provided
        if args.checkpoint:
            print(f"Resuming from checkpoint: {args.checkpoint}")
            experiment_manager.trainer.resume_training(args.checkpoint)

        # Train
        history = experiment_manager.train()

        # Visualize training
        if config.visualization.generate_plots:
            print("\nGenerating training visualizations...")
            training_viz.plot_training_curves(history)

    if args.mode in ['test', 'eval', 'full']:
        print("\n" + "=" * 80)
        print("TESTING")
        print("=" * 80)

        # Test
        predictions = experiment_manager.test(load_best=True)

        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)

        # Evaluate
        metrics = experiment_manager.evaluate(predictions)

        # Visualize results
        if config.visualization.generate_plots:
            print("\nGenerating result visualizations...")

            # Confusion matrix
            if config.visualization.plot_confusion_matrix:
                results_viz.plot_confusion_matrix(
                    predictions['predictions'],
                    predictions['labels'],
                    num_classes=config.model.num_classes,
                    normalize=True
                )

            # Sample predictions
            if config.visualization.plot_predictions and len(predictions['predictions']) > 0:
                sample_idx = 0
                results_viz.plot_prediction_timeline(
                    predictions['predictions'][sample_idx],
                    predictions['labels'][sample_idx],
                    title=f"Sample Prediction {sample_idx}"
                )
                results_viz.plot_segment_comparison(
                    predictions['predictions'][sample_idx],
                    predictions['labels'][sample_idx]
                )

    # Generate reports
    if args.mode == 'full':
        print("\n" + "=" * 80)
        print("GENERATING REPORTS")
        print("=" * 80)

        # Collect plots
        plots = {}
        viz_path = Path(viz_dir)
        if viz_path.exists():
            for plot_file in viz_path.glob("*.png"):
                plots[plot_file.stem] = str(plot_file)

        # Generate reports
        if config.reporting.generate_html_report:
            report_gen.generate_html_report(
                config=config.to_dict(),
                metrics=metrics,
                history=history,
                plots=plots
            )

        report_gen.generate_text_report(
            config=config.to_dict(),
            metrics=metrics,
            history=history
        )

        report_gen.generate_json_report(
            config=config.to_dict(),
            metrics=metrics,
            history=history
        )

    # Close logger
    experiment_manager.close()

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {config.output_dir}/{config.experiment_name}")


if __name__ == "__main__":
    main()
