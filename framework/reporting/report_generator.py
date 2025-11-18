"""
Report generation utilities
Creates HTML and text reports for experiments
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import base64
from io import BytesIO


class ReportGenerator:
    """
    Generates comprehensive experiment reports in HTML and text formats
    """

    def __init__(self, save_dir: str, experiment_name: str):
        """
        Initialize report generator

        Args:
            save_dir: Directory to save reports
            experiment_name: Name of the experiment
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_html_report(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        history: Optional[Dict[str, List[float]]] = None,
        plots: Optional[Dict[str, str]] = None,
        save_name: str = "experiment_report.html"
    ) -> str:
        """
        Generate HTML report

        Args:
            config: Experiment configuration
            metrics: Final metrics dictionary
            history: Training history
            plots: Dictionary of {plot_name: plot_path}
            save_name: Name of the report file

        Returns:
            Path to saved report
        """
        html_content = self._create_html_template(config, metrics, history, plots)

        report_path = self.save_dir / save_name
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML report saved to: {report_path}")
        return str(report_path)

    def generate_text_report(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        history: Optional[Dict[str, List[float]]] = None,
        save_name: str = "experiment_report.txt"
    ) -> str:
        """
        Generate text report

        Args:
            config: Experiment configuration
            metrics: Final metrics dictionary
            history: Training history
            save_name: Name of the report file

        Returns:
            Path to saved report
        """
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append(f"EXPERIMENT REPORT: {self.experiment_name}")
        report_lines.append(f"Generated: {self.timestamp}")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Configuration
        report_lines.append("-" * 80)
        report_lines.append("CONFIGURATION")
        report_lines.append("-" * 80)
        for section, params in config.items():
            if isinstance(params, dict):
                report_lines.append(f"\n{section.upper()}:")
                for key, value in params.items():
                    report_lines.append(f"  {key}: {value}")
            else:
                report_lines.append(f"{section}: {params}")
        report_lines.append("")

        # Metrics
        report_lines.append("-" * 80)
        report_lines.append("FINAL METRICS")
        report_lines.append("-" * 80)
        for metric, value in metrics.items():
            report_lines.append(f"{metric:30s}: {value:.6f}")
        report_lines.append("")

        # Training summary
        if history is not None:
            report_lines.append("-" * 80)
            report_lines.append("TRAINING SUMMARY")
            report_lines.append("-" * 80)

            if 'train_loss' in history and len(history['train_loss']) > 0:
                report_lines.append(f"Total Epochs: {len(history['train_loss'])}")
                report_lines.append(f"Best Training Loss: {min(history['train_loss']):.6f}")
                report_lines.append(f"Final Training Loss: {history['train_loss'][-1]:.6f}")

            if 'val_loss' in history and len(history['val_loss']) > 0:
                report_lines.append(f"Best Validation Loss: {min(history['val_loss']):.6f}")
                report_lines.append(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")

            if 'train_acc' in history and len(history['train_acc']) > 0:
                report_lines.append(f"Best Training Accuracy: {max(history['train_acc']):.6f}")
                report_lines.append(f"Final Training Accuracy: {history['train_acc'][-1]:.6f}")

            if 'val_acc' in history and len(history['val_acc']) > 0:
                report_lines.append(f"Best Validation Accuracy: {max(history['val_acc']):.6f}")
                report_lines.append(f"Final Validation Accuracy: {history['val_acc'][-1]:.6f}")

            report_lines.append("")

        # Footer
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        # Save report
        report_path = self.save_dir / save_name
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"Text report saved to: {report_path}")
        return str(report_path)

    def generate_json_report(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        history: Optional[Dict[str, List[float]]] = None,
        save_name: str = "experiment_report.json"
    ) -> str:
        """
        Generate JSON report

        Args:
            config: Experiment configuration
            metrics: Final metrics dictionary
            history: Training history
            save_name: Name of the report file

        Returns:
            Path to saved report
        """
        report_data = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'config': config,
            'metrics': metrics,
            'history': history if history else {},
        }

        report_path = self.save_dir / save_name
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

        print(f"JSON report saved to: {report_path}")
        return str(report_path)

    def _create_html_template(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        history: Optional[Dict[str, List[float]]],
        plots: Optional[Dict[str, str]]
    ) -> str:
        """Create HTML report template"""

        # Convert plots to base64 if they exist
        plot_html = ""
        if plots:
            plot_html = '<div class="section">\n<h2>Visualizations</h2>\n'
            for plot_name, plot_path in plots.items():
                if os.path.exists(plot_path):
                    with open(plot_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    plot_html += f'''
                    <div class="plot">
                        <h3>{plot_name}</h3>
                        <img src="data:image/png;base64,{img_data}" alt="{plot_name}">
                    </div>
                    '''
            plot_html += '</div>\n'

        # Create metrics table
        metrics_rows = ""
        for metric, value in metrics.items():
            metrics_rows += f"<tr><td>{metric}</td><td>{value:.6f}</td></tr>\n"

        # Create config table
        config_html = ""
        for section, params in config.items():
            if isinstance(params, dict):
                config_html += f'<h3>{section.upper()}</h3>\n<table class="config-table">\n'
                for key, value in params.items():
                    config_html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
                config_html += "</table>\n"

        # Create training summary
        training_summary = ""
        if history:
            training_summary = '<div class="section">\n<h2>Training Summary</h2>\n<table class="metrics-table">\n'

            if 'train_loss' in history and len(history['train_loss']) > 0:
                training_summary += f"<tr><td>Total Epochs</td><td>{len(history['train_loss'])}</td></tr>\n"
                training_summary += f"<tr><td>Best Training Loss</td><td>{min(history['train_loss']):.6f}</td></tr>\n"
                training_summary += f"<tr><td>Final Training Loss</td><td>{history['train_loss'][-1]:.6f}</td></tr>\n"

            if 'val_loss' in history and len(history['val_loss']) > 0:
                training_summary += f"<tr><td>Best Validation Loss</td><td>{min(history['val_loss']):.6f}</td></tr>\n"
                training_summary += f"<tr><td>Final Validation Loss</td><td>{history['val_loss'][-1]:.6f}</td></tr>\n"

            if 'train_acc' in history and len(history['train_acc']) > 0:
                training_summary += f"<tr><td>Best Training Accuracy</td><td>{max(history['train_acc']):.6f}</td></tr>\n"
                training_summary += f"<tr><td>Final Training Accuracy</td><td>{history['train_acc'][-1]:.6f}</td></tr>\n"

            if 'val_acc' in history and len(history['val_acc']) > 0:
                training_summary += f"<tr><td>Best Validation Accuracy</td><td>{max(history['val_acc']):.6f}</td></tr>\n"
                training_summary += f"<tr><td>Final Validation Accuracy</td><td>{history['val_acc'][-1]:.6f}</td></tr>\n"

            training_summary += "</table>\n</div>\n"

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Report - {self.experiment_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}

        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
            margin-bottom: 10px;
        }}

        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}

        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 10px;
        }}

        .section {{
            margin-bottom: 30px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}

        .metrics-table th,
        .metrics-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}

        .metrics-table th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}

        .metrics-table tr:hover {{
            background-color: #f8f9fa;
        }}

        .config-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}

        .config-table td:first-child {{
            font-weight: bold;
            width: 40%;
            color: #555;
        }}

        .plot {{
            margin: 20px 0;
            text-align: center;
        }}

        .plot img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            background: white;
        }}

        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}

        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
        }}

        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Experiment Report</h1>
            <h2>{self.experiment_name}</h2>
            <div class="timestamp">Generated: {self.timestamp}</div>
        </div>

        <div class="section">
            <h2>Final Metrics</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {metrics_rows}
                </tbody>
            </table>
        </div>

        {training_summary}

        <div class="section">
            <h2>Configuration</h2>
            {config_html}
        </div>

        {plot_html}

        <div class="footer">
            <p>Report generated by Modular Training Framework</p>
        </div>
    </div>
</body>
</html>
"""

        return html_template

    def generate_comparison_report(
        self,
        experiments: Dict[str, Dict[str, Any]],
        save_name: str = "comparison_report.html"
    ) -> str:
        """
        Generate comparison report for multiple experiments

        Args:
            experiments: Dictionary of {exp_name: {metrics, config, ...}}
            save_name: Name of the report file

        Returns:
            Path to saved report
        """
        # Create comparison table
        exp_names = list(experiments.keys())
        all_metrics = set()

        for exp_data in experiments.values():
            if 'metrics' in exp_data:
                all_metrics.update(exp_data['metrics'].keys())

        all_metrics = sorted(list(all_metrics))

        # Build comparison table HTML
        table_html = '<table class="metrics-table">\n<thead>\n<tr>\n<th>Metric</th>\n'
        for exp_name in exp_names:
            table_html += f'<th>{exp_name}</th>\n'
        table_html += '</tr>\n</thead>\n<tbody>\n'

        for metric in all_metrics:
            table_html += f'<tr>\n<td><strong>{metric}</strong></td>\n'
            for exp_name in exp_names:
                value = experiments[exp_name].get('metrics', {}).get(metric, 'N/A')
                if isinstance(value, float):
                    table_html += f'<td>{value:.6f}</td>\n'
                else:
                    table_html += f'<td>{value}</td>\n'
            table_html += '</tr>\n'

        table_html += '</tbody>\n</table>\n'

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Experiment Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        .metrics-table th {{ background-color: #3498db; color: white; }}
        .metrics-table tr:hover {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Experiment Comparison Report</h1>
        <p>Generated: {self.timestamp}</p>
        <h2>Metrics Comparison</h2>
        {table_html}
    </div>
</body>
</html>
"""

        report_path = self.save_dir / save_name
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Comparison report saved to: {report_path}")
        return str(report_path)
