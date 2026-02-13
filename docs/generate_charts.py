#!/usr/bin/env python3
"""
Generate interactive Plotly charts for stats performance documentation.
Run this script to create HTML files that can be embedded in Sphinx docs.

Usage:
    python generate_stats_charts.py [--output-dir DIR] [--include-js CDN|INLINE|TRUE]
"""

import argparse
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.offline import plot


def load_chart_data(json_path: Path) -> dict[str, Any]:
    """Load chart configuration from JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON data file not found: {json_path}")

    with open(json_path) as f:
        return json.load(f)


def create_chart(chart_data: list[dict[str, Any]], chart_layout: dict[str, Any]) -> go.Figure:
    """Create a Plotly figure from data and layout configuration."""
    fig = go.Figure()

    for trace in chart_data:
        trace_type = trace.get("type", "scatter")

        if trace_type == "scatter":
            fig.add_trace(
                go.Scatter(
                    x=trace["x"],
                    y=trace["y"],
                    name=trace["name"],
                    mode=trace.get("mode", "lines+markers"),
                    line=trace.get("line", {}),
                    marker=trace.get("marker", {}),
                ),
            )
        elif trace_type == "bar":
            fig.add_trace(
                go.Bar(
                    x=trace["x"],
                    y=trace["y"],
                    name=trace["name"],
                    marker=trace.get("marker", {}),
                    opacity=trace.get("marker", {}).get("opacity", 0.8),
                ),
            )

    # Enhanced layout configuration
    layout_config = {
        **chart_layout,
        "hovermode": "x unified",
        "showlegend": True,
        "font": {"family": "Arial, sans-serif", "size": 12},
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    }

    fig.update_layout(**layout_config)
    return fig


def generate_chart_html(data_file: Path, output_dir: Path, chart_key: str, include_plotlyjs: str = "cdn") -> str:
    """Generate HTML file for a specific chart."""
    chart_data = load_chart_data(data_file)
    try:
        chart_config = chart_data["charts"][chart_key]

        fig = create_chart(chart_config["data"], chart_config["layout"])

        # Generate HTML
        html_filename = f"{chart_key}.html"
        html_path = output_dir / html_filename

        # Enhanced Plotly config for documentation
        plot_config = {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "pan2d",
                "lasso2d",
                "select2d",
                "autoScale2d",
                "toggleHover",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
            ],
            "responsive": True,
        }

        plot(
            fig,
            filename=str(html_path),
            auto_open=False,
            include_plotlyjs=include_plotlyjs,  # type: ignore
            config=plot_config,
        )

        return html_filename

    except KeyError:
        raise KeyError(
            f"Chart key '{chart_key}' not found in data file. "
            f"Available keys: {list(chart_data.get('charts', {}).keys())}",
        )
    except Exception as e:
        raise Exception(f"Failed to generate chart '{chart_key}': {e}")


def main() -> int:
    """Generate all chart HTML files."""
    parser = argparse.ArgumentParser(description="Generate interactive charts for Sphinx documentation")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("docs/source/_static/stats_perf.json"),
        help="Path to JSON data file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/source/_static/charts"),
        help="Output directory for HTML charts",
    )
    parser.add_argument(
        "--include-js",
        choices=["cdn", "inline", "true"],
        default="cdn",
        help="How to include Plotly.js in generated HTML",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output except for errors")

    args = parser.parse_args()

    def log(message):
        """Log message unless in quiet mode."""
        if not args.quiet:
            print(message)

    # Validate input file
    if not args.data_file.exists():
        print(f"❌ Data file not found: {args.data_file}")
        print("   Please ensure the JSON file exists at the specified path.")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data to get available chart keys
    try:
        data = load_chart_data(args.data_file)
        chart_keys = list(data.get("charts", {}).keys())
    except Exception as e:
        print(f"❌ Failed to load data file: {e}")
        return 1

    if not chart_keys:
        print(f"❌ No charts found in data file: {args.data_file}")
        return 1

    log(f"📊 Generating {len(chart_keys)} interactive charts...")
    log(f"   Data file: {args.data_file}")
    log(f"   Output dir: {args.output_dir}")
    log(f"   Plotly.js: {args.include_js}")
    if not args.quiet:
        print()

    success_count = 0
    for chart_key in chart_keys:
        try:
            html_file = generate_chart_html(args.data_file, args.output_dir, chart_key, args.include_js)
            log(f"✅ Generated: {html_file}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed: {chart_key} - {e}")

    if success_count > 0:
        log(f"\n📈 Successfully generated {success_count}/{len(chart_keys)} charts")
        log(f"   Charts saved to: {args.output_dir}")

        if not args.quiet:
            print("\n📝 Include in your .md files using:")
            print("   ```{raw} html")
            print("   :file: _static/charts/CHART_NAME.html")
            print("   ```")
    else:
        print("\n❌ No charts were generated successfully")

    return 0 if success_count == len(chart_keys) else 1


if __name__ == "__main__":
    exit(main())
