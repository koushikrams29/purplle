"""
Live Store Intelligence Dashboard

Usage:
python scripts/dashboard.py --store STORE_BLR_002
"""

import argparse
import time
from datetime import datetime, timezone

import requests

from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.console import Console


API_BASE_URL = "http://localhost:8000"


def get_metrics(store_id: str):
    """
    Fetch metrics from API.
    """
    response = requests.get(
        f"{API_BASE_URL}/stores/{store_id}/metrics",
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def get_anomalies(store_id: str):
    """
    Fetch anomalies from API.
    """
    response = requests.get(
        f"{API_BASE_URL}/stores/{store_id}/anomalies",
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def format_last_event_age(minutes: float) -> str:
    """
    Convert freshness metric into readable text.
    """

    if minutes < 1:
        return "Live"

    if minutes < 60:
        return f"{int(minutes)} min ago"

    hours = int(minutes // 60)

    return f"{hours} hr ago"


def build_dashboard(store_id: str):
    """
    Build Rich dashboard panel.
    """

    try:
        metrics = get_metrics(store_id)
        anomalies = get_anomalies(store_id)

        unique_visitors = metrics.get(
            "unique_visitors",
            0,
        )

        conversion_rate = metrics.get(
            "conversion_rate",
            0,
        )

        queue_depth = metrics.get(
            "queue_depth_current",
            0,
        )

        freshness = metrics.get(
            "data_freshness_minutes",
            0,
        )

        anomaly_list = anomalies.get(
            "anomalies",
            [],
        )

        if freshness > 10:
            live_status = "[red]STALE[/red]"
        else:
            live_status = "[green]LIVE[/green]"

        table = Table(
            title=f"Store Intelligence Dashboard - {store_id}"
        )

        table.add_column(
            "Metric",
            style="cyan",
            no_wrap=True,
        )

        table.add_column(
            "Value",
            style="white",
        )

        visitor_color = (
            "green"
            if unique_visitors > 0
            else "yellow"
        )

        conversion_color = (
            "green"
            if conversion_rate >= 20
            else "yellow"
        )

        queue_color = (
            "red"
            if queue_depth > 5
            else "green"
        )

        table.add_row(
            "Status",
            live_status,
        )

        table.add_row(
            "Unique Visitors",
            f"[{visitor_color}]{unique_visitors}[/{visitor_color}]"
        )

        table.add_row(
            "Conversion Rate",
            f"[{conversion_color}]{conversion_rate:.2f}%[/{conversion_color}]"
        )

        table.add_row(
            "Queue Depth",
            f"[{queue_color}]{queue_depth}[/{queue_color}]"
        )

        table.add_row(
            "Last Event",
            format_last_event_age(
                freshness
            )
        )

        if anomaly_list:

            anomaly_text = "\n".join(
                f"{a['severity']} - {a['type']}"
                for a in anomaly_list
            )

        else:
            anomaly_text = (
                "[green]None[/green]"
            )

        table.add_row(
            "Active Anomalies",
            anomaly_text,
        )

        table.add_row(
            "Updated",
            datetime.now(
                timezone.utc
            ).strftime("%H:%M:%S UTC")
        )

        return Panel(
            table,
            border_style="blue",
        )

    except requests.exceptions.ConnectionError:

        return Panel(
            "[red]Cannot connect to API[/red]\n\n"
            "Is FastAPI running on port 8000?",
            title="Connection Error",
            border_style="red",
        )

    except Exception as exc:

        return Panel(
            f"[red]{str(exc)}[/red]",
            title="Dashboard Error",
            border_style="red",
        )


def main():
    """
    Dashboard entrypoint.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--store",
        required=True,
        help="Store ID",
    )

    args = parser.parse_args()

    console = Console()

    with Live(
        build_dashboard(args.store),
        refresh_per_second=1,
        console=console,
    ) as live:

        while True:

            live.update(
                build_dashboard(
                    args.store
                )
            )

            time.sleep(3)


if __name__ == "__main__":
    main()