from __future__ import annotations

import time
from datetime import datetime, timezone
from io import StringIO
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="Timestamp Logger",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

if "sequence" not in st.session_state:
    st.session_state.sequence = 0

if "last_activity" not in st.session_state:
    st.session_state.last_activity = 0

if "data_version" not in st.session_state:
    st.session_state.data_version = 0

if "first_timestamp" not in st.session_state:
    st.session_state.first_timestamp = 0


def _record_timestamp() -> None:
    """Callback function to record a timestamp."""
    now: datetime = datetime.now(timezone.utc)
    unix_ts: int = int(time.time() * 1000)
    timestamp_iso: str = now.isoformat()

    state: Any = st.session_state
    timestamps: list[dict[str, Any]] = state.timestamps
    seq: int = state.sequence + 1
    is_first: bool = not timestamps

    delta_ms: int = 0 if is_first else unix_ts - state.first_timestamp

    entry: dict[str, int | str] = {
        "seq": seq,
        "timestamp": timestamp_iso,
        "unix_ts": unix_ts,
        "delta_ms": delta_ms,
    }

    state.sequence = seq
    state.timestamps.append(entry)
    if is_first:
        state.first_timestamp = unix_ts

    state.last_activity = time.time()
    state.data_version += 1


def render_record_button() -> None:
    """Render a prominent record button with Space shortcut and timestamp logging."""
    st.button(
        "RECORD TIMESTAMP",
        type="primary",
        width="stretch",
        shortcut="Space",
        on_click=_record_timestamp,
    )

    st.caption("Press SPACE")


def _render_consolidated_metric(df: pd.DataFrame) -> None:
    """Render a single metric showing mean interval and std deviation."""
    if len(df) < 2:
        st.metric(
            "Interval Stats", "N/A", help="Need at least 2 timestamps", border=True
        )
        return

    intervals: pd.Series = df["delta_ms"].diff().dropna()
    mean_interval: float = float(intervals.mean())
    std_interval: float = float(intervals.std())

    st.metric(
        label="Mean Interval ± Std",
        value=f"{mean_interval:.1f} ms",
        delta=f"±{std_interval:.1f} ms",
        border=True,
    )


def _format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    elif seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


@st.cache_data(show_spinner=False)
def _calculate_timing_stats(
    df: pd.DataFrame, _data_version: int
) -> tuple[int, float, float, float]:
    count: int = len(df)
    duration: float = 0.0
    mean_interval: float = 0.0
    std_interval: float = 0.0

    if count >= 2:
        duration = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
        timestamps: pd.Series = df["timestamp"]
        deltas = cast(pd.Series, timestamps.diff())
        intervals: pd.Series = deltas.dt.total_seconds().dropna()
        mean_interval = float(intervals.mean())
        std_interval = float(intervals.std())

    return count, duration, mean_interval, std_interval


@st.cache_data(show_spinner=False)
def _get_dataframe(
    timestamps_tuple: tuple[dict[str, Any], ...], _version: int
) -> pd.DataFrame:
    """Cache DataFrame creation. _version ensures cache invalidation when data changes."""
    if not timestamps_tuple:
        return pd.DataFrame()
    df: pd.DataFrame = pd.DataFrame(list(timestamps_tuple))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _render_events_editor(df: pd.DataFrame) -> None:
    """Render an editable dataframe with proper column types."""
    if df.empty:
        return

    column_config: dict[str, Any] = {
        "seq": st.column_config.NumberColumn(
            "#",
            help="Sequence number",
            format="%d",
            disabled=True,
        ),
        "timestamp": st.column_config.DatetimeColumn(
            "Timestamp",
            help="ISO timestamp",
            format="YYYY-MM-DD HH:mm:ss.SSS",
        ),
        "unix_ts": st.column_config.NumberColumn(
            "Unix (ms)",
            help="Unix timestamp in milliseconds",
            format="%d",
        ),
        "delta_ms": st.column_config.NumberColumn(
            "Delta (ms)",
            help="Milliseconds from first timestamp",
            format="%d",
        ),
    }

    st.data_editor(
        df,
        column_config=column_config,
        hide_index=True,
        width="stretch",
        height=250,
        num_rows="fixed",
        key="events_editor",
    )


@st.cache_data(show_spinner=False)
def _get_recent_events_display(df_json: str, _version: int) -> pd.DataFrame:
    """Cache the recent events display DataFrame creation."""
    df: pd.DataFrame = pd.read_json(StringIO(df_json), orient="split")
    recent: pd.DataFrame = df.tail(5).iloc[::-1]
    if "timestamp" not in recent.columns:
        return recent
    recent["Time"] = recent["timestamp"].dt.strftime("%H:%M:%S")
    recent["Relative"] = recent["timestamp"].apply(
        lambda x: (
            f"{(pd.Timestamp.now(tz='UTC') - (x.tz_convert('UTC') if x.tzinfo else x.tz_localize('UTC'))).total_seconds():.1f}s ago"
        )
    )
    return recent[["Time", "Relative"]]


def _render_recent_events(df: pd.DataFrame) -> None:
    st.markdown("---")
    st.subheader("Recent Events")
    df_json: str = df.to_json(orient="split")
    recent: pd.DataFrame = _get_recent_events_display(
        df_json, st.session_state.data_version
    )
    if "timestamp" in df.columns and "Time" in recent.columns:
        recent = recent.copy()
        recent["Relative"] = (
            df.tail(5)
            .iloc[::-1]["timestamp"]
            .apply(
                lambda x: (
                    f"{(pd.Timestamp.now(tz='UTC') - (x.tz_convert('UTC') if x.tzinfo else x.tz_localize('UTC'))).total_seconds():.1f}s ago"
                )
            )
        )
    display_cols: list[str] = ["Time", "Relative"]
    if "event_type" in recent.columns:
        display_cols.insert(1, "event_type")
    st.dataframe(recent[display_cols], width="stretch", hide_index=True, height=200)


def render_stats(timestamps_df: pd.DataFrame) -> None:
    if timestamps_df.empty or len(timestamps_df) < 1:
        st.info("No timing data available yet. Add some timestamps to see stats.")
        return
    _render_events_editor(timestamps_df)


@st.cache_data(show_spinner=False)
def _calc_regression(
    df: pd.DataFrame, _data_version: int
) -> tuple[tuple[np.ndarray, np.ndarray, float] | None, str]:
    if len(df) < 2:
        return None, "Timestamp Delta Over Time"
    slope: float
    intercept: float
    r_value: float
    slope, intercept, r_value, _, _ = stats.linregress(df["seq"], df["delta_ms"])
    r_squared: float = r_value**2
    x_line: np.ndarray = np.linspace(df["seq"].min(), df["seq"].max(), 100)
    y_line: np.ndarray = slope * x_line + intercept
    return (
        x_line,
        y_line,
        r_squared,
    ), f"Timestamp Delta Over Time"


def _apply_figure_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title="Sequence",
        yaxis_title="Delta from Start (ms)",
        template="plotly_dark",
        height=850,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
    )
    return fig


@st.cache_data(show_spinner=False, max_entries=10)
def create_scatterplot(df: pd.DataFrame, _version: int) -> go.Figure:
    if df.empty:
        return go.Figure()
    fig: go.Figure = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["seq"],
            y=df["delta_ms"],
            mode="markers+lines",
            marker=dict(size=10, color="#00CC96"),
            line=dict(color="rgba(100, 100, 100, 0.3)", width=1),
            text=df["timestamp"],
            hovertemplate="<b>Sequence:</b> %{x}<br><b>Delta:</b> %{y} ms<br><b>Time:</b> %{text}<extra></extra>",
            name="Timestamps",
        )
    )
    reg_data: tuple[np.ndarray, np.ndarray, float] | None
    title: str
    reg_data, title = _calc_regression(df, _version)
    if reg_data:
        x_line, y_line, r_squared = reg_data
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name=f"Regression (R²={r_squared:.3f})",
            )
        )
    return _apply_figure_layout(fig, title)


@st.fragment(run_every="1s")
def live_dashboard_plot(df: pd.DataFrame) -> None:
    """Render just the plot in the right column - updates every second."""
    # Create and display scatterplot
    fig: go.Figure = create_scatterplot(df, st.session_state.data_version)
    st.plotly_chart(fig, width="stretch", height="stretch")


def render_export_section(df: pd.DataFrame) -> None:
    """Render CSV export button with raw data."""
    filename: str = "timestamps.csv"

    # Only regenerate CSV when data actually changes, not on every rerun
    current_version: int = st.session_state.data_version
    cached_version: int = st.session_state.get("csv_version", -1)

    if cached_version != current_version or "csv_content" not in st.session_state:
        csv_content: str = df.to_csv(index=False)
        st.session_state.csv_content = csv_content
        st.session_state.csv_version = current_version
    else:
        csv_content: str = st.session_state.csv_content

    import hashlib

    file_hash = hashlib.md5(csv_content.encode()).hexdigest()[:8]
    st.download_button(
        label="📥 Export CSV",
        data=csv_content,
        file_name=filename,
        mime="text/csv",
        key=f"export_csv_{file_hash}",
        width="stretch",
    )


# Title
st.title("◉ Timestamp Generator")
st.markdown("Press SPACE to record timestamps")

# Create DataFrame once at the top (cached)
version: int = st.session_state.data_version
df: pd.DataFrame = _get_dataframe(tuple(st.session_state.timestamps), version)

# Two-column layout: 1/5 left for controls, 4/5 right for plot
col1, col2 = st.columns([1, 4])

with col1:
    with st.container(height="stretch", vertical_alignment="distribute"):
        render_record_button()

        if not df.empty:
            _render_consolidated_metric(df)

            st.subheader(f"All Events ({len(df)})")
            render_stats(df)

            render_export_section(df)

with col2:
    if not df.empty:
        live_dashboard_plot(df)
    else:
        st.info(
            "📊 No timestamps recorded yet. Press SPACE or click RECORD to start collecting data."
        )
