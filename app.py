import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from datetime import datetime, timezone
import time

st.set_page_config(
    page_title="Timestamp Logger",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
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


def render_record_button():
    """Render a prominent record button with Space shortcut and timestamp logging."""

    def record_timestamp():
        now = datetime.now(timezone.utc)
        unix_ts = int(time.time() * 1000)
        timestamp_iso = now.isoformat()

        # Cache session_state locally - single lookup per key
        state = st.session_state
        timestamps = state.timestamps
        seq = state.sequence + 1

        # Determine if this is the first timestamp
        is_first = not timestamps

        if is_first:
            delta_ms = 0
        else:
            delta_ms = unix_ts - state.first_timestamp

        entry = {
            "seq": seq,
            "timestamp": timestamp_iso,
            "unix_ts": unix_ts,
            "delta_ms": delta_ms,
        }

        # Batch updates - write back once
        state.sequence = seq
        state.timestamps.append(entry)
        if is_first:
            state.first_timestamp = unix_ts

        # Track activity for the dashboard
        state.last_activity = time.time()
        state.data_version += 1
        # Note: Button callbacks naturally trigger script rerun after completion

    st.button(
        "RECORD TIMESTAMP",
        type="primary",
        width="stretch",
        shortcut="Space",
        on_click=record_timestamp,
    )

    st.caption("Press SPACE")

    if st.session_state.timestamps:
        last = st.session_state.timestamps[-1]
        st.metric(
            label=f"#{last['seq']} Recorded",
            value=f"+{last['delta_ms']} ms",
            help=f"ISO: {last['timestamp']}\nUnix: {last['unix_ts']}",
        )


def _format_duration(seconds):
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    elif seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


@st.cache_data(show_spinner=False)
def _calculate_timing_stats(df, data_version):
    count = len(df)
    if count >= 2:
        duration = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
        intervals = df["timestamp"].diff().dt.total_seconds().dropna()
        mean_interval = intervals.mean()
        std_interval = intervals.std()
    else:
        duration = mean_interval = std_interval = 0
    return count, duration, mean_interval, std_interval


@st.cache_data(show_spinner=False)
def _get_dataframe(timestamps_tuple, _version):
    """Cache DataFrame creation. _version ensures cache invalidation when data changes."""
    if not timestamps_tuple:
        return pd.DataFrame()
    df = pd.DataFrame(list(timestamps_tuple))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _render_metrics_row(count, duration, mean_interval, std_interval):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", f"{count:,}")
    with col2:
        st.metric("Session Duration", _format_duration(duration))
    with col3:
        st.metric("Mean Interval", _format_duration(mean_interval))
    with col4:
        st.metric("Std Deviation", _format_duration(std_interval))


@st.cache_data(show_spinner=False)
def _get_recent_events_display(df_json, _version):
    """Cache the recent events display DataFrame creation."""
    df = pd.read_json(df_json, orient="split")
    recent = df.tail(5).iloc[::-1]
    if "timestamp" not in recent.columns:
        return recent
    recent["Time"] = recent["timestamp"].dt.strftime("%H:%M:%S")
    recent["Relative"] = recent["timestamp"].apply(
        lambda x: f"{(pd.Timestamp.now(tz='UTC') - x).total_seconds():.1f}s ago"
    )
    return recent[["Time", "Relative"]]


def _render_recent_events(df):
    st.markdown("---")
    st.subheader("Recent Events")
    df_json = df.to_json(orient="split")
    recent = _get_recent_events_display(df_json, st.session_state.data_version)
    if "timestamp" in df.columns and "Time" in recent.columns:
        recent = recent.copy()
        recent["Relative"] = (
            df.tail(5)
            .iloc[::-1]["timestamp"]
            .apply(
                lambda x: f"{(pd.Timestamp.now(tz='UTC') - x).total_seconds():.1f}s ago"
            )
        )
    display_cols = ["Time", "Relative"]
    if "event_type" in recent.columns:
        display_cols.insert(1, "event_type")
    st.dataframe(recent[display_cols], width="stretch", hide_index=True, height=200)


def render_stats(timestamps_df):
    if timestamps_df.empty or len(timestamps_df) < 1:
        st.info("No timing data available yet. Add some timestamps to see stats.")
        return
    count, duration, mean_interval, std_interval = _calculate_timing_stats(
        timestamps_df, st.session_state.data_version
    )
    _render_metrics_row(count, duration, mean_interval, std_interval)
    _render_recent_events(timestamps_df)


@st.cache_data(show_spinner=False)
def _calc_regression(df, data_version):
    if len(df) < 2:
        return None, "Timestamp Delta Over Time"
    slope, intercept, r_value, _, _ = stats.linregress(df["seq"], df["delta_ms"])
    r_squared = r_value**2
    x_line = np.linspace(df["seq"].min(), df["seq"].max(), 100)
    y_line = slope * x_line + intercept
    return (
        x_line,
        y_line,
        r_squared,
    ), f"Timestamp Delta Over Time (R² = {r_squared:.3f})"


def _apply_figure_layout(fig, title):
    fig.update_layout(
        title=title,
        xaxis_title="Sequence",
        yaxis_title="Delta from Start (ms)",
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
    )
    return fig


@st.cache_data(show_spinner=False, max_entries=10)
def create_scatterplot(df, _version):
    if df.empty:
        return go.Figure()
    fig = go.Figure()
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
def live_dashboard(df=None):
    """
    Dashboard fragment - updates every second to show elapsed time.
    Button clicks trigger full script rerun for instant updates.
    """
    if df is None:
        timestamps = st.session_state.timestamps
        if not timestamps:
            st.info(
                "📊 No timestamps recorded yet. Press SPACE or click RECORD to start collecting data."
            )
            return
        df = pd.DataFrame(timestamps)

    # Calculate metrics
    count = len(df)

    if count >= 2:
        duration_sec = df["delta_ms"].iloc[-1] / 1000.0

        # Calculate average interval in milliseconds
        intervals = df["delta_ms"].diff().dropna()
        avg_interval_ms = intervals.mean()
    else:
        duration_sec = 0
        avg_interval_ms = 0

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Count", f"{count}")
    with col2:
        st.metric("Duration (s)", f"{duration_sec:.2f}")
    with col3:
        st.metric("Avg Interval (ms)", f"{avg_interval_ms:.1f}")

    # Create and display scatterplot
    fig = create_scatterplot(df, st.session_state.data_version)
    st.plotly_chart(fig, width="stretch")


def render_export_section(df):
    """Render CSV export button with metadata header."""
    filename = "timestamps.csv"

    # Only regenerate CSV when data actually changes, not on every rerun
    current_version = st.session_state.data_version
    cached_version = st.session_state.get("csv_version", -1)

    if cached_version != current_version or "csv_content" not in st.session_state:
        # Create CSV with metadata header as comments
        lines = []
        lines.append("# Timestamp Export")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append(f"# Total Events: {len(df)}")
        if len(df) > 0:
            lines.append(f"# First: {df['timestamp'].iloc[0]}")
            lines.append(f"# Last: {df['timestamp'].iloc[-1]}")
        lines.append("#")

        # Add data
        csv_content = df.to_csv(index=False)
        full_content = "\n".join(lines) + "\n" + csv_content

        st.session_state.csv_content = full_content
        st.session_state.csv_version = current_version
    else:
        full_content = st.session_state.csv_content

    st.download_button(
        label="📥 Export CSV",
        data=full_content,
        file_name=filename,
        mime="text/csv",
        key="export_csv",
        width="stretch",
    )


# Title
st.title("◉ Timestamp Generator")
st.markdown("Press SPACE to record timestamps")

# Create DataFrame once at the top (cached)
version = st.session_state.data_version
df = _get_dataframe(tuple(st.session_state.timestamps), version)

# Two-column layout: button left, stats right
col1, col2 = st.columns([1, 2])
with col1:
    render_record_button()
    if not df.empty:
        render_export_section(df)
with col2:
    if not df.empty:
        render_stats(df)

# Full-width live dashboard
st.markdown("---")
live_dashboard(df)
