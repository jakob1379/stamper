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


def render_record_button():
    """Render a prominent record button with Space shortcut and timestamp logging."""

    def record_timestamp():
        now = datetime.now(timezone.utc)
        unix_ts = int(time.time() * 1000)
        timestamp_iso = now.isoformat()

        if not st.session_state.timestamps:
            delta_ms = 0
            st.session_state.first_timestamp = unix_ts
        else:
            delta_ms = unix_ts - st.session_state.first_timestamp

        st.session_state.sequence += 1

        entry = {
            "seq": st.session_state.sequence,
            "timestamp": timestamp_iso,
            "unix_ts": unix_ts,
            "delta_ms": delta_ms,
        }
        st.session_state.timestamps.append(entry)

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


def _calculate_timing_stats(df):
    count = len(df)
    if count >= 2:
        duration = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
        intervals = df["timestamp"].diff().dt.total_seconds().dropna()
        mean_interval = intervals.mean()
        std_interval = intervals.std()
    else:
        duration = mean_interval = std_interval = 0
    return count, duration, mean_interval, std_interval


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


def _render_recent_events(df):
    st.markdown("---")
    st.subheader("Recent Events")
    recent = df.tail(5).copy().iloc[::-1]
    if "timestamp" not in recent.columns:
        st.write(recent)
        return
    recent["Time"] = recent["timestamp"].dt.strftime("%H:%M:%S")
    recent["Relative"] = recent["timestamp"].apply(
        lambda x: f"{(pd.Timestamp.now(tz='UTC') - x).total_seconds():.1f}s ago"
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
        timestamps_df
    )
    _render_metrics_row(count, duration, mean_interval, std_interval)
    _render_recent_events(timestamps_df)


def _calc_regression(df):
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


def create_scatterplot(df):
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
    reg_data, title = _calc_regression(df)
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


@st.fragment(run_every="0.5s")
def live_dashboard():
    """
    Live auto-refresh dashboard for timestamp visualization.
    Renders the scatterplot and updates automatically every 500ms.
    Shows metrics: count, elapsed time, avg interval.
    """
    timestamps = st.session_state.timestamps

    if not timestamps:
        st.info(
            "📊 No timestamps recorded yet. Press SPACE or click RECORD to start collecting data."
        )
        return

    # Convert to DataFrame
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
    fig = create_scatterplot(df)
    st.plotly_chart(fig, width="stretch")


def render_export_section(df):
    """Render CSV export button with metadata header."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"timestamps_{timestamp_str}.csv"

    # Create CSV with metadata header as comments
    lines = []
    lines.append(f"# Timestamp Export")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append(f"# Total Events: {len(df)}")
    if len(df) > 0:
        lines.append(f"# First: {df['timestamp'].iloc[0]}")
        lines.append(f"# Last: {df['timestamp'].iloc[-1]}")
    lines.append(f"#")

    # Add data
    csv_content = df.to_csv(index=False)
    full_content = "\n".join(lines) + "\n" + csv_content

    st.download_button(
        label="📥 Export CSV",
        data=full_content,
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )


# Title
st.title("◉ Timestamp Generator")
st.markdown("Press SPACE to record timestamps")

# Two-column layout: button left, stats right
col1, col2 = st.columns([1, 2])
with col1:
    render_record_button()
    if st.session_state.timestamps:
        df = pd.DataFrame(st.session_state.timestamps)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        render_export_section(df)
with col2:
    if st.session_state.timestamps:
        df = pd.DataFrame(st.session_state.timestamps)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        render_stats(df)

# Full-width live dashboard
st.markdown("---")
live_dashboard()
