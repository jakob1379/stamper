# Timestamp Logger

A simple timestamp recording tool with live visualization. Built with Streamlit.

## What It Does

Press **SPACE** (or click the button) to record timestamps. The app tracks:
- Sequence numbers for each timestamp
- Time delta from the first recorded timestamp
- Statistics and visualizations of your timing data

Perfect for:
- Timing events during presentations
- Logging reaction times
- Recording interval data for analysis
- Debugging timing issues in applications

## Features

### ⚡ Instant Recording
- **SPACE** keyboard shortcut for rapid timestamp capture
- Visual feedback with sequence counter and delta display
- Session-based storage (resets on page reload)

### 📊 Live Dashboard
- Auto-refreshing scatterplot of timestamps
- Linear regression overlay with R² value
- Real-time metrics: count, duration, average interval

### 📈 Statistics Panel
- Total events and session duration
- Mean interval between timestamps
- Standard deviation of intervals
- Recent events table with relative timestamps

### 📥 CSV Export
- One-click export with metadata header
- Includes: sequence, timestamp (ISO 8601), unix_ms, delta_ms
- Automatic filename with timestamp

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Space` | Record timestamp |

## Running Locally

```bash
# Using uv (recommended)
uv run streamlit run app.py

# Or with pip
pip install streamlit pandas plotly scipy
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Data Format

Exported CSV contains:

```csv
# Timestamp Export
# Generated: 2024-01-15T10:30:00
# Total Events: 42
# First: 2024-01-15T10:00:00
# Last: 2024-01-15T10:15:30
#
seq,timestamp,unix_ts,delta_ms
1,2024-01-15T10:00:00.000000+00:00,1705312800000,0
2,2024-01-15T10:00:05.234000+00:00,1705312805234,5234
...
```

## Architecture

```
app.py
├── render_record_button()    # Primary record button with SPACE shortcut
├── render_stats()            # Statistics display and recent events table
├── render_export_section()   # CSV download with metadata header
├── create_scatterplot()      # Plotly scatterplot with regression line
└── live_dashboard()          # Auto-refreshing metrics and visualization
```

## Dependencies

- streamlit >= 1.54.0
- pandas >= 2.0.0
- plotly >= 5.0.0
- scipy >= 1.10.0

## License

MIT License
