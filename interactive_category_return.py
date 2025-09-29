#!/usr/bin/env python3
"""Interactive Dash app to explore category returns by quarter or year."""

from __future__ import annotations

import pandas as pd
import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go


def get_universe(long_history: bool = False) -> dict[str, str]:
    if not long_history:
        return {
            "R1000 Value (IWD)": "IWD",
            "R1000 Growth (IWF)": "IWF",
            "R2000 (IWM)": "IWM",
            "MSCI EAFE (EFA)": "EFA",
            "US Agg Bonds (AGG)": "AGG",
        }
    return {
        "Large-Cap Value (VIVAX)": "VIVAX",
        "Large-Cap Growth (VIGRX)": "VIGRX",
        "Small-Cap (NAESX)": "NAESX",
        "EAFE (VTMGX)": "VTMGX",
        "US Agg Bonds (VBMFX)": "VBMFX",
    }


def download_adj_close(tickers: list[str], start: str | None) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        interval="1d",
        threads=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        adj = {t: df[t]["Adj Close"] for t in tickers if (t in df.columns.get_level_values(0))}
        out = pd.concat(adj, axis=1)
        out.columns = tickers
        return out
    return df["Adj Close"].to_frame(tickers[0])


def to_period_returns(adj_close: pd.DataFrame, freq: str) -> pd.DataFrame:
    period_prices = adj_close.resample(freq).last()
    period_returns = period_prices.pct_change().dropna(how="all")
    return period_returns


def tidy_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    tidy = returns_df.copy()
    tidy.index.name = "date"
    tidy = tidy.reset_index().melt(id_vars="date", var_name="label", value_name="value")
    tidy = tidy.sort_values(["date", "label"])
    return tidy


def filter_returns(returns_df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if returns_df.empty:
        return returns_df
    start_ts = pd.to_datetime(start) if start else returns_df.index.min()
    end_ts = pd.to_datetime(end) if end else returns_df.index.max()
    if start_ts > end_ts:
        return returns_df.iloc[0:0]
    return returns_df.loc[start_ts:end_ts]


def build_return_figure(returns_df: pd.DataFrame, title: str) -> go.Figure:
    tidy = tidy_returns(returns_df)
    fig = px.line(
        tidy,
        x="date",
        y="value",
        color="label",
        title=title,
        labels={"value": "Return", "date": "Date", "label": "Category"},
    )
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(template="plotly_white")
    return fig


def build_cumulative_figure(returns_df: pd.DataFrame, title: str) -> go.Figure:
    cumulative = (1 + returns_df).cumprod() - 1
    tidy = tidy_returns(cumulative)
    fig = px.line(
        tidy,
        x="date",
        y="value",
        color="label",
        title=title,
        labels={"value": "Cumulative Return", "date": "Date", "label": "Category"},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(template="plotly_white")
    return fig


def make_empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white", title=title)
    return fig


def to_date_input(ts: pd.Timestamp) -> str:
    return ts.date().isoformat()


def prepare_dataset(long_history: bool) -> dict[str, object]:
    name_map = get_universe(long_history)
    tickers = list(name_map.values())
    friendly_names = {v: k for k, v in name_map.items()}
    raw_prices = download_adj_close(tickers, start=DEFAULT_START).dropna(axis=1, how="all")
    raw_prices = raw_prices.rename(columns=friendly_names)
    if raw_prices.empty:
        min_date = pd.Timestamp(DEFAULT_START)
        max_date = pd.Timestamp.today().normalize()
        quarterly = pd.DataFrame()
        yearly = pd.DataFrame()
    else:
        min_date = raw_prices.index.min()
        max_date = raw_prices.index.max()
        quarterly = to_period_returns(raw_prices, "Q")
        yearly = to_period_returns(raw_prices, "Y")
    return {
        "raw": raw_prices,
        "quarterly": quarterly,
        "yearly": yearly,
        "min_date": min_date,
        "max_date": max_date,
    }


DEFAULT_START = "1985-01-01"
DATASETS = {
    False: prepare_dataset(False),
    True: prepare_dataset(True),
}
GLOBAL_MIN_DATE = min(dataset["min_date"] for dataset in DATASETS.values())
GLOBAL_MAX_DATE = max(dataset["max_date"] for dataset in DATASETS.values())
DEFAULT_START_SELECTION = max(pd.Timestamp("1923-01-01"), GLOBAL_MIN_DATE)
DEFAULT_END_SELECTION = GLOBAL_MAX_DATE

app = dash.Dash(__name__)
app.title = "Category Returns"
app.layout = html.Div(
    [
        html.H1("Category Returns"),
        html.Div(
            [
                html.Label("Select start and end date"),
                dcc.DatePickerRange(
                    id="date-picker",
                    start_date=to_date_input(DEFAULT_START_SELECTION),
                    end_date=to_date_input(DEFAULT_END_SELECTION),
                    min_date_allowed=to_date_input(GLOBAL_MIN_DATE),
                    max_date_allowed=to_date_input(GLOBAL_MAX_DATE),
                    display_format="YYYY-MM-DD",
                ),
            ],
            style={"width": "40%", "display": "inline-block", "padding": "0 20px 20px 0"},
        ),
        html.Div(
            [
                html.Label("Return frequency"),
                dcc.RadioItems(
                    id="return-frequency",
                    options=[
                        {"label": "Quarterly", "value": "quarterly"},
                        {"label": "Yearly", "value": "yearly"},
                    ],
                    value="quarterly",
                    labelStyle={"display": "block"},
                ),
            ],
            style={"width": "20%", "display": "inline-block", "verticalAlign": "top"},
        ),
        html.Div(
            [
                html.Label("Data history"),
                dcc.RadioItems(
                    id="history-length",
                    options=[
                        {"label": "ETF history", "value": "short"},
                        {"label": "Long history (mutual funds)", "value": "long"},
                    ],
                    value="short",
                    labelStyle={"display": "block"},
                ),
            ],
            style={"width": "25%", "display": "inline-block", "verticalAlign": "top", "padding": "0 0 0 20px"},
        ),
        dcc.Graph(id="return-graph"),
        dcc.Graph(id="cumulative-graph"),
    ]
)


@app.callback(
    Output("return-graph", "figure"),
    Output("cumulative-graph", "figure"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date"),
    Input("return-frequency", "value"),
    Input("history-length", "value"),
)
def update_graphs(
    start_date: str | None,
    end_date: str | None,
    frequency: str,
    history_length: str,
) -> tuple[go.Figure, go.Figure]:
    use_long_history = history_length == "long"
    dataset = DATASETS[use_long_history]

    if frequency == "yearly":
        returns = dataset["yearly"]
        frequency_label = "Yearly"
    else:
        returns = dataset["quarterly"]
        frequency_label = "Quarterly"

    scoped_returns = filter_returns(returns, start_date, end_date)
    if scoped_returns.empty:
        message = "No data available for the selected range."
        return (
            make_empty_figure(f"{frequency_label} Returns", message),
            make_empty_figure(f"{frequency_label} Cumulative Returns", message),
        )

    return (
        build_return_figure(scoped_returns, f"{frequency_label} Returns"),
        build_cumulative_figure(scoped_returns, f"{frequency_label} Cumulative Returns"),
    )


if __name__ == "__main__":
    app.run(debug=True)
