import math
import os
import urllib
import pandas as pd
import plotly.express as px
import gcsfs
import dash
import fsspec
from dash import dcc, callback, Output, Input, dash_table, State, html, ctx
import dash_bootstrap_components as dbc
import polars as pl
from urllib.parse import quote, parse_qs
import google.auth
from gcsfs import GCSFileSystem
from functools import lru_cache

dash.register_page(
    __name__,
    path_template="/dashboards",
    title="Dashboards",
)

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser("~/.gcp/dash-service-key.json")
# fs = gcsfs.GCSFileSystem(token=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])

credentials, project = google.auth.default()
fs = GCSFileSystem(token=credentials)

DATASETS = {}

# map project to parquet file path
PROJECT_PARQUET_MAP = {
    "dtol": "python_dash_data_bucket/metadata_dtol*.parquet",
    "erga": "python_dash_data_bucket/metadata_erga*.parquet",
    "asg": "python_dash_data_bucket/metadata_asg*.parquet",
    "gbdp": "python_dash_data_bucket/metadata_gbdp*.parquet"
}

PORTAL_URL_PREFIX = {
    "dtol": "https://portal.darwintreeoflife.org/data/",
    "erga": "https://portal.erga-biodiversity.eu/data_portal/",
    "asg": "https://portal.aquaticsymbiosisgenomics.org/data/root/details/",
    "gbdp": "https://www.ebi.ac.uk/biodiversity/data_portal/"
}

def limit_grouped_data(df: pl.DataFrame, col: str, df_filtered: pl.DataFrame) -> pl.DataFrame:
    top_n = 10

    # first_public (group by year)
    if col == "first_public":
        df = df.with_columns(
            pl.col("first_public").dt.year().cast(pl.Utf8).alias("first_public_year")
        )

        df = df.group_by("first_public_year").agg(
            pl.col("Record Count").sum().alias("Record Count")
        ).sort(by="first_public_year").head(10)

        # Use first_public_year as the category column
        col = "first_public_year"

    # "Others" category
    if len(df) > top_n:
        top_rows = df.head(top_n)
        other_rows = df.slice(top_n, len(df) - top_n)

        def concat_unique_polars(series):
            all_values = []
            for val in series:
                if val is not None:
                    items = str(val).split(',')
                    all_values.extend([item.strip() for item in items if item.strip()])
            return ", ".join(set(all_values))

        # get biosample IDs from other_rows
        other_biosample_ids = set()
        for val in other_rows.select("organisms.biosample_id").to_series():
            if val is not None:
                ids = [x.strip() for x in str(val).split(",") if x.strip()]
                other_biosample_ids.update(ids)

        # filter the raw data using biosample IDs from "Others"
        filtered_df = df_filtered.filter(
            pl.col("organisms.biosample_id").is_in(list(other_biosample_ids))
        )

        grouped_df = filtered_df.group_by("organisms.biosample_id").agg([
            pl.col("common_name").first(),
            pl.col("current_status").first(),
            pl.col("symbionts_status").first(),
            pl.col("organisms.organism").first()
        ])

        # create organism links
        grouped_df = grouped_df.with_columns(
            pl.when(pl.col("organisms.organism").is_not_null())
            .then(
                pl.col("organisms.organism").map_elements(
                    lambda x: f"[{x}](https://portal.darwintreeoflife.org/data/{urllib.parse.quote(str(x))})",
                    return_dtype=pl.Utf8
                )
            )
            .otherwise(pl.lit(""))
            .alias("organism_link")
        )

        # remove duplicates
        grouped_df = grouped_df.unique(subset=[
            "organism_link",
            "common_name",
            "current_status",
            "symbionts_status"
        ])

        record_count = len(grouped_df)

        # create the "Others" row with the same column structure as top_rows
        other_biosample_id = concat_unique_polars(other_rows.select("organisms.biosample_id").to_series())
        other_organism = concat_unique_polars(other_rows.select("organisms.organism").to_series())
        other_sex = concat_unique_polars(other_rows.select("organisms.sex").to_series())
        other_current_status = concat_unique_polars(other_rows.select("current_status").to_series())

        other_row_data = {}

        # initialize with same columns as top_rows
        for column in top_rows.columns:
            if column == col:
                other_row_data[column] = ["Others"]
            elif column == "organisms.biosample_id":
                other_row_data[column] = [other_biosample_id]
            elif column == "organisms.organism":
                other_row_data[column] = [other_organism]
            elif column == "organisms.sex":
                other_row_data[column] = [other_sex]
            elif column == "current_status":
                other_row_data[column] = [other_current_status]
            elif column == "Record Count":
                other_row_data[column] = [record_count]
            else:
                other_row_data[column] = [None]

        other_row_df = pl.DataFrame(other_row_data)

        # ensure data types match between the dataframes before concatenating
        for col_name in top_rows.columns:
            if col_name in other_row_df.columns:
                top_dtype = top_rows[col_name].dtype
                other_dtype = other_row_df[col_name].dtype

                # cast other_row_df column to match top_rows dtype
                if top_dtype != other_dtype:
                    other_row_df = other_row_df.with_columns(
                        pl.col(col_name).cast(top_dtype).alias(col_name)
                    )

        # concatenate top rows with Others row
        result_df = pl.concat([top_rows, other_row_df], how="vertical")

    else:
        result_df = df

    # cast to String/Utf8 and remove whitespace
    if col in result_df.columns and result_df[col].dtype != pl.Utf8:
        result_df = result_df.with_columns(pl.col(col).cast(pl.Utf8))

    # remove whitespace from string columns
    for c in result_df.columns:
        if result_df[c].dtype == pl.Utf8:
            result_df = result_df.with_columns(pl.col(c).str.strip_chars().alias(c))

    return result_df


def generate_grouped_data(df: pl.LazyFrame, group_by_col: str) -> pl.LazyFrame:
    df = df.filter(pl.col(group_by_col).is_not_null())
    grouped_data = (df.group_by(group_by_col)).agg([
                          pl.col("organisms.biosample_id")
                            .map_elements(lambda x: ", ".join(list(set(val for val in x if val is not None))), return_dtype=pl.Utf8)
                            .alias("organisms.biosample_id"),
                          pl.col("organisms.organism")
                            .map_elements(lambda x: ", ".join(list(set(val for val in x if val is not None))), return_dtype=pl.Utf8)
                            .alias("organisms.organism"),
                          pl.col("organisms.sex")
                            .map_elements(lambda x: ", ".join(list(set(val for val in x if val is not None))), return_dtype=pl.Utf8)
                            .alias("organisms.sex"),
                          pl.col("current_status")
                            .map_elements(lambda x: ", ".join(list(set(val for val in x if val is not None))), return_dtype=pl.Utf8)
                            .alias("current_status"),
                          pl.col("symbionts_status")
                            .map_elements(lambda x: ", ".join(list(set(val for val in x if val is not None))), return_dtype=pl.Utf8)
                            .alias("symbionts_status"),
                          pl.col("organisms.organism").n_unique().alias("Record Count"),
                          pl.col("organisms.biosample_id").n_unique().alias("BioSample IDs Count")
                      ]).sort("Record Count", descending=True)

    return grouped_data


def preprocess_chunk(df_nested: pl.LazyFrame) -> pl.LazyFrame:
    # explode organisms and flatten fields
    df_exploded = df_nested.explode("organisms").with_columns([
        pl.col("organisms").struct.field("biosample_id").alias("organisms.biosample_id"),
        pl.col("organisms").struct.field("organism").alias("organisms.organism"),
        pl.col("organisms").struct.field("sex").alias("organisms.sex"),
        pl.col("organisms").struct.field("lifestage").alias("organisms.lifestage"),
        pl.col("organisms").struct.field("habitat").alias("organisms.habitat"),
    ]).drop("organisms")

    # explode raw_data and flatten fields
    df_exploded = df_exploded.explode("raw_data").with_columns([
        pl.col("raw_data").struct.field("instrument_platform").alias("raw_data.instrument_platform"),
        pl.col("raw_data").struct.field("instrument_model").alias("raw_data.instrument_model"),
        pl.col("raw_data").struct.field("library_construction_protocol").alias("raw_data.library_construction_protocol"),
        pl.col("raw_data").struct.field("first_public").alias("raw_data.first_public"),
    ]).drop("raw_data")

    return df_exploded


def load_data(project_name):
    if project_name in DATASETS:
        return DATASETS[project_name]

    # get url configuration
    link_prefix = PORTAL_URL_PREFIX.get(project_name, "")
    url_param = "tax_id" if project_name in ["erga", "gbdp"] else "organisms.organism"

    # get file pattern and scan files
    gcs_pattern = PROJECT_PARQUET_MAP.get(project_name, PROJECT_PARQUET_MAP["dtol"])
    fs = fsspec.filesystem("gcs")
    matching_files = fs.glob(gcs_pattern)

    if not matching_files:
        raise FileNotFoundError(f"No Parquet files found for pattern: {gcs_pattern}")

    def quote_organism(o):
        return urllib.parse.quote(str(o)) if o is not None else None

    # check available columns in first file
    first_file = f"gs://{matching_files[0]}"
    available_columns = pl.scan_parquet(first_file).collect_schema().names()

    required_columns = [
        "organisms", "raw_data", "current_status", "tax_id", "symbionts_status", "common_name"
    ]

    select_columns = [col for col in required_columns if col in available_columns]

    # lazy frames for all files
    lazy_chunks = []
    for file in matching_files:
        file_path = f"gs://{file}"
        lf = pl.scan_parquet(file_path).select(select_columns)
        lazy_chunks.append(lf)

    combined_lf = pl.concat(lazy_chunks)


    raw_data_lf = (
        combined_lf
        # organisms
        .explode("organisms")
        .with_columns([
            pl.col("organisms").struct.field("biosample_id").alias("organisms.biosample_id"),
            pl.col("organisms").struct.field("organism").alias("organisms.organism"),
            pl.col("organisms").struct.field("sex").alias("organisms.sex"),
            pl.col("organisms").struct.field("lifestage").alias("organisms.lifestage"),
            pl.col("organisms").struct.field("habitat").alias("organisms.habitat"),
        ])

        # raw_data
        .explode("raw_data")
        .with_columns([
            pl.col("raw_data").struct.field("instrument_platform").alias("raw_data.instrument_platform"),
            pl.col("raw_data").struct.field("instrument_model").alias("raw_data.instrument_model"),
            pl.col("raw_data").struct.field("library_construction_protocol").alias(
                "raw_data.library_construction_protocol"),
            pl.col("raw_data").struct.field("first_public").alias("raw_data.first_public"),
        ])

        .with_columns([
            pl.col("organisms.sex").cast(pl.Utf8).alias("sex"),
            pl.col("organisms.lifestage").cast(pl.Utf8).alias("lifestage"),
            pl.col("organisms.habitat").cast(pl.Utf8).alias("habitat"),
            pl.col("raw_data.instrument_platform").cast(pl.Utf8).alias("instrument_platform"),
            pl.col("raw_data.instrument_model").cast(pl.Utf8).alias("instrument_model"),
            pl.col("raw_data.library_construction_protocol").cast(pl.Utf8).alias("library_construction_protocol"),
            pl.col("raw_data.first_public")
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False, exact=True)
            .cast(pl.Datetime)
            .alias("first_public"),
        ])

        # organism link
        .with_columns(
            pl.when(pl.col("organisms.organism").is_not_null())
            .then(pl.format(
                "[{}]({}{})",
                pl.col("organisms.organism"),
                pl.lit(link_prefix),
                pl.col(url_param).map_elements(quote_organism, return_dtype=pl.Utf8)
            ))
            .otherwise(pl.lit(""))
            .alias("organism_link")
        )
    )

    raw_data = raw_data_lf.collect()

    DATASETS[project_name] = {"raw_data": raw_data}

    return DATASETS[project_name]


@lru_cache(maxsize=10)
def load_data_cached(project_name):
    return load_data(project_name)


def wrap_text_for_display(text, max_chars_per_line=15):
    text_str = str(text)
    if len(text_str) <= max_chars_per_line:
        return text_str

    words = text_str.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line + " " + word) > max_chars_per_line and current_line:
            lines.append(current_line.strip())
            current_line = word
        else:
            if current_line:
                current_line += " " + word
            else:
                current_line = word

    if current_line:
        lines.append(current_line.strip())

    return "<br>".join(lines)


def build_pie(df: pl.DataFrame, col_name: str, selected_val: str, type='piechart'):
    disabled_color = '#cccccc'
    palette = px.colors.qualitative.Plotly
    hole = 0.3 if type == 'doughnut_chart' else 0
    piechart_title = col_name.replace("_", " ").title()

    df_cleaned = df.drop_nulls(subset=["Record Count", col_name])

    # add legend display column for wrapped text using map_elements
    df_cleaned = df_cleaned.with_columns(
        pl.col(col_name).map_elements(wrap_text_for_display, return_dtype=pl.Utf8).alias(col_name + '_display')
    )

    # convert to pandas dataframe for plotly operations
    df_pandas = df_cleaned.to_pandas()

    pie = px.pie(
        data_frame=df_pandas,
        names=col_name + '_display',  # Use wrapped text for display
        values="Record Count",
        title=piechart_title,
        hole=hole,
        hover_data={"BioSample IDs Count": True, col_name: False}
    )

    ordered_display_names = list(pie.data[0]['labels'])

    # create a lookup from display name to original value and biosample count
    lookup_df = df_cleaned.select(
        pl.col(col_name + '_display'),
        pl.col(col_name).alias("original_value"),
        pl.col("BioSample IDs Count").alias("biosample_count")
    ).to_dicts()

    lookup_map = {item[col_name + '_display']: (item["biosample_count"], item["original_value"]) for item in lookup_df}

    reordered_custom_data = []
    reordered_colors = []
    reordered_pulls = []

    for i, display_name in enumerate(ordered_display_names):
        original_bs_count, original_val = lookup_map.get(display_name, (None, None))
        if original_val is not None:
            reordered_custom_data.append([original_bs_count, original_val])
            reordered_colors.append(
                disabled_color if original_val == "Others" else palette[i % len(palette)])  # <--- Use 'i'
            reordered_pulls.append(0.1 if original_val == selected_val else 0)
        else:
            reordered_custom_data.append([None, None])
            reordered_colors.append(disabled_color)
            reordered_pulls.append(0)

    pie.data[0].customdata = reordered_custom_data
    pie.data[0].marker.colors = reordered_colors
    pie.data[0].pull = reordered_pulls

    pie.update_traces(
        text=df_pandas["text"],
        textinfo="text",
        textposition="inside",
        hovertemplate="<b>%{customdata[0][1]}</b><br>Record Count: %{value}<br>BioSample IDs Count: %{customdata[0][0]}<extra></extra>"
    )

    pie.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        ),
        width=500,
        height=400,
        margin=dict(l=20, r=180, t=50, b=20),
        title={
            'text': piechart_title,
            'x': 0.5,  # Center the title
            'xanchor': 'center'
        }
    )

    return pie


def generate_pie_labels(df: pl.DataFrame) -> pl.DataFrame:
    total = df["Record Count"].sum()
    if total > 0:
        df = df.with_columns([
            (pl.col("Record Count") / total * 100).alias("Percentage")
        ])
        df = df.with_columns([
            pl.when(pl.col("Percentage") > 5)
            .then(
                pl.col("Percentage").map_elements(lambda x: f"{x:.1f}%", return_dtype=pl.Utf8)
            )
            .otherwise(pl.lit(""))
            .alias("text")
        ])
    else:
        df = df.with_columns([
            pl.lit(0.0).alias("Percentage"),
            pl.lit("").alias("text")
        ])
    return df


def layout(**kwargs):
    project_name = kwargs.get("projectName", "dtol")

    load_data(project_name)

    header_colour = "#f1f3f4"
    if project_name == "dtol":
        header_colour = "#8fbc45"
    elif project_name == "erga":
        header_colour = "#e0efea"
    elif project_name == "asg":
        header_colour = "#add8e6"
    elif project_name == "gbdp":
        header_colour = "#d0d0ce"

    tab1_content = html.Div([
        dcc.Store(id="stored-selection", data={"sex": None, "lifestage": None, "habitat": None}),
        dcc.Store(id="project-store", data=project_name),
        html.Div(id="metadata-filter-selection",
                 style={"marginBottom": "30px", "marginTop": "40px", "textAlign": "center",
                        "fontWeight": "bold", "color": "#394959"}),

        dbc.Row([
            dbc.Col(
                dcc.Loading(
                    id="loading-pie-sex",
                    type="circle",
                    color="#0d6efd",
                    children=dcc.Graph(id="pie-sex")
                ),
                width=4
            ),
            dbc.Col(
                dcc.Loading(
                    id="loading-pie-lifestage",
                    type="circle",
                    color="#0d6efd",
                    children=dcc.Graph(id="pie-lifestage")
                ),
                width=4
            ),
            dbc.Col(
                dcc.Loading(
                    id="loading-pie-habitat",
                    type="circle",
                    color="#0d6efd",
                    children=dcc.Graph(id="pie-habitat")
                ),
                width=4
            ),
        ], style={"marginBottom": "40px"}),

        html.Div(
            dbc.Button("Reset All", id="reset-all-dashboards", color="danger", className="mb-3"),
            style={"display": "flex", "justifyContent": "flex-end"}
        ),

        dbc.Row(dbc.Col(
            dcc.Loading(
                id="loading-dashboard-datatable",
                type="circle",
                color="#0d6efd",
                children=dash_table.DataTable(
                    id="dashboard-datatable-paging",
                    columns=[
                        {"name": "Scientific Name", "id": "organism_link", "presentation": "markdown"},
                        {"name": "Common Name", "id": "common_name"},
                        {"name": "Current Status", "id": "current_status"},
                        {"name": "Symbionts Status", "id": "symbionts_status"}
                    ],
                    style_cell={"textAlign": "center"},
                    style_header={
                        "backgroundColor": header_colour,
                        "color": "#141414",

                    },
                    css=[dict(selector="p", rule="margin: 0; text-align: center"),
                         dict(selector="a", rule="text-decoration: none")],
                    page_current=0,
                    page_size=10,
                    page_action="custom",
                    style_table={'overflowX': 'scroll'}
                )
            ), md=12, id="col-table"),
            style={"marginTop": "20px", "marginBottom": "20px"}
        )
    ])

    tab2_content = html.Div([
        dcc.Store(id="rawdata-stored-selection",
                  data={"instrument_platform": None, "instrument_model": None, "library_construction_protocol": None}),
        dcc.Store(id="rawdata-selected-date", data=None),
        dcc.Store(id="tab2-project-store", data=project_name),


        html.Div(id="rawdata-filter-selection",
                 style={"marginBottom": "30px", "marginTop": "40px", "textAlign": "center",
                        "fontWeight": "bold", "color": "#394959"}),

        dbc.Row([
            dbc.Col(
                dcc.Loading(
                    id="loading-pie-platform",
                    type="circle",
                    color="#0d6efd",
                    children=dcc.Graph(id="pie-instrument-platform")
                ),
                width=4
            ),
            dbc.Col(
                dcc.Loading(
                    id="loading-pie-model",
                    type="circle",
                    color="#0d6efd",
                    children=dcc.Graph(id="pie-instrument-model")
                ),
                width=4
            ),
            dbc.Col(
                dcc.Loading(
                    id="loading-pie-protocol",
                    type="circle",
                    color="#0d6efd",
                    children=dcc.Graph(id="pie-library-construction-protocol")
                ),
                width=4
            ),
        ], style={"marginBottom": "40px"}),

        # date range picker
        dbc.Row([
            dbc.Col(width=8),  # spacer
            dbc.Col(
                dcc.DatePickerRange(
                    id='date-range-picker',
                    display_format='DD/MM/YYYY',
                    month_format='MMM YYYY',
                    clearable=True,
                    style={"display": "inline-block"}
                ),
                width=4,
                style={"textAlign": "right", "marginBottom": "20px"}
            ),
        ]),

        dbc.Row([
            dbc.Col(
                dcc.Loading(
                    id="loading-time-series",
                    type="circle",
                    color="#0d6efd",
                    children=dcc.Graph(id="time-series-graph")
                ),
                width=12
            )
        ], style={"marginBottom": "20px"}),

        html.Div(
            dbc.Button("Reset All", id="tab2-reset-all-dashboards", color="danger", className="mb-3"),
            style={"display": "flex", "justifyContent": "flex-end"}
        ),

        dbc.Row(dbc.Col(
            dcc.Loading(
                id="tab2-loading-dashboard-datatable",
                type="circle",
                color="#0d6efd",
                children=dash_table.DataTable(
                    id="tab2-dashboard-datatable-paging",
                    columns=[
                        {"name": "Scientific Name", "id": "organism_link", "presentation": "markdown"},
                        {"name": "Common Name", "id": "common_name"},
                        {"name": "Current Status", "id": "current_status"},
                        {"name": "Symbionts Status", "id": "symbionts_status"}
                    ],
                    style_cell={"textAlign": "center"},
                    style_header={
                        "backgroundColor": header_colour,
                        "color": "#141414",

                    },
                    css=[dict(selector="p", rule="margin: 0; text-align: center"),
                         dict(selector="a", rule="text-decoration: none")],
                    page_current=0,
                    page_size=10,
                    page_action="custom",
                    style_table={'overflowX': 'scroll'}
                )
            ), md=12, id="tab2-col-table"),
            style={"marginTop": "20px", "marginBottom": "20px"}
        )
    ])

    return dbc.Container([
        dcc.Tabs([
            dcc.Tab(label='Metadata', children=[tab1_content]),
            dcc.Tab(label='Raw Data', children=[tab2_content]),
        ])
    ], fluid=True)


@callback(
    Output("metadata-filter-selection", "children"),
    Input("stored-selection", "data")
)
def update_selection_text(selected):
    default_msg = "Please click on the pie charts to filter the data"
    if not selected:
        return default_msg

    parts = []
    for key in ["sex", "lifestage", "habitat"]:
        value = selected.get(key)
        if value:
            label = key.capitalize()
            parts.extend([
                html.Span(f"{label}: "),
                html.Span(f"{value}", style={"color": "#8fa8c2"}),
                html.Span(" | ")
            ])

    return parts[:-1] if parts else default_msg  # remove last " | "


@callback(
    Output("rawdata-filter-selection", "children"),
    Input("rawdata-stored-selection", "data"),
    Input("rawdata-selected-date", "data")
)
def update_selection_text(selected_piecharts, selected_date):
    default_msg = "Please click on the charts to filter the data"
    if not selected_piecharts and not selected_date:
        return default_msg

    parts = []
    for key in ["instrument_platform", "instrument_model", "library_construction_protocol"]:
        value = selected_piecharts.get(key)
        if value:
            label = key.replace("_", " ").title()
            parts.extend([
                html.Span(f"{label}: "),
                html.Span(f"{value}", style={"color": "#8fa8c2"}),
                html.Span(" | ")
            ])

    if selected_date:
        if isinstance(selected_date, str): # single date selection
            parts.extend([
                html.Span(f"Date: "),
                html.Span(f"{selected_date}", style={"color": "#8fa8c2"}),
                html.Span(" | ")
            ])
        elif isinstance(selected_date, dict): # date range selection
            start_date = selected_date.get("start")
            end_date = selected_date.get("end")
            if start_date and end_date:
                parts.extend([
                    html.Span(f"Date Range: "),
                    html.Span(f"{start_date} to {end_date}", style={"color": "#8fa8c2"}),
                    html.Span(" | ")
                ])
            elif start_date:
                parts.extend([
                    html.Span(f"Start Date: "),
                    html.Span(f"{start_date}", style={"color": "#8fa8c2"}),
                    html.Span(" | ")
                ])
            elif end_date:
                parts.extend([
                    html.Span(f"End Date: "),
                    html.Span(f"{end_date}", style={"color": "#8fa8c2"}),
                    html.Span(" | ")
                ])

    return parts[:-1] if parts else default_msg


@callback(
    Output("dashboard-datatable-paging", "page_current", allow_duplicate=True),
    Input("reset-all-dashboards", "n_clicks"),
    Input('pie-sex', 'clickData'),
    Input('pie-lifestage', 'clickData'),
    Input('pie-habitat', 'clickData'),
    State("dashboard-datatable-paging", "page_current"),
    prevent_initial_call=True
)
def reset_metadata_paging(n_clicks, sex_click, lifestage_click, habitat_click, current_page):
    if current_page != 0:
        return 0
    raise dash.exceptions.PreventUpdate


@callback(
    Output("tab2-dashboard-datatable-paging", "page_current", allow_duplicate=True),
    Input("tab2-reset-all-dashboards", "n_clicks"),
    Input('pie-instrument-platform', 'clickData'),
    Input('pie-instrument-model', 'clickData'),
    Input('pie-library-construction-protocol', 'clickData'),
    Input('time-series-graph', 'clickData'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
    State("tab2-dashboard-datatable-paging", "page_current"),
    prevent_initial_call=True
)
def reset_rawdata_paging(n_clicks, instrument_platform_click, instrument_model_click,
                         library_construction_protocol_click, time_series_click,
                         start_date, end_date, current_page):
    # only reset if the current page is not already 0, to avoid unnecessary updates
    if current_page != 0:
        return 0
    raise dash.exceptions.PreventUpdate


@dash.callback(
    Output("pie-sex", "figure"),
    Output("pie-lifestage", "figure"),
    Output("pie-habitat", "figure"),
    Output("stored-selection", "data"),
    Input("pie-sex", "clickData"),
    Input("pie-lifestage", "clickData"),
    Input("pie-habitat", "clickData"),
    Input("project-store", "data"),
    Input("stored-selection", "data"),
    Input("reset-all-dashboards", "n_clicks"),
    State("stored-selection", "data")
)
def build_metadata_charts(pie_sex_click, pie_lifestage_click, pie_habitat_click,
                          project_name, selected_pie_input,
                          reset_n_clicks, stored_selection):

    triggered_id = ctx.triggered_id

    stored_selection = selected_pie_input or {"sex": None, "lifestage": None, "habitat": None}

    def extract_label(click_data):
        if not click_data:
            return None
        return click_data["points"][0]["customdata"][1]

    if triggered_id == "pie-sex":
        label = extract_label(pie_sex_click)
        if label != "Others":
            stored_selection["sex"] = label
    elif triggered_id == "pie-lifestage":
        label = extract_label(pie_lifestage_click)
        if label != "Others":
            stored_selection["lifestage"] = label
    elif triggered_id == "pie-habitat":
        label = extract_label(pie_habitat_click)
        if label != "Others":
            stored_selection["habitat"] = label
    elif triggered_id == "reset-all-dashboards":
        stored_selection = {"sex": None, "lifestage": None, "habitat": None}

    # convert to lazyframe
    raw_lf = DATASETS[project_name]["raw_data"].lazy()

    pies = {}
    for dim in ["sex", "lifestage", "habitat"]:
        lf_filtered = raw_lf
        for key, value in stored_selection.items():
            if key != dim and value:
                lf_filtered = lf_filtered.filter(pl.col(key) == value)

        lf_grouped = generate_grouped_data(lf_filtered, dim)

        # collect for post-processing
        df_grouped = lf_grouped.collect()
        df_filtered = lf_filtered.collect()

        df_limited = limit_grouped_data(df_grouped, dim, df_filtered)
        df_final   = generate_pie_labels(df_limited)
        pies[dim] = build_pie(df_final, dim, stored_selection[dim])

    return pies["sex"], pies["lifestage"], pies["habitat"], stored_selection



# Replace your existing table callbacks with these fixed versions:

@callback(
    Output("dashboard-datatable-paging", "data"),
    Output("dashboard-datatable-paging", "page_count"),
    Output("dashboard-datatable-paging", "page_current"),

    Input('dashboard-datatable-paging', "page_current"),
    Input('dashboard-datatable-paging', "page_size"),
    Input("project-store", "data"),
    Input("stored-selection", "data"),
    # Add these pie chart inputs to trigger the loader
    Input("pie-sex", "clickData"),
    Input("pie-lifestage", "clickData"),
    Input("pie-habitat", "clickData"),
    Input("reset-all-dashboards", "n_clicks"),
)
def build_metadata_table(page_current: int, page_size: int, project_name: str, stored_selection: dict,
                        pie_sex_click, pie_lifestage_click, pie_habitat_click, reset_clicks):
    filtered_df = DATASETS[project_name]["raw_data"].clone()
    stored_selection = stored_selection or {"sex": None, "lifestage": None, "habitat": None}

    # data table filtering
    for key, value in stored_selection.items():
        if value:
            filtered_df = filtered_df.filter(pl.col(key) == value)

    grouped_df = filtered_df.group_by('organisms.biosample_id').agg(
        pl.col('common_name').first().alias('common_name'),
        pl.col('current_status').first().alias('current_status'),
        pl.col('tax_id').first().alias('tax_id'),
        pl.col('symbionts_status').first().alias('symbionts_status'),
        pl.col('organisms.organism').first().alias('organisms.organism'),
    )

    link_prefix =  PORTAL_URL_PREFIX.get(project_name, "")

    if project_name == "erga" or project_name == "gbdp":
        url_param = "tax_id"
    else:
        url_param = "organisms.organism"

    grouped_df = grouped_df.with_columns(
        pl.when(pl.col("organisms.organism").is_not_null())
        .then(
            pl.format(
                f"[{{}}]({link_prefix}{{}})",
                pl.col("organisms.organism"),
                pl.col(url_param)
                .map_elements(lambda x: urllib.parse.quote(str(x)), return_dtype=pl.Utf8)
            )
        )
        .otherwise(pl.lit(""))
        .alias("organism_link")
    )

    grouped_df = grouped_df.unique(subset=[
        "organism_link",
        "common_name",
        "current_status",
        "symbionts_status"
    ])

    grouped_df = grouped_df.sort(by="current_status", descending=False)

    # pagination
    total_rows = len(grouped_df)
    total_pages = max(math.ceil(total_rows / page_size), 1)
    page_current = min(page_current, total_pages - 1)

    return (
        grouped_df.slice(page_current * page_size, page_size).to_dicts(),
        total_pages,
        page_current
    )



# **** Tab 2 ***#
@dash.callback(
    Output('pie-instrument-platform', 'figure'),
    Output('pie-instrument-model', 'figure'),
    Output('pie-library-construction-protocol', 'figure'),
    Output('time-series-graph', 'figure'),
    Output('rawdata-stored-selection', 'data'),
    Output('rawdata-selected-date', 'data'),
    Output('date-range-picker', 'start_date'),
    Output('date-range-picker', 'end_date'),

    Input('pie-instrument-platform', 'clickData'),
    Input('pie-instrument-model', 'clickData'),
    Input('pie-library-construction-protocol', 'clickData'),
    Input('time-series-graph', 'clickData'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
    Input("tab2-project-store", "data"),
    Input("rawdata-stored-selection", "data"),
    Input("rawdata-selected-date", "data"),
    Input("tab2-reset-all-dashboards", "n_clicks"),

    State('rawdata-stored-selection', 'data'),
    State('rawdata-selected-date', 'data')
)
def build_rawdata_charts(instrument_platform_click, instrument_model_click,
                         library_construction_protocol_click, time_series_click,
                         picker_start_date, picker_end_date, # New arguments
                         project_name, selected_charts_input, selected_date_input,
                         reset_n_clicks, stored_selection, stored_date):
    triggered_id = ctx.triggered_id

    stored_selection = selected_charts_input or {"instrument_platform": None, "instrument_model": None,
                                                 "library_construction_protocol": None}

    stored_date = selected_date_input or None

    # clear date picker if a single bar is clicked or reset
    clear_picker_start = dash.no_update
    clear_picker_end = dash.no_update

    def extract_label(click_data):
        if not click_data:
            return None
        return click_data['points'][0]['customdata'][1]


    if triggered_id == "pie-instrument-platform":
        label = extract_label(instrument_platform_click)
        if label != "Others":
            stored_selection["instrument_platform"] = label
    elif triggered_id == "pie-instrument-model":
        label = extract_label(instrument_model_click)
        if label != "Others":
            stored_selection["instrument_model"] = label
    elif triggered_id == "pie-library-construction-protocol":
        label = extract_label(library_construction_protocol_click)
        if label != "Others":
            stored_selection["library_construction_protocol"] = label
    elif triggered_id == "time-series-graph":
        # if a time series bar is clicked, set stored_date to that single date
        if time_series_click:
            stored_date = time_series_click['points'][0]['x']
        else:
            stored_date = None
        clear_picker_start = None
        clear_picker_end = None
    elif triggered_id == "date-range-picker":
        if picker_start_date or picker_end_date:
            stored_date = {"start": picker_start_date, "end": picker_end_date}
        else:
            stored_date = None
    elif triggered_id == "tab2-reset-all-dashboards":
        stored_selection = {"instrument_platform": None, "instrument_model": None,
                            "library_construction_protocol": None}
        stored_date = None
        clear_picker_start = None
        clear_picker_end = None

    df_raw = DATASETS[project_name]["raw_data"]

    # apply filtering for each pie chart based on selection of the other two and date
    def filter_df(pie_key):
        df = df_raw.clone()
        for key, value in stored_selection.items():
            if key != pie_key and value:
                df = df.filter(pl.col(key) == value)

        if stored_date:
            if isinstance(stored_date, str):
                stored_date_obj = pl.lit(stored_date).str.to_date()
                df = df.filter(pl.col('first_public').dt.date() == stored_date_obj)
            elif isinstance(stored_date, dict):
                start_date = stored_date.get("start")
                end_date = stored_date.get("end")
                if start_date:
                    start_parsed = pl.lit(start_date).str.to_date()
                    df = df.filter(pl.col("first_public").dt.date() >= start_parsed)
                if end_date:
                    end_parsed = pl.lit(end_date).str.to_date()
                    df = df.filter(pl.col("first_public").dt.date() <= end_parsed)
        return df


    # filter and update pie datasets
    for key in ["instrument_platform", "instrument_model", "library_construction_protocol"]:
        df_sub = filter_df(key)
        grouped_data = generate_grouped_data(df_sub.lazy(), key).collect()
        updated_grouped_data = limit_grouped_data(grouped_data, key, df_sub)
        updated_grouped_data = generate_pie_labels(updated_grouped_data)
        DATASETS[project_name][key] = {"df_data": df_sub, "grouped_data": updated_grouped_data}

    df_instrument_platform = DATASETS[project_name]["instrument_platform"]["grouped_data"]
    df_instrument_model = DATASETS[project_name]["instrument_model"]["grouped_data"]
    df_library_construction_protocol = DATASETS[project_name]["library_construction_protocol"]["grouped_data"]


    pie_instrument_platform = build_pie(df_instrument_platform, "instrument_platform",
                                        stored_selection["instrument_platform"], 'doughnut_chart')

    pie_instrument_model = build_pie(df_instrument_model, "instrument_model",
                                     stored_selection["instrument_model"], 'doughnut_chart')

    pie_instrument_construction_protocol = build_pie(df_library_construction_protocol,
                                                     "library_construction_protocol",
                                                     stored_selection["library_construction_protocol"],
                                                     'doughnut_chart')

    # time series section
    df_all_dates = generate_grouped_data(df_raw.lazy(), "first_public").collect()
    df_all_dates = df_all_dates.with_columns([
        pl.col('first_public').dt.date().alias('Date'),
        pl.col('Record Count').alias('Total Organism Count'),
        pl.col('BioSample IDs Count').alias('Total BioSample IDs Count')
    ]).select(['Date', 'Total Organism Count', 'Total BioSample IDs Count'])

    df_filtered_ts = df_raw.clone()

    # apply pie selections for the time series
    for key, value in stored_selection.items():
        if value:
            df_filtered_ts = df_filtered_ts.filter(pl.col(key) == value)

    filtered_ts = generate_grouped_data(df_filtered_ts.lazy(), "first_public").collect()
    filtered_ts = filtered_ts.with_columns([
        pl.col('first_public').dt.date().alias('Date'),
        pl.col('Record Count').alias('Organism Count'),
        pl.col('BioSample IDs Count').alias('BioSample IDs Count')
    ]).select(['Date', 'Organism Count', 'BioSample IDs Count'])

    df_all_dates_pd = df_all_dates.to_pandas()
    filtered_ts_pd = filtered_ts.to_pandas()

    grouped_ts = pd.merge(df_all_dates_pd[['Date']], filtered_ts_pd, on='Date', how='left').fillna(0)

    # highlight selected date range
    highlight_color = 'rgb(255, 99, 71)'
    default_color = 'rgb(0, 102, 255)'

    grouped_ts['bar_color'] = default_color

    if stored_date:
        if isinstance(stored_date, str):
            stored_date_obj = pd.to_datetime(stored_date).date()
            grouped_ts['bar_color'] = grouped_ts['Date'].dt.date.apply(
                lambda d: highlight_color if d == stored_date_obj else default_color
            )
        elif isinstance(stored_date, dict):  # Date range selected
            start_date_range = stored_date.get("start")
            end_date_range = stored_date.get("end")
            if start_date_range:
                start_date_range_obj = pd.to_datetime(start_date_range).date()
            else:
                start_date_range_obj = None
            if end_date_range:
                end_date_range_obj = pd.to_datetime(end_date_range).date()
            else:
                end_date_range_obj = None

            def get_color_for_date(d):
                is_in_range = True
                if start_date_range_obj and d < start_date_range_obj:
                    is_in_range = False
                if end_date_range_obj and d > end_date_range_obj:
                    is_in_range = False
                return highlight_color if is_in_range else default_color

            grouped_ts['bar_color'] = grouped_ts['Date'].dt.date.apply(get_color_for_date)


    # customdata for hovertemplate
    customdata = grouped_ts[['Organism Count', 'BioSample IDs Count']].values

    time_series_fig = px.bar(
        grouped_ts,
        x='Date',
        y='Organism Count',
        title='Record Count Over Time',
    )

    time_series_fig.update_traces(
        marker_color=grouped_ts['bar_color'],
        marker_line_width=0,
        customdata=customdata,
        hovertemplate=(
            "<b>Date: %{x|%d %b %Y}</b><br>"
            "Organism Count: %{customdata[0]}<br>"
            "BioSample IDs Count: %{customdata[1]}<extra></extra>"
        )
    )

    time_series_fig.update_layout(
        height=300,
        plot_bgcolor='white',
        title_font_size=16,
        showlegend=False,
        margin=dict(l=30, r=30, t=40, b=40),
        xaxis=dict(
            title='',
            showgrid=True,
            gridcolor='rgba(230,230,230,0.5)',
            tickformat='%d %b %Y',
            tickangle=0
        ),
        yaxis=dict(
            title='',
            showgrid=True,
            gridcolor='rgba(230,230,230,0.5)'
        )
    )

    return (
        pie_instrument_platform,
        pie_instrument_model,
        pie_instrument_construction_protocol,
        time_series_fig,
        stored_selection,
        stored_date,
        clear_picker_start,
        clear_picker_end
    )


@callback(
    Output("tab2-dashboard-datatable-paging", "data"),
    Output("tab2-dashboard-datatable-paging", "page_count"),
    Output("tab2-dashboard-datatable-paging", "page_current"),

    Input("tab2-dashboard-datatable-paging", "page_current"),
    Input("tab2-dashboard-datatable-paging", "page_size"),
    Input("tab2-project-store", "data"),
    Input("rawdata-stored-selection", "data"),
    Input("rawdata-selected-date", "data"),
    Input('pie-instrument-platform', 'clickData'),
    Input('pie-instrument-model', 'clickData'),
    Input('pie-library-construction-protocol', 'clickData'),
    Input('time-series-graph', 'clickData'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
    Input("tab2-reset-all-dashboards", "n_clicks"),
)
def build_rawdata_table(page_current: int, page_size: int, project_name: str,
                       selected_pie_data: dict, selected_date,
                       platform_click, model_click, protocol_click, time_click,
                       start_date, end_date, reset_clicks):
    filtered_df = DATASETS[project_name]["raw_data"].clone()

    # apply pie chart filters
    if selected_pie_data:
        for category in ["instrument_platform", "instrument_model", "library_construction_protocol"]:
            selected_value = selected_pie_data.get(category)
            if selected_value and selected_value != "Others":
                filtered_df = filtered_df.filter(pl.col(category) == selected_value)

    # apply time series date filtering
    if selected_date:
        if isinstance(selected_date, str):
            # for exact date match, convert to date for comparison
            selected_date_parsed = pd.to_datetime(selected_date).date()
            filtered_df = filtered_df.filter(pl.col("first_public").dt.date() == selected_date_parsed)
        elif isinstance(selected_date, dict):
            start_date = selected_date.get("start")
            end_date = selected_date.get("end")

            if start_date:
                start_parsed = pl.lit(start_date).str.to_date()
                filtered_df = filtered_df.filter(pl.col("first_public").dt.date() >= start_parsed)
            if end_date:
                end_parsed = pl.lit(end_date).str.to_date()
                filtered_df = filtered_df.filter(pl.col("first_public").dt.date() <= end_parsed)

    # group by biosample and aggregate
    grouped_df = filtered_df.group_by("organisms.biosample_id").agg([
        pl.col("common_name").first().alias("common_name"),
        pl.col("current_status").first().alias("current_status"),
        pl.col("tax_id").first().alias("tax_id"),
        pl.col("symbionts_status").first().alias("symbionts_status"),
        pl.col("organisms.organism").first().alias("organisms.organism"),
    ])

    link_prefix =  PORTAL_URL_PREFIX.get(project_name, "")

    if project_name == "erga" or project_name == "gbdp":
        url_param = "tax_id"
    else:
        url_param = "organisms.organism"

    grouped_df = grouped_df.with_columns(
        pl.when(pl.col("organisms.organism").is_not_null())
        .then(
            pl.format(
                f"[{{}}]({link_prefix}{{}})",
                pl.col("organisms.organism"),
                pl.col(url_param)
                .map_elements(lambda x: urllib.parse.quote(str(x)), return_dtype=pl.Utf8)
            )
        )
        .otherwise(pl.lit(""))
        .alias("organism_link")
    )

    grouped_df = grouped_df.unique(subset=[
        "organism_link", "common_name", "current_status", "symbionts_status"
    ])

    grouped_df = grouped_df.sort(by="current_status", descending=False)

    # pagination
    total_rows = len(grouped_df)
    total_pages = max(math.ceil(total_rows / page_size), 1)
    page_current = min(page_current, total_pages - 1)

    return (
        grouped_df.slice(page_current * page_size, page_size).to_dicts(),
        total_pages,
        page_current
    )