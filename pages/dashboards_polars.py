import math
import urllib
import plotly.express as px
import dash
from dash import dcc, callback, Output, Input, dash_table, State, html, ctx
import dash_bootstrap_components as dbc
import polars as pl
from urllib.parse import quote

from dash.exceptions import PreventUpdate
from google.cloud import bigquery
from datetime import datetime
from google.oauth2 import service_account

dash.register_page(
    __name__,
    path_template="/dashboards",
    title="Dashboards",
)

# initialize BigQuery client
client = bigquery.Client(
    project="prj-ext-prod-biodiv-data-in"
)


# cache for pre-aggregated data
METADATA_AGGREGATED_DATA = {}
RAWDATA_AGGREGATED_DATA = {}


VIEW_MAPPINGS = {
    "dtol": {
        "metadata_aggregated": "prj-ext-prod-biodiv-data-in.dtol.metadata_aggregated",
        "rawdata_aggregated": "prj-ext-prod-biodiv-data-in.dtol.rawdata_aggregated",
        "table_data": "prj-ext-prod-biodiv-data-in.dtol.table_data"
    },
    "erga": {
        "metadata_aggregated": "prj-ext-prod-biodiv-data-in.erga.metadata_aggregated",
        "rawdata_aggregated": "prj-ext-prod-biodiv-data-in.erga.rawdata_aggregated",
        "table_data": "prj-ext-prod-biodiv-data-in.erga.table_data"
    },
    "asg": {
        "metadata_aggregated": "prj-ext-prod-biodiv-data-in.asg.metadata_aggregated",
        "rawdata_aggregated": "prj-ext-prod-biodiv-data-in.asg.rawdata_aggregated",
        "table_data": "prj-ext-prod-biodiv-data-in.asg.table_data"
    },
    "gbdp": {
        "metadata_aggregated": "prj-ext-prod-biodiv-data-in.gbdp.metadata_aggregated",
        "rawdata_aggregated": "prj-ext-prod-biodiv-data-in.gbdp.rawdata_aggregated",
        "table_data": "prj-ext-prod-biodiv-data-in.gbdp.table_data"
    }
}

PORTAL_URL_PREFIX = {
    "dtol": "https://portal.darwintreeoflife.org/data/",
    "erga": "https://portal.erga-biodiversity.eu/data_portal/",
    "asg": "https://portal.aquaticsymbiosisgenomics.org/data/root/details/",
    "gbdp": "https://www.ebi.ac.uk/biodiversity/data_portal/"
}

# get base data
def build_table_data_query(project_name: str, tab: str, filters: dict, offset: int = 0,
                           limit: int = 10) -> str:

    view_name = VIEW_MAPPINGS[project_name]["table_data"]

    where_conditions = [f"project_name = '{project_name}'"]

    for key, value in filters.items():
        if value and value != "NULL":
            if tab == 'metadata':
                if key == "sex":
                    where_conditions.append(f"sex = '{value}'")
                elif key == "lifestage":
                    where_conditions.append(f"lifestage = '{value}'")
                elif key == "habitat":
                    where_conditions.append(f"habitat = '{value}'")
            elif tab == "rawdata":
                if key == "instrument_platform":
                    where_conditions.append(f"instrument_platform = '{value}'")
                elif key == "instrument_model":
                    where_conditions.append(f"instrument_model = '{value}'")
                elif key == "library_construction_protocol":
                    where_conditions.append(f"library_construction_protocol = '{value}'")
                elif key == "date_filter":
                    # Date filtering
                    date_info = value
                    if isinstance(date_info, str):
                        where_conditions.append(f"first_public = '{date_info}'")
                    elif isinstance(date_info, dict):
                        start_date = date_info.get("start")
                        end_date = date_info.get("end")
                        if start_date:
                            where_conditions.append(f"first_public >= '{start_date}'")
                        if end_date:
                            where_conditions.append(f"first_public <= '{end_date}'")

    where_clause = " AND ".join(where_conditions)

    return f"""
    SELECT DISTINCT
        scientific_name,
        tax_id,
        common_name,
        current_status,
        symbionts_status
    FROM `{view_name}`
    WHERE {where_clause}
    ORDER BY current_status
    LIMIT {limit}
    OFFSET {offset}
    """


def build_table_count_query(project_name: str, tab: str, filters: dict) -> str:

    view_name = VIEW_MAPPINGS[project_name]["table_data"]

    where_conditions = [f"project_name = '{project_name}'"]

    for key, value in filters.items():
        if value and value != "NULL":
            if tab == 'metadata':
                if key == "sex":
                    where_conditions.append(f"sex = '{value}'")
                elif key == "lifestage":
                    where_conditions.append(f"lifestage = '{value}'")
                elif key == "habitat":
                    where_conditions.append(f"habitat = '{value}'")
            elif tab == "rawdata":
                if key == "instrument_platform":
                    where_conditions.append(f"instrument_platform = '{value}'")
                elif key == "instrument_model":
                    where_conditions.append(f"instrument_model = '{value}'")
                elif key == "library_construction_protocol":
                    where_conditions.append(f"library_construction_protocol = '{value}'")
                elif key == "date_filter":
                    # Date filtering
                    date_info = value
                    if isinstance(date_info, str):
                        where_conditions.append(f"first_public = '{date_info}'")
                    elif isinstance(date_info, dict):
                        start_date = date_info.get("start")
                        end_date = date_info.get("end")
                        if start_date:
                            where_conditions.append(f"first_public >= '{start_date}'")
                        if end_date:
                            where_conditions.append(f"first_public <= '{end_date}'")

    where_clause = " AND ".join(where_conditions)

    return f"""
        SELECT COUNT(DISTINCT tax_id) as total_count  -- Count by tax_id for unique records
        FROM `{view_name}`
        WHERE {where_clause}
        """




# metadata
# load pre-aggregated data for charts
def load_metadata_aggregated_data(project_name: str) -> dict:

    if project_name in METADATA_AGGREGATED_DATA:
        return METADATA_AGGREGATED_DATA[project_name]

    print(f"Loading aggregated data for {project_name} from views...")

    view_name = VIEW_MAPPINGS[project_name]["metadata_aggregated"]
    query = f"SELECT * FROM `{view_name}` ORDER BY dimension, record_count DESC"

    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
        use_legacy_sql=False,
        maximum_bytes_billed=10 ** 9
    )

    query_job = client.query(query, job_config=job_config)
    df_pandas = query_job.to_dataframe(create_bqstorage_client=True)
    df_polars = pl.from_pandas(df_pandas)

    aggregated = {
        'sex': df_polars.filter(pl.col('dimension') == 'sex'),
        'lifestage': df_polars.filter(pl.col('dimension') == 'lifestage'),
        'habitat': df_polars.filter(pl.col('dimension') == 'habitat'),
        'cross_filter': df_polars.filter(pl.col('dimension') == 'cross_filter')
    }

    METADATA_AGGREGATED_DATA[project_name] = aggregated
    print(f"Aggregated data loaded for {project_name}")
    return aggregated


# get chart data with cross-filtering applied
def get_metadata_chart_data(project_name: str, dimension: str, filters: dict) -> pl.DataFrame:

    aggregated = METADATA_AGGREGATED_DATA[project_name]
    cross_filter_data = aggregated['cross_filter']

    # apply filters to cross-filter data
    filtered_data = cross_filter_data
    for key, value in filters.items():
        if key != dimension and value and value != "NULL":
            filter_col = f"filter_{key}"
            filtered_data = filtered_data.filter(pl.col(filter_col) == value)

    # aggregate by selected slice
    if dimension == "sex":
        result = filtered_data.group_by("filter_sex").agg([
            pl.col("record_count").sum().alias("record_count"),
            pl.col("biosample_count").sum().alias("biosample_count")
        ]).rename({"filter_sex": "value"}).filter(pl.col("value").is_not_null())
    elif dimension == "lifestage":
        result = filtered_data.group_by("filter_lifestage").agg([
            pl.col("record_count").sum().alias("record_count"),
            pl.col("biosample_count").sum().alias("biosample_count")
        ]).rename({"filter_lifestage": "value"}).filter(pl.col("value").is_not_null())
    elif dimension == "habitat":
        result = filtered_data.group_by("filter_habitat").agg([
            pl.col("record_count").sum().alias("record_count"),
            pl.col("biosample_count").sum().alias("biosample_count")
        ]).rename({"filter_habitat": "value"}).filter(pl.col("value").is_not_null())

    return result.sort("record_count", descending=True)


# raw_data
def load_rawdata_aggregated_data(project_name: str) -> dict:

    if project_name in RAWDATA_AGGREGATED_DATA:
        return RAWDATA_AGGREGATED_DATA[project_name]

    print(f"Loading raw data aggregated data for {project_name} from views...")

    view_name = VIEW_MAPPINGS[project_name]["rawdata_aggregated"]
    query = f"SELECT * FROM `{view_name}` ORDER BY dimension, record_count DESC"

    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
        use_legacy_sql=False,
        maximum_bytes_billed=10 ** 9
    )

    query_job = client.query(query, job_config=job_config)
    df_pandas = query_job.to_dataframe(create_bqstorage_client=True)
    df_polars = pl.from_pandas(df_pandas)

    aggregated = {
        'instrument_platform': df_polars.filter(pl.col('dimension') == 'instrument_platform'),
        'instrument_model': df_polars.filter(pl.col('dimension') == 'instrument_model'),
        'library_construction_protocol': df_polars.filter(pl.col('dimension') == 'library_construction_protocol'),
        'time_series': df_polars.filter(pl.col('dimension') == 'time_series'),
        'cross_filter': df_polars.filter(pl.col('dimension') == 'cross_filter')
    }

    RAWDATA_AGGREGATED_DATA[project_name] = aggregated
    print(f"Raw data aggregated data loaded for {project_name}")
    return aggregated


# get chart data with cross-filtering applied
def get_rawdata_chart_data(project_name: str, dimension: str, filters: dict) -> pl.DataFrame:
    aggregated = RAWDATA_AGGREGATED_DATA[project_name]
    cross_filter_data = aggregated['cross_filter']

    # apply filters to cross-filter data
    filtered_data = cross_filter_data
    for key, value in filters.items():
        if key != dimension and value and value != "NULL":
            if key == "instrument_platform":
                filtered_data = filtered_data.filter(pl.col("filter_platform") == value)
            elif key == "instrument_model":
                filtered_data = filtered_data.filter(pl.col("filter_model") == value)
            elif key == "library_construction_protocol":
                filtered_data = filtered_data.filter(pl.col("filter_protocol") == value)
            elif key == "date_filter":
                # Handle date filtering for cross-filter
                date_info = value
                if isinstance(date_info, str):
                    date_parsed = pl.lit(date_info).str.to_date()
                    filtered_data = filtered_data.filter(pl.col("filter_date") == date_parsed)
                elif isinstance(date_info, dict):
                    start_date = date_info.get("start")
                    end_date = date_info.get("end")
                    if start_date:
                        start_parsed = pl.lit(start_date).str.to_date()
                        filtered_data = filtered_data.filter(pl.col("filter_date") >= start_parsed)
                    if end_date:
                        end_parsed = pl.lit(end_date).str.to_date()
                        filtered_data = filtered_data.filter(pl.col("filter_date") <= end_parsed)

    # aggregate by dimension
    if dimension == "instrument_platform":
        result = filtered_data.group_by("filter_platform").agg([
            pl.col("record_count").sum().alias("record_count"),
            pl.col("biosample_count").sum().alias("biosample_count")
        ]).rename({"filter_platform": "value"}).filter(pl.col("value").is_not_null())
    elif dimension == "instrument_model":
        result = filtered_data.group_by("filter_model").agg([
            pl.col("record_count").sum().alias("record_count"),
            pl.col("biosample_count").sum().alias("biosample_count")
        ]).rename({"filter_model": "value"}).filter(pl.col("value").is_not_null())
    elif dimension == "library_construction_protocol":
        result = filtered_data.group_by("filter_protocol").agg([
            pl.col("record_count").sum().alias("record_count"),
            pl.col("biosample_count").sum().alias("biosample_count")
        ]).rename({"filter_protocol": "value"}).filter(pl.col("value").is_not_null())
    elif dimension == "time_series":
        result = filtered_data.group_by("filter_date").agg([
            pl.col("record_count").sum().alias("record_count"),
            pl.col("biosample_count").sum().alias("biosample_count")
        ]).rename({"filter_date": "value"}).filter(pl.col("value").is_not_null())

    return result.sort("record_count", descending=True)


def fill_missing_dates_polars(time_series_df: pl.DataFrame) -> pl.DataFrame:
    if len(time_series_df) == 0:
        # return empty dataframe with correct schema
        return pl.DataFrame({
            'Date': [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            'Organism Count': [0, 0],
            'BioSample IDs Count': [0, 0]
        })

    # get date range
    min_date = time_series_df['Date'].min()
    max_date = time_series_df['Date'].max()

    date_range = pl.date_range(
        min_date,
        max_date,
        interval="1d",
        eager=True
    )

    full_dates_df = pl.DataFrame({'Date': date_range})

    # join and fill missing values
    result = full_dates_df.join(
        time_series_df,
        on='Date',
        how='left'
    ).fill_null(0)

    return result


def create_time_series_colors_polars(time_series_df: pl.DataFrame, stored_date) -> pl.DataFrame:
    highlight_color = 'rgb(255, 99, 71)'
    default_color = 'rgb(0, 102, 255)'

    if stored_date is None:
        return time_series_df.with_columns(pl.lit(default_color).alias('bar_color'))

    if isinstance(stored_date, str):
        # single date selection
        stored_date_pl = pl.lit(stored_date).str.to_date()
        return time_series_df.with_columns(
            pl.when(pl.col('Date') == stored_date_pl)
            .then(pl.lit(highlight_color))
            .otherwise(pl.lit(default_color))
            .alias('bar_color')
        )
    elif isinstance(stored_date, dict):
        # date range selection
        start_date = stored_date.get("start")
        end_date = stored_date.get("end")

        condition = pl.lit(True)

        if start_date:
            start_date_pl = pl.lit(start_date).str.to_date()
            condition = condition & (pl.col('Date') >= start_date_pl)

        if end_date:
            end_date_pl = pl.lit(end_date).str.to_date()
            condition = condition & (pl.col('Date') <= end_date_pl)

        return time_series_df.with_columns(
            pl.when(condition)
            .then(pl.lit(highlight_color))
            .otherwise(pl.lit(default_color))
            .alias('bar_color')
        )

    return time_series_df.with_columns(pl.lit(default_color).alias('bar_color'))


# generic functions
def limit_grouped_data(df: pl.DataFrame, col: str, top_n: int = 10) -> pl.DataFrame:
    if len(df) <= top_n:
        return df

    top_rows = df.head(top_n)
    other_rows = df.slice(top_n, len(df) - top_n)

    # "Others"
    others_count = other_rows["record_count"].sum()
    others_biosample_count = other_rows["biosample_count"].sum()

    others_row = pl.DataFrame({
        "value": ["Others"],
        "record_count": [others_count],
        "biosample_count": [others_biosample_count]
    })

    return pl.concat([top_rows, others_row], how="vertical")


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


def build_pie_optimized(df: pl.DataFrame, col_name: str, selected_val: str, type='piechart'):
    disabled_color = '#cccccc'
    palette = px.colors.qualitative.Plotly
    hole = 0.3 if type == 'doughnut_chart' else 0
    piechart_title = col_name.replace("_", " ").title()

    # display column for wrapped text
    df_display = df.with_columns(
        pl.col("value").map_elements(wrap_text_for_display, return_dtype=pl.Utf8).alias("value_display")
    )

    # pie labels
    total = df_display["record_count"].sum()
    if total > 0:
        df_display = df_display.with_columns([
            (pl.col("record_count") / total * 100).alias("percentage")
        ])
        df_display = df_display.with_columns([
            pl.when(pl.col("percentage") > 5)
            .then(pl.col("percentage").map_elements(lambda x: f"{x:.1f}%", return_dtype=pl.Utf8))
            .otherwise(pl.lit(""))
            .alias("text")
        ])
    else:
        df_display = df_display.with_columns([
            pl.lit(0.0).alias("percentage"),
            pl.lit("").alias("text")
        ])

    df_pandas = df_display.to_pandas()

    pie = px.pie(
        data_frame=df_pandas,
        names="value_display",
        values="record_count",
        title=piechart_title,
        hole=hole,
        hover_data={"biosample_count": True, "value": False}
    )

    # ordered lists for custom data, colors, and pulls to match the pie chart order
    ordered_display_names = list(pie.data[0]['labels'])

    # lookup from display name to original value and percentage
    lookup_df = df_display.select(
        pl.col("value_display"),
        pl.col("value").alias("original_value"),
        pl.col("biosample_count").alias("biosample_count"),
        pl.col("percentage").alias("percentage")
    ).to_dicts()

    lookup_map = {item["value_display"]: (item["biosample_count"], item["original_value"], item["percentage"]) for item
                  in lookup_df}

    reordered_custom_data = []
    reordered_colors = []
    reordered_pulls = []

    for i, display_name in enumerate(ordered_display_names):
        original_bs_count, original_val, percentage = lookup_map.get(display_name, (None, None, 0))
        if original_val is not None:
            reordered_custom_data.append([original_bs_count, original_val, percentage])  # Include percentage
            reordered_colors.append(
                disabled_color if original_val == "Others" else palette[i % len(palette)])
            reordered_pulls.append(0.1 if original_val == selected_val else 0)
        else:
            reordered_custom_data.append([None, None, 0])
            reordered_colors.append(disabled_color)
            reordered_pulls.append(0)

    pie.data[0].customdata = reordered_custom_data
    pie.data[0].marker.colors = reordered_colors
    pie.data[0].pull = reordered_pulls

    pie.update_traces(
        text=df_pandas["text"],
        textinfo="text",
        textposition="inside",
        hovertemplate="<b>%{customdata[0][1]}</b><br>Percentage: %{customdata[0][2]:.1f}%<extra></extra>"
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
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    return pie


def load_table_data(project_name: str, filters: dict, page_current: int, page_size: int, tab: str):

    offset = page_current * page_size

    # table data using views
    data_query = build_table_data_query(project_name, tab, filters, offset, page_size)
    count_query = build_table_count_query(project_name, tab, filters)

    data_job = client.query(data_query)
    data_df = data_job.to_dataframe()

    # pagination count
    count_job = client.query(count_query)
    count_result = count_job.to_dataframe()
    total_count = count_result['total_count'].iloc[0] if not count_result.empty else 0

    # organism links
    if not data_df.empty:
        link_prefix = PORTAL_URL_PREFIX.get(project_name, "")
        url_param = "tax_id" if project_name in ["erga", "gbdp"] else "organism"

        def create_organism_link(row):
            scientific_name = row.get('scientific_name', '')
            if not scientific_name:
                return ''

            if url_param == 'tax_id':
                url_value = str(row.get('tax_id', ''))
            else:
                url_value = urllib.parse.quote(scientific_name, safe='')

            return f'[{scientific_name}]({link_prefix}{url_value})'

        data_df['organism_link'] = data_df.apply(create_organism_link, axis=1)

    return data_df, total_count


def layout(**kwargs):
    project_name = kwargs.get("projectName", "dtol")

    # load aggregated data
    load_metadata_aggregated_data(project_name)
    load_rawdata_aggregated_data(project_name)

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

        # filter selection text
        html.Div(
            id="metadata-filter-selection",
            className="my-4 text-center fw-bold text-dark"
        ),

        # charts row
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
        ], className="mb-4"),

        # Reset button
        html.Div(
            dbc.Button(
                "Reset All",
                id="reset-all-dashboards",
                color="danger",
                className="mb-3"
            ),
            className="d-flex justify-content-end"
        ),

        # table section
        dbc.Row([
            dbc.Col([
                # data table
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
                        css=[
                            dict(selector="p", rule="margin: 0; text-align: center"),
                            dict(selector="a", rule="text-decoration: none")
                        ],
                        page_current=0,
                        page_size=10,
                        page_action="custom",
                        style_table={'overflowX': 'auto'}
                    )
                )
            ], md=12, id="col-table")
        ], className="my-3")
    ])

    tab2_content = html.Div([
        dcc.Store(id="rawdata-stored-selection",
                  data={"instrument_platform": None, "instrument_model": None, "library_construction_protocol": None}),
        dcc.Store(id="rawdata-selected-date", data=None),
        dcc.Store(id="tab2-project-store", data=project_name),

        # filter selection text
        html.Div(
            id="rawdata-filter-selection",
            className="my-4 text-center fw-bold text-dark"
        ),

        # charts row
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
        ], className="mb-4"),

        # date range picker
        dbc.Row([
            dbc.Col(width=8),
            dbc.Col(
                dcc.DatePickerRange(
                    id='date-range-picker',
                    display_format='DD/MM/YYYY',
                    month_format='MMM YYYY',
                    clearable=True,
                    style={"display": "inline-block"}
                ),
                width=4,
                className="text-end mb-3"
            ),
        ]),

        # time series chart
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
        ], className="mb-3"),

        # Reset button
        html.Div(
            dbc.Button(
                "Reset All",
                id="tab2-reset-all-dashboards",
                color="danger",
                className="mb-3"
            ),
            className="d-flex justify-content-end"
        ),

        # table section
        dbc.Row([
            dbc.Col([
                # data table
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
                        css=[
                            dict(selector="p", rule="margin: 0; text-align: center"),
                            dict(selector="a", rule="text-decoration: none")
                        ],
                        page_current=0,
                        page_size=10,
                        page_action="custom",
                        style_table={'overflowX': 'auto'}
                    )
                )
            ], md=12, id="tab2-col-table")
        ], className="my-3")
    ])

    return dbc.Container([
        dcc.Store(id="active-tab-store", data="metadata"),
        dcc.Tabs(
            id="main-tabs",
            value="metadata",
            children=[
                dcc.Tab(label='Metadata', value="metadata", children=[tab1_content]),
                dcc.Tab(label='Raw Data', value="rawdata", children=[tab2_content]),
            ]
        )
    ], fluid=True)


@callback(
    Output("active-tab-store", "data"),
    Input("main-tabs", "value")
)
def update_active_tab(active_tab):
    return active_tab

@callback(
    Output("metadata-filter-selection", "children"),
    Input("stored-selection", "data"),
    Input("active-tab-store", "data")
)
def update_selection_text(selected, active_tab):

    if active_tab != "metadata":
        raise PreventUpdate

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
                html.Span(f"{value}", className="text-selection-blue"),
                html.Span(" | ")
            ])

    return parts[:-1] if parts else default_msg


@callback(
    Output("dashboard-datatable-paging", "page_current", allow_duplicate=True),
    Input("reset-all-dashboards", "n_clicks"),
    Input('pie-sex', 'clickData'),
    Input('pie-lifestage', 'clickData'),
    Input('pie-habitat', 'clickData'),
    Input("active-tab-store", "data"),
    State("dashboard-datatable-paging", "page_current"),
    prevent_initial_call=True
)
def reset_metadata_paging(n_clicks, sex_click, lifestage_click, habitat_click, active_tab, current_page):

    if active_tab != "metadata":
        raise PreventUpdate

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
    Input("active-tab-store", "data"),
    State("stored-selection", "data")
)
def build_metadata_charts(pie_sex_click, pie_lifestage_click, pie_habitat_click,
                          project_name, selected_pie_input,
                          reset_n_clicks, active_tab, stored_selection):

    if active_tab != "metadata":
        raise PreventUpdate

    triggered_id = ctx.triggered_id

    stored_selection = selected_pie_input or {"sex": None, "lifestage": None, "habitat": None}

    def extract_label(click_data):
        if not click_data:
            return None
        return click_data["points"][0]["customdata"][1]

    if triggered_id == "pie-sex":
        label = extract_label(pie_sex_click)
        if label and label != "Others":
            stored_selection["sex"] = label
    elif triggered_id == "pie-lifestage":
        label = extract_label(pie_lifestage_click)
        if label and label != "Others":
            stored_selection["lifestage"] = label
    elif triggered_id == "pie-habitat":
        label = extract_label(pie_habitat_click)
        if label and label != "Others":
            stored_selection["habitat"] = label
    elif triggered_id == "reset-all-dashboards":
        stored_selection = {"sex": None, "lifestage": None, "habitat": None}

    # build charts with pre-aggregated data
    pies = {}
    for dimension in ["sex", "lifestage", "habitat"]:
        chart_data = get_metadata_chart_data(project_name, dimension, stored_selection)
        limited_data = limit_grouped_data(chart_data, dimension)
        pies[dimension] = build_pie_optimized(limited_data, dimension, stored_selection[dimension])

    return pies["sex"], pies["lifestage"], pies["habitat"], stored_selection


@callback(
    Output("dashboard-datatable-paging", "data"),
    Output("dashboard-datatable-paging", "page_count"),
    Output("dashboard-datatable-paging", "page_current"),
    Input('dashboard-datatable-paging', "page_current"),
    Input('dashboard-datatable-paging', "page_size"),
    Input("project-store", "data"),
    Input("stored-selection", "data"),
    Input("pie-sex", "clickData"),
    Input("pie-lifestage", "clickData"),
    Input("pie-habitat", "clickData"),
    Input("reset-all-dashboards", "n_clicks"),
    Input("active-tab-store", "data"),
)
def build_metadata_table(page_current: int, page_size: int, project_name: str, stored_selection: dict,
                         pie_sex_click, pie_lifestage_click, pie_habitat_click, reset_clicks, active_tab):

    if active_tab != "metadata":
        raise PreventUpdate

    if page_current is None or page_size is None or not project_name:
        raise PreventUpdate

    stored_selection = stored_selection or {"sex": None, "lifestage": None, "habitat": None}

    table_data, total_count = load_table_data(project_name, stored_selection, page_current, page_size, "metadata")

    # pagination
    total_pages = max(math.ceil(total_count / page_size), 1)
    page_current = min(page_current, total_pages - 1)

    # organism count header
    filters_applied = any(v for v in stored_selection.values() if v)
    if filters_applied:
        filter_text = "matching your selection"
    else:
        filter_text = "in total"


    return (
        table_data.to_dict('records'),
        total_pages,
        page_current
    )

# ** TAB 2 **#
@callback(
    Output("rawdata-filter-selection", "children"),
    Input("rawdata-stored-selection", "data"),
    Input("rawdata-selected-date", "data"),
    Input("active-tab-store", "data")
)
def update_rawdata_selection_text(selected_piecharts, selected_date, active_tab):

    if active_tab != "rawdata":
        raise PreventUpdate

    default_msg = "Please click on the charts to filter the data"
    if not selected_piecharts and not selected_date:
        return default_msg

    parts = []
    for key in ["instrument_platform", "instrument_model", "library_construction_protocol"]:
        value = selected_piecharts.get(key) if selected_piecharts else None
        if value:
            label = key.replace("_", " ").title()
            parts.extend([
                html.Span(f"{label}: "),
                html.Span(f"{value}", className="text-selection-blue"),
                html.Span(" | ")
            ])

    if selected_date:
        if isinstance(selected_date, str):  # single date selection
            parts.extend([
                html.Span(f"Date: "),
                html.Span(f"{selected_date}", className="text-selection-blue"),
                html.Span(" | ")
            ])
        elif isinstance(selected_date, dict):  # date range selection
            start_date = selected_date.get("start")
            end_date = selected_date.get("end")
            if start_date and end_date:
                parts.extend([
                    html.Span(f"Date Range: "),
                    html.Span(f"{start_date} to {end_date}", className="text-selection-blue"),
                    html.Span(" | ")
                ])
            elif start_date:
                parts.extend([
                    html.Span(f"Start Date: "),
                    html.Span(f"{start_date}", className="text-selection-blue"),
                    html.Span(" | ")
                ])
            elif end_date:
                parts.extend([
                    html.Span(f"End Date: "),
                    html.Span(f"{end_date}", className="text-selection-blue"),
                    html.Span(" | ")
                ])

    return parts[:-1] if parts else default_msg


@callback(
    Output("tab2-dashboard-datatable-paging", "page_current", allow_duplicate=True),
    Input("tab2-reset-all-dashboards", "n_clicks"),
    Input('pie-instrument-platform', 'clickData'),
    Input('pie-instrument-model', 'clickData'),
    Input('pie-library-construction-protocol', 'clickData'),
    Input('time-series-graph', 'clickData'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
    Input("active-tab-store", "data"),
    State("tab2-dashboard-datatable-paging", "page_current"),
    prevent_initial_call=True
)
def reset_rawdata_paging(n_clicks, instrument_platform_click, instrument_model_click,
                         library_construction_protocol_click, time_series_click,
                         start_date, end_date, active_tab, current_page):

    if active_tab != "rawdata":
        raise PreventUpdate

    if current_page != 0:
        return 0
    raise dash.exceptions.PreventUpdate



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
    Input("active-tab-store", "data"),

    State('rawdata-stored-selection', 'data'),
    State('rawdata-selected-date', 'data')
)
def build_rawdata_charts(instrument_platform_click, instrument_model_click,
                         library_construction_protocol_click, time_series_click,
                         picker_start_date, picker_end_date,
                         project_name, selected_charts_input, selected_date_input,
                         reset_n_clicks, active_tab, stored_selection, stored_date):

    if active_tab != "rawdata":
        raise PreventUpdate

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
        if label and label != "Others":
            stored_selection["instrument_platform"] = label
    elif triggered_id == "pie-instrument-model":
        label = extract_label(instrument_model_click)
        if label and label != "Others":
            stored_selection["instrument_model"] = label
    elif triggered_id == "pie-library-construction-protocol":
        label = extract_label(library_construction_protocol_click)
        if label and label != "Others":
            stored_selection["library_construction_protocol"] = label
    elif triggered_id == "time-series-graph":
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

    # filters for cross-filtering
    filters = stored_selection.copy()
    if stored_date:
        filters["date_filter"] = stored_date

    # charts
    pies = {}
    for dimension in ["instrument_platform", "instrument_model", "library_construction_protocol"]:
        chart_data = get_rawdata_chart_data(project_name, dimension, filters)
        limited_data = limit_grouped_data(chart_data, dimension)
        pies[dimension] = build_pie_optimized(limited_data, dimension, stored_selection[dimension],
                                                      'doughnut_chart')

    # time series chart
    time_series_data = get_rawdata_chart_data(project_name, "time_series",
                                              {k: v for k, v in filters.items() if k != "date_filter"})

    # time series data
    if len(time_series_data) > 0:
        # check if 'value' is date type or string
        if time_series_data.schema['value'] == pl.Date:
            # date - rename
            time_series_df = time_series_data.with_columns([
                pl.col("value").alias("Date"),
                pl.col("record_count").alias("Organism Count"),
                pl.col("biosample_count").alias("BioSample IDs Count")
            ]).sort("Date")
        else:
            # string
            time_series_df = time_series_data.with_columns([
                pl.col("value").str.strptime(pl.Date, format="%Y-%m-%d").alias("Date"),
                pl.col("record_count").alias("Organism Count"),
                pl.col("biosample_count").alias("BioSample IDs Count")
            ]).sort("Date")
    else:
        # empty dataframe if no data
        time_series_df = pl.DataFrame({
            'Date': [],
            'Organism Count': [],
            'BioSample IDs Count': []
        }, schema={
            'Date': pl.Date,
            'Organism Count': pl.Int64,
            'BioSample IDs Count': pl.Int64
        })

    time_series_df = fill_missing_dates_polars(time_series_df)

    time_series_df = create_time_series_colors_polars(time_series_df, stored_date)

    # convert to pandas for plotly graph
    time_series_pandas = time_series_df.to_pandas()

    # time series chart
    customdata = time_series_pandas[['Organism Count', 'BioSample IDs Count']].values

    time_series_fig = px.bar(
        time_series_pandas,
        x='Date',
        y='Organism Count',
        title='Record Count Over Time',
    )

    time_series_fig.update_traces(
        marker_color=time_series_pandas['bar_color'],
        marker_line_width=0,
        customdata=customdata,
        hovertemplate=(
            "<b>Date: %{x|%d %b %Y}</b><br>"
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
        pies["instrument_platform"],
        pies["instrument_model"],
        pies["library_construction_protocol"],
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
    Input("active-tab-store", "data"),
)
def build_rawdata_table(page_current: int, page_size: int, project_name: str,
                        selected_pie_data: dict, selected_date,
                        platform_click, model_click, protocol_click, time_click,
                        start_date, end_date, reset_clicks, active_tab):

    if active_tab != "rawdata":
        raise PreventUpdate

    if page_current is None or page_size is None or not project_name:
        raise PreventUpdate

    filters = selected_pie_data or {}
    if selected_date:
        filters["date_filter"] = selected_date

    table_data, total_count = load_table_data(project_name, filters, page_current, page_size, "rawdata")

    # pagination
    total_pages = max(math.ceil(total_count / page_size), 1)
    page_current = min(page_current, total_pages - 1)

    # organism count header
    filters_applied = any(v for v in filters.values() if v)
    if filters_applied:
        filter_text = "matching your selection"
    else:
        filter_text = "in total"


    return (
        table_data.to_dict('records'),
        total_pages,
        page_current
    )