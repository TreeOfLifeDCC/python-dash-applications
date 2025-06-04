import math
import urllib
import pandas as pd
import plotly.express as px
import google.auth
from gcsfs import GCSFileSystem
import dash
import fsspec
from dash import dcc, callback, Output, Input, dash_table, State, html, ctx
import dash_bootstrap_components as dbc

dash.register_page(
    __name__,
    path_template="/dashboards",
    title="Dashboards",
)

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


def limit_grouped_data(grouped_data, category, raw_data):
    top_n = 10
    if len(grouped_data) > top_n:
        top_rows = grouped_data.iloc[:top_n].copy()
        other_rows = grouped_data.iloc[top_n:]

        def concat_unique(col):
            unique_rows_values_str = ", ".join(set(filter(None, col.astype(str))))
            set_values = set(item.strip() for item in unique_rows_values_str.split(','))
            return ", ".join(set_values)

        # retrieve biosample ids from other_rows
        other_biosample_ids = set()
        for val in other_rows["organisms.biosample_id"]:
            other_biosample_ids.update([x.strip() for x in str(val).split(",") if x.strip()])

        # filter the raw_data using biosample ids from "Others"
        filtered_df = raw_data[raw_data["organisms.biosample_id"].isin(other_biosample_ids)]

        grouped_df = filtered_df.groupby("organisms.biosample_id").agg({
            "common_name": "first",
            "current_status": "first",
            "symbionts_status": "first",
            "organisms.organism": "first"
        }).reset_index()

        grouped_df["organism_link"] = grouped_df["organisms.organism"].apply(
            lambda x: f"[{x}](https://portal.darwintreeoflife.org/data/{urllib.parse.quote(str(x))})" if pd.notna(
                x) else ""
        )

        grouped_df = grouped_df.drop_duplicates(subset=[
            "organism_link",
            "common_name",
            "current_status",
            "symbionts_status"
        ])

        # aggregate values for the "Others" row
        other_row = {
            category: "Others",
            "organisms.biosample_id": concat_unique(other_rows["organisms.biosample_id"]),
            "organisms.organism": concat_unique(other_rows["organisms.organism"]),
            "organisms.sex": concat_unique(other_rows["organisms.sex"]),
            "current_status": concat_unique(other_rows["current_status"]),
            "Record Count": len(grouped_df),
        }
        top_rows = pd.concat([top_rows, pd.DataFrame([other_row])], ignore_index=True)
        return top_rows

    return grouped_data


def generate_grouped_data(df, group_by_col):
    df = df[df[group_by_col].notna()]
    grouped_data = df.groupby(group_by_col).agg({
        "organisms.biosample_id": lambda x: ", ".join(set(filter(None, x))),
        "organisms.organism": lambda x: ", ".join(set(filter(None, x))),
        "organisms.sex": lambda x: ", ".join(set(filter(None, x))),
        "current_status": lambda x: ", ".join(set(filter(None, x))),
        "symbionts_status": lambda x: ", ".join(set(filter(None, x))),
    }).reset_index()
    grouped_data["Record Count"] = df.groupby(group_by_col)["organisms.organism"].nunique().values
    grouped_data["BioSample IDs Count"] = df.groupby(group_by_col)["organisms.biosample_id"].nunique().values
    grouped_data = grouped_data.sort_values(by="Record Count", ascending=False).reset_index(drop=True)
    return grouped_data


def preprocess_chunk(df_nested):
    # organisms repeated record
    df_exploded = df_nested.explode('organisms').reset_index(drop=True)
    organisms_df = pd.json_normalize(df_exploded['organisms'].dropna(), sep='.')
    organisms_df.columns = [f'organisms.{col}' for col in organisms_df.columns]
    df_exploded = df_exploded.drop(columns=['organisms']).reset_index(drop=True)
    df = pd.concat([df_exploded, organisms_df], axis=1)

    # raw_data repeated record
    df = df.explode('raw_data').reset_index(drop=True)
    raw_data_df = pd.json_normalize(df['raw_data'].dropna(), sep='.')
    raw_data_df.columns = [f'raw_data.{col}' for col in raw_data_df.columns]
    df = df.drop(columns=['raw_data']).reset_index(drop=True)
    df = pd.concat([df, raw_data_df], axis=1)

    return df


def load_data(project_name):
    if project_name not in DATASETS:
        DATASETS[project_name] = {}
        gcs_pattern = PROJECT_PARQUET_MAP.get(project_name, PROJECT_PARQUET_MAP["dtol"])
        fs = fsspec.filesystem("gcs")
        matching_files = fs.glob(gcs_pattern)

        if not matching_files:
            raise FileNotFoundError(f"No Parquet files found for pattern: {gcs_pattern}")

        processed_chunks = []
        for file in matching_files:
            with fs.open(file) as f:
                raw_df = pd.read_parquet(f, columns=[
                    "organisms", "raw_data", "current_status", "symbionts_status", "common_name"
                ], engine="pyarrow")

                processed_df = preprocess_chunk(raw_df)
                processed_chunks.append(processed_df)

        final_df = pd.concat(processed_chunks, ignore_index=True)

        # for tab1 piecharts
        final_df["sex"] = final_df["organisms.sex"].astype(str)
        final_df["lifestage"] = final_df["organisms.lifestage"].astype(str)
        final_df["habitat"] = final_df["organisms.habitat"].astype(str)

        # for tab2 piecharts
        final_df["instrument_platform"] = final_df["raw_data.instrument_platform"].astype(str)
        final_df["instrument_model"] = final_df["raw_data.instrument_model"].astype(str)
        final_df["library_construction_protocol"] = final_df["raw_data.library_construction_protocol"].astype(str)

        # Convert 'first_public' to datetime
        final_df['first_public'] = pd.to_datetime(final_df['raw_data.first_public'], errors='coerce')

        final_df["organism_link"] = final_df["organisms.organism"].apply(
            lambda x: f"[{x}](https://portal.darwintreeoflife.org/data/{urllib.parse.quote(str(x))})" if pd.notna(
                x) else ""
        )

        DATASETS[project_name]["raw_data"] = final_df.copy()


def build_pie(df, col_name, selected_val, type='piechart'):
    disabled_color = '#cccccc'
    palette = px.colors.qualitative.Plotly
    hole = 0.3 if type == 'doughnut_chart' else 0
    piechart_title = col_name.replace("_", " ").title()

    # wrap long text for legend display only
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

    # add legend display column for wrapped text
    df_with_display = df.copy()
    df_with_display[col_name + '_display'] = df_with_display[col_name].apply(wrap_text_for_display)

    pie = px.pie(
        data_frame=df_with_display.dropna(subset=["Record Count", col_name]),
        names=col_name + '_display',  # Use wrapped text for display
        values="Record Count",
        title=piechart_title,
        hole=hole,
        hover_data={"BioSample IDs Count": True}
    )

    colors = [
        disabled_color if val == "Others" else palette[i % len(palette)]
        for i, val in enumerate(df[col_name])  # Use original data for colors
    ]

    # Map display names back to original names for click functionality
    # Store original values as custom data for the click handler
    # Get original values and biosample counts
    original_values = df[col_name].tolist()
    biosample_counts = df["BioSample IDs Count"].tolist()

    # custom data array - each slice gets [biosample_count, original_value]
    custom_data = []
    for biosample_count, original_val in zip(biosample_counts, original_values):
        custom_data.append([biosample_count, original_val])

    pie.data[0].customdata = custom_data

    # update hover template to use original names
    pie.update_traces(
        text=df["text"],
        textinfo="text",
        textposition="inside",
        pull=[0.1 if v == selected_val else 0 for v in df[col_name]],  # Use original data for selection
        marker=dict(colors=colors),
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
        margin=dict(l=20, r=180, t=50, b=20)
    )

    return pie


def generate_pie_labels(df):
    total = df["Record Count"].sum()
    df["Percentage"] = df["Record Count"] / total * 100
    df["text"] = df["Percentage"].apply(lambda x: f"{x:.1f}%" if x > 5 else "")


def layout(**kwargs):
    project_name = kwargs.get("projectName", "dtol")

    # lazy load project's dataset
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
            dbc.Col(dcc.Graph(id="pie-sex"), width=4),
            dbc.Col(dcc.Graph(id="pie-lifestage"), width=4),
            dbc.Col(dcc.Graph(id="pie-habitat"), width=4),
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
        dcc.Store(id="tab2-project-store", data=project_name),
        dcc.Store(id="rawdata-selected-date"),

        html.Div(id="rawdata-filter-selection",
                 style={"marginBottom": "30px", "marginTop": "40px", "textAlign": "center",
                        "fontWeight": "bold", "color": "#394959"}),

        dbc.Row([
            dbc.Col(dcc.Graph(id="pie-instrument-platform"), width=4),
            dbc.Col(dcc.Graph(id="pie-instrument-model"), width=4),
            dbc.Col(dcc.Graph(id="pie-library-construction-protocol"), width=4),
        ], style={"marginBottom": "40px"}),

        dbc.Row([
            dbc.Col(dcc.Graph(id="time-series-graph"), width=12)
        ], style={"marginBottom": "40px"}),

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
        parts.extend([
            html.Span(f"Date: "),
            html.Span(f"{selected_date}", style={"color": "#8fa8c2"}),
            html.Span(" | ")
        ])

    return parts[:-1] if parts else default_msg


@callback(
    Output("dashboard-datatable-paging", "page_current", allow_duplicate=True),
    Input("reset-all-dashboards", "n_clicks"),
    Input('pie-sex', 'clickData'),
    Input('pie-lifestage', 'clickData'),
    Input('pie-habitat', 'clickData'),
    State("dashboard-datatable-paging", "page_current"),  # Add current page as state
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
    State("tab2-dashboard-datatable-paging", "page_current"),  # Add current page as state
    prevent_initial_call=True
)
def reset_rawdata_paging(n_clicks, instrument_platform_click, instrument_model_click,
                         library_construction_protocol_click, time_series_click, current_page):
    if current_page != 0:
        return 0
    raise dash.exceptions.PreventUpdate


@dash.callback(
    Output('pie-sex', 'figure'),
    Output('pie-lifestage', 'figure'),
    Output('pie-habitat', 'figure'),
    Output("stored-selection", 'data'),

    Input('pie-sex', 'clickData'),
    Input('pie-lifestage', 'clickData'),
    Input('pie-habitat', 'clickData'),
    Input("project-store", "data"),
    Input("stored-selection", "data"),
    Input("reset-all-dashboards", "n_clicks"),
    State("stored-selection", 'data')
)
def build_metadata_charts(pie_sex_click, pie_lifestage_click, pie_habitat_click, project_name, selected_pie_input,
                          reset_n_clicks, stored_selection):
    triggered_id = ctx.triggered_id

    stored_selection = selected_pie_input or {"sex": None, "lifestage": None, "habitat": None}

    def extract_label(click_data):
        if not click_data:
            return None
        return click_data['points'][0]['customdata'][1]

    # update filters based on trigger
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

    df_raw = DATASETS[project_name]["raw_data"]

    # we need to keep the selected piechart's slices as they were when first selected
    def filter_df(pie_key):
        df = df_raw.copy()
        for key, value in stored_selection.items():
            if key != pie_key and value:
                df = df[df[key] == value]
        return df

    for dim in ["sex", "lifestage", "habitat"]:
        df_filtered = filter_df(dim)
        grouped = generate_grouped_data(df_filtered, dim)
        updated_grouped_data = limit_grouped_data(grouped, dim, df_filtered)
        DATASETS[project_name][dim] = {"df_data": df_filtered, "grouped_data": updated_grouped_data}
        generate_pie_labels(updated_grouped_data)

    df_sex = DATASETS[project_name]["sex"]["grouped_data"]
    df_lifestage = DATASETS[project_name]["lifestage"]["grouped_data"]
    df_habitat = DATASETS[project_name]["habitat"]["grouped_data"]

    pie_sex = build_pie(df_sex, "sex", stored_selection["sex"])
    pie_lifestage = build_pie(df_lifestage, "lifestage", stored_selection["lifestage"])
    pie_habitat = build_pie(df_habitat, "habitat", stored_selection["habitat"])

    return pie_sex, pie_lifestage, pie_habitat, stored_selection


@callback(
    Output("dashboard-datatable-paging", "data"),
    Output("dashboard-datatable-paging", "page_count"),
    Output("dashboard-datatable-paging", "page_current"),

    Input('dashboard-datatable-paging', "page_current"),
    Input('dashboard-datatable-paging', "page_size"),
    Input("project-store", "data"),
    Input("stored-selection", "data"),
)
def build_metadata_table(page_current, page_size, project_name, stored_selection):
    filtered_df = DATASETS[project_name]["raw_data"].copy()
    stored_selection = stored_selection or {"sex": None, "lifestage": None, "habitat": None}

    # data table filtering
    for key, value in stored_selection.items():
        if value:
            filtered_df = filtered_df[filtered_df[key] == value]

    # we have to do the gouping again because of the unnesting of raw_data which is a repeated record - array of structs
    # see BigQuery metadata schema
    grouped_df = filtered_df.groupby('organisms.biosample_id').agg({
        'common_name': 'first',
        'current_status': 'first',
        'symbionts_status': 'first',
        'organisms.organism': 'first',
    }).reset_index()

    grouped_df["organism_link"] = grouped_df["organisms.organism"].apply(
        lambda x: f"[{x}](https://portal.darwintreeoflife.org/data/{urllib.parse.quote(str(x))})" if pd.notna(x) else ""
    )

    grouped_df = grouped_df.drop_duplicates(subset=[
        "organism_link",
        "common_name",
        "current_status",
        "symbionts_status"
    ])

    grouped_df = grouped_df.sort_values(by="current_status", ascending=True)

    # pagination
    total_pages = max(math.ceil(len(grouped_df) / page_size), 1)
    page_current = min(page_current, total_pages - 1)

    return (
        grouped_df.iloc[page_current * page_size:(page_current + 1) * page_size].to_dict("records"),
        total_pages,
        page_current
    )


# **** Tab 2 ***#
@dash.callback(
    Output('pie-instrument-platform', 'figure'),
    Output('pie-instrument-model', 'figure'),
    Output('pie-library-construction-protocol', 'figure'),
    Output('time-series-graph', 'figure'),  # NEW OUTPUT
    Output('rawdata-stored-selection', 'data'),
    Output('rawdata-selected-date', 'data'),

    Input('pie-instrument-platform', 'clickData'),
    Input('pie-instrument-model', 'clickData'),
    Input('pie-library-construction-protocol', 'clickData'),
    Input('time-series-graph', 'clickData'),  # NEW INPUT
    Input("tab2-project-store", "data"),
    Input("rawdata-stored-selection", "data"),
    Input("rawdata-selected-date", "data"),  # NEW INPUT
    Input("tab2-reset-all-dashboards", "n_clicks"),

    State('rawdata-stored-selection', 'data'),
    State('rawdata-selected-date', 'data')
)
def build_rawdata_charts(instrument_platform_click, instrument_model_click,
                         library_construction_protocol_click, time_series_click,
                         project_name, selected_charts_input, selected_date_input,
                         reset_n_clicks, stored_selection, stored_date):
    triggered_id = ctx.triggered_id

    stored_selection = selected_charts_input or {"instrument_platform": None, "instrument_model": None,
                                                 "library_construction_protocol": None}

    stored_date = selected_date_input or None

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
    elif triggered_id == "tab2-reset-all-dashboards":
        stored_selection = {"instrument_platform": None, "instrument_model": None,
                            "library_construction_protocol": None}
        stored_date = None
    elif triggered_id == "time-series-graph":
        stored_date = time_series_click['points'][0]['x'] if time_series_click else None

    df_raw = DATASETS[project_name]["raw_data"]

    # Apply filtering for each pie chart based on selection of the other two
    def filter_df(pie_key):
        df = df_raw.copy()
        for key, value in stored_selection.items():
            if key != pie_key and value:
                df = df[df[key] == value]
        if stored_date:
            date_obj = pd.to_datetime(stored_date).date()
            df = df[df['first_public'].dt.date == date_obj]
        return df

        # Filter and update pie datasets

    for key in ["instrument_platform", "instrument_model", "library_construction_protocol"]:
        df_sub = filter_df(key)
        grouped_data = generate_grouped_data(df_sub, key)
        updated_grouped_data = limit_grouped_data(grouped_data, key, df_sub)
        DATASETS[project_name][key] = {"df_data": df_sub, "grouped_data": updated_grouped_data}
        generate_pie_labels(updated_grouped_data)

    df_instrument_platform = DATASETS[project_name]["instrument_platform"]["grouped_data"]
    df_instrument_model = DATASETS[project_name]["instrument_model"]["grouped_data"]
    df_library_construction_protocol = DATASETS[project_name]["library_construction_protocol"]["grouped_data"]

    disabled_color = '#cccccc'
    palette = px.colors.qualitative.Plotly

    pie_instrument_platform = build_pie(df_instrument_platform, "instrument_platform",
                                        stored_selection["instrument_platform"], 'doughnut_chart')

    pie_instrument_model = build_pie(df_instrument_model, "instrument_model",
                                     stored_selection["instrument_model"], 'doughnut_chart')

    pie_instrument_construction_protocol = build_pie(df_library_construction_protocol,
                                                     "library_construction_protocol",
                                                     stored_selection["library_construction_protocol"],
                                                     'doughnut_chart')

    # time series section - always use the full timeline for x-axis
    df_all_dates = generate_grouped_data(df_raw, "first_public")
    df_all_dates['first_public'] = pd.to_datetime(df_all_dates['first_public']).dt.date
    df_all_dates = df_all_dates.rename(columns={
        'first_public': 'Date',
        'Record Count': 'Total Organism Count',
        'BioSample IDs Count': 'Total BioSample IDs Count'
    })

    # Now generate the filtered version of the time series
    df_filtered = df_raw.copy()

    # Apply all pie selections
    for key, value in stored_selection.items():
        if value:
            df_filtered = df_filtered[df_filtered[key] == value]

    # Apply date filter only to pie chart data, not time series (important)
    filtered_ts = generate_grouped_data(df_filtered, "first_public")
    filtered_ts['first_public'] = pd.to_datetime(filtered_ts['first_public']).dt.date
    filtered_ts = filtered_ts.rename(columns={
        'first_public': 'Date',
        'Record Count': 'Organism Count',
        'BioSample IDs Count': 'BioSample IDs Count'
    })

    # Merge to retain all original dates
    grouped_ts = pd.merge(df_all_dates[['Date']], filtered_ts, on='Date', how='left').fillna(0)

    # Highlight selected date bar
    highlight_color = 'rgb(255, 99, 71)'  # red
    default_color = 'rgb(0, 102, 255)'  # blue
    grouped_ts['bar_color'] = grouped_ts['Date'].apply(
        lambda d: highlight_color if stored_date and pd.to_datetime(stored_date).date() == d else default_color
    )
    selected_index = (
        grouped_ts.index[grouped_ts['Date'] == pd.to_datetime(stored_date).date()].tolist()
        if stored_date else []
    )
    # Prepare customdata for hovertemplate
    customdata = grouped_ts[['Organism Count', 'BioSample IDs Count']].values

    # Create the bar chart figure
    time_series_fig = px.bar(
        grouped_ts,
        x='Date',
        y='Organism Count',
        title='Record Count Over Time',
    )

    time_series_fig.update_traces(
        marker_color=grouped_ts['bar_color'],
        selectedpoints=selected_index,
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
        stored_date
    )


@callback(
    Output("tab2-dashboard-datatable-paging", "data"),
    Output("tab2-dashboard-datatable-paging", "page_count"),
    Output("tab2-dashboard-datatable-paging", "page_current"),

    Input("tab2-dashboard-datatable-paging", "page_current"),
    Input("tab2-dashboard-datatable-paging", "page_size"),
    Input("tab2-project-store", "data"),
    Input("rawdata-stored-selection", "data"),  # {"sex": "...", "lifestage": "...", "habitat": "..."}
    Input("rawdata-selected-date", "data"),  # "YYYY-MM-DD" or {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
)
def build_rawdata_table(page_current, page_size, project_name, selected_pie_data, selected_date):
    filtered_df = DATASETS[project_name]["raw_data"].copy()

    # Apply pie chart filters
    if selected_pie_data:
        for category in ["instrument_platform", "instrument_model", "library_construction_protocol"]:
            selected_value = selected_pie_data.get(category)
            if selected_value and selected_value != "Others":
                filtered_df = filtered_df[filtered_df[category] == selected_value]

    # Apply time series date filtering
    if selected_date:
        if isinstance(selected_date, str):
            filtered_df = filtered_df[filtered_df["first_public"] == selected_date]
        elif isinstance(selected_date, dict):
            start = pd.to_datetime(selected_date.get("start"))
            end = pd.to_datetime(selected_date.get("end"))
            filtered_df = filtered_df[
                (filtered_df["first_public"] >= start) &
                (filtered_df["first_public"] <= end)
                ]

    # Group by biosample and aggregate
    grouped_df = filtered_df.groupby("organisms.biosample_id").agg({
        "common_name": "first",
        "current_status": "first",
        "symbionts_status": "first",
        "organisms.organism": "first",
    }).reset_index()

    grouped_df["organism_link"] = grouped_df["organisms.organism"].apply(
        lambda x: f"[{x}](https://portal.darwintreeoflife.org/data/{urllib.parse.quote(str(x))})" if pd.notna(x) else ""
    )

    grouped_df = grouped_df.drop_duplicates(subset=[
        "organism_link", "common_name", "current_status", "symbionts_status"
    ])

    grouped_df = grouped_df.sort_values(by="current_status", ascending=True)

    # Pagination
    total_pages = max(math.ceil(len(grouped_df) / page_size), 1)
    page_current = min(page_current, total_pages - 1)

    return (
        grouped_df.iloc[page_current * page_size:(page_current + 1) * page_size].to_dict("records"),
        total_pages,
        page_current
    )
