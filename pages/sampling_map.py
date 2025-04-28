import math
import urllib
import dash
import numpy as np
from dash import dcc, callback, Output, Input, dash_table, State, no_update, html, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import google.auth
from gcsfs import GCSFileSystem

dash.register_page(
    __name__,
    path_template="/sampling-map",
    title="Sampling Map",
)

credentials, project = google.auth.default()
fs = GCSFileSystem(token=credentials)

DATASETS = {}

# map project to parquet file path
PROJECT_PARQUET_MAP = {
    "dtol": "python_dash_data_bucket/metadata_dtol.parquet",
    "erga": "python_dash_data_bucket/metadata_erga.parquet",
    "asg": "python_dash_data_bucket/metadata_asg.parquet",
    "gbdp": "python_dash_data_bucket/metadata_gbdp.parquet"
}


def load_data(project_name):
    if project_name not in DATASETS:
        gcs_path = PROJECT_PARQUET_MAP.get(project_name, PROJECT_PARQUET_MAP["dtol"])
        with fs.open(gcs_path) as f:
            df_nested = pd.read_parquet(f, engine="pyarrow")

        df_exploded = df_nested.explode('organisms').reset_index(drop=True)

        organisms_df = pd.json_normalize(df_exploded['organisms'].dropna(), sep='.')

        # add prefix
        organisms_df.columns = [f'organisms.{col}' for col in organisms_df.columns]

        df = pd.concat([df_exploded.drop(columns=['organisms']), organisms_df], axis=1)

        def extract_unique_protocols(raw_data_entry):
            if isinstance(raw_data_entry, (list, np.ndarray)):
                protocols = {
                    item.get('library_construction_protocol')
                    for item in raw_data_entry
                    if isinstance(item, dict) and item.get('library_construction_protocol')
                }
                return ','.join(protocols) if protocols else None
            return None

        if 'raw_data' in df.columns:
            df['experiment_type'] = df['raw_data'].apply(extract_unique_protocols)

        df['experiment_type'] = df['experiment_type'].str.split(',')

        df['experiment_type'] = df['experiment_type'].apply(
            lambda x: [str(item) for item in x] if isinstance(x, list) else []
        )
        # explode the experiment_type so that we have a row for each experiment_type
        df = df.explode('experiment_type').reset_index(drop=True)

        # process dataframe data
        df["lat"] = pd.to_numeric(df["organisms.latitude"], errors='coerce')
        df["lon"] = pd.to_numeric(df["organisms.longitude"], errors='coerce')
        df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
        df["geotag"] = df["lat"].astype(str) + "," + df["lon"].astype(str)
        df["Kingdom"] = df["phylogenetic_tree"].apply(
            lambda x: x.get("kingdom", {}).get("scientific_name") if isinstance(x, dict) else None
        )
        df["biosample_link"] = df["organisms.biosample_id"].apply(
            lambda x: f"[{x}](https://portal.darwintreeoflife.org/organism/{x})")
        df["organism_link"] = df["organisms.organism"].apply(
            lambda x: f"[{x}](https://portal.darwintreeoflife.org/data/{urllib.parse.quote(x)})")

        # group by 'geotag' and aggregate data
        grouped_data = df.groupby("geotag").agg({
            "lat": "first",
            "lon": "first",
            "organisms.biosample_id": lambda x: ", ".join(set(filter(None, x))),
            "organisms.organism": lambda x: ", ".join(set(filter(None, x))),
            "organisms.common_name": lambda x: ", ".join(set(filter(None, x))),
            "organisms.sex": lambda x: ", ".join(set(filter(None, x))),
            "organisms.organism_part": lambda x: ", ".join(set(filter(None, x))),
            "Kingdom": lambda x: ", ".join(set(filter(None, x))),
            "current_status": lambda x: ", ".join(set(filter(None, x))),
            "experiment_type": lambda x: ", ".join(set(str(i).strip() for i in x if pd.notna(i))),
        }).reset_index()

        grouped_data["Record Count"] = df.groupby("geotag")["organisms.biosample_id"].nunique().values

        DATASETS[project_name] = {"df_data": df, "grouped_data": grouped_data}
    return DATASETS[project_name]


def layout(**kwargs):
    project_name = kwargs.get("projectName", "dtol")
    header_colour = "#f1f3f4"

    if project_name == "dtol":
        header_colour = "#8fbc45"
    elif project_name == "erga":
        header_colour = "#e0efea"
    elif project_name == "asg":
        header_colour = "#add8e6"
    elif project_name == "gbdp":
        header_colour = "#d0d0ce"

    return dbc.Container([
        dcc.Store(id="map-click-flag", data=False),
        dcc.Store(id="prev-click-data", data=None),
        dcc.Store(id="checklist-selection-order", data=[]),
        dcc.Store(id="project-store", data=project_name),

        # checklist section
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        dbc.InputGroup([
                            dbc.Input(
                                id="search-organism",
                                type="text",
                                placeholder="Search organism",
                                style={"borderRadius": "20px 0 0 20px", "backgroundColor": "#f1f3f4"}
                            ),
                            dbc.Button(
                                "Clear",
                                id="clear-organism",
                                color="dark",
                                size="sm",
                                style={
                                    "borderRadius": "0 20px 20px 0",
                                    "color": "#fff",
                                    "boxShadow": "0 0 4px rgba(0,0,0,0.3)"
                                }
                            )
                        ], className="mb-2"),
                    ], className="mb-2",
                        style={"backgroundColor": header_colour, "padding": "10px", "borderRadius": "5px"}),

                    dcc.Loading(
                        type="circle",
                        color="#0d6efd",
                        children=html.Div([
                            dcc.Checklist(
                                id="organism-checklist",
                                options=[],
                                className="custom-checklist",
                                inline=False,
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "7px"
                                },
                                labelStyle={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "5px"
                                }
                            )
                        ], style={
                            "overflowY": "auto",
                            "maxHeight": "260px"
                        })
                    )
                ])
            ], style={
                "maxHeight": "300px",
                "overflow": "hidden",
                "border": "1px solid #dee2e6",
                "borderRadius": "5px",
                "padding": "10px",
                "backgroundColor": "#f8f9fa",
                "flex": "1"
            }),

            dbc.Col([
                html.Div([
                    html.Div([
                        dbc.InputGroup([
                            dbc.Input(
                                id="search-common-name",
                                type="text",
                                placeholder="Search Common Name",
                                style={"borderRadius": "20px 0 0 20px", "backgroundColor": "#f1f3f4"}
                            ),
                            dbc.Button(
                                "Clear",
                                id="clear-common-name",
                                color="dark",
                                size="sm",
                                style={
                                    "borderRadius": "0 20px 20px 0",
                                    "color": "#fff",
                                    "boxShadow": "0 0 4px rgba(0,0,0,0.3)"
                                }
                            )
                        ], className="mb-2"),
                    ], className="mb-2",
                        style={"backgroundColor": header_colour, "padding": "10px", "borderRadius": "5px"}),

                    dcc.Loading(
                        type="circle",
                        color="#0d6efd",
                        children=html.Div([
                            dcc.Checklist(
                                id="common-name-checklist",
                                options=[],
                                className="custom-checklist",
                                inline=False,
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "7px"
                                },
                                labelStyle={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "5px"
                                }
                            )
                        ], style={
                            "overflowY": "auto",
                            "maxHeight": "260px"
                        })
                    )
                ])
            ], style={
                "maxHeight": "300px",
                "overflow": "hidden",
                "border": "1px solid #dee2e6",
                "borderRadius": "5px",
                "padding": "10px",
                "backgroundColor": "#f8f9fa",
                "flex": "1"
            }),
        ], className="mb-4", style={"display": "flex", "gap": "20px", "marginTop": "20px"}),

        # map + left-side filters (current status and experiment type)
        dbc.Row([
            dbc.Col([
                html.Div([

                    html.Div([
                        dbc.InputGroup([
                            dbc.Input(
                                id="search-current-status",
                                type="text",
                                placeholder="Search Current Status",
                                style={"borderRadius": "20px 0 0 20px", "backgroundColor": "#f1f3f4"}
                            ),
                            dbc.Button(
                                "Clear",
                                id="clear-current-status",
                                color="dark",
                                size="sm",
                                style={
                                    "borderRadius": "0 20px 20px 0",
                                    "color": "#fff",
                                    "boxShadow": "0 0 4px rgba(0,0,0,0.3)"
                                }
                            )
                        ], className="mb-2"),
                    ], className="mb-2",
                        style={"backgroundColor": header_colour, "padding": "10px", "borderRadius": "5px"}),

                    dcc.Loading(
                        type="circle",
                        color="#0d6efd",
                        children=html.Div([
                            dcc.Checklist(
                                id="current-status-checklist",
                                options=[],
                                className="custom-checklist",
                                inline=False,
                                style={"display": "flex", "flexDirection": "column", "gap": "7px"},
                                labelStyle={"display": "flex", "alignItems": "center", "gap": "5px"}
                            )
                        ], style={"overflowY": "auto", "maxHeight": "260px"})
                    )
                ], style={
                    "border": "1px solid #dee2e6",
                    "borderRadius": "5px",
                    "padding": "10px",
                    "backgroundColor": "#f8f9fa",
                    "marginBottom": "20px"
                }),

                html.Div([
                    html.Div([
                        dbc.InputGroup([
                            dbc.Input(
                                id="search-experiment-type",
                                type="text",
                                placeholder="Search Experiment Type",
                                style={"borderRadius": "20px 0 0 20px", "backgroundColor": "#f1f3f4"}
                            ),
                            dbc.Button(
                                "Clear",
                                id="clear-experiment-type",
                                color="dark",
                                size="sm",
                                style={
                                    "borderRadius": "0 20px 20px 0",
                                    "color": "#fff",
                                    "boxShadow": "0 0 4px rgba(0,0,0,0.3)"
                                }
                            )
                        ], className="mb-2"),
                    ], className="mb-2",
                        style={"backgroundColor": header_colour, "padding": "10px", "borderRadius": "5px"}),

                    dcc.Loading(
                        type="circle",
                        color="#0d6efd",
                        children=html.Div([
                            dcc.Checklist(
                                id="experiment-type-checklist",
                                options=[],
                                className="custom-checklist",
                                inline=False,
                                style={"display": "flex", "flexDirection": "column", "gap": "7px"},
                                labelStyle={"display": "flex", "alignItems": "center", "gap": "5px"}
                            )
                        ], style={"overflowY": "auto", "maxHeight": "260px"})
                    )
                ], style={
                    "border": "1px solid #dee2e6",
                    "borderRadius": "5px",
                    "padding": "10px",
                    "backgroundColor": "#f8f9fa"
                })
            ], width=3),

            dbc.Col(
                dbc.Spinner(dcc.Graph(id="sampling-map")),
                width=9
            )
        ], className="mb-4", style={"marginTop": "20px"}),

        dbc.Row(dbc.Col(
            dcc.Loading(
                id="loading-datatable",
                type="circle",
                color="#0d6efd",
                children=dash_table.DataTable(
                    id="datatable-paging",
                    columns=[
                        {"name": "BioSample ID", "id": "biosample_link", "presentation": "markdown"},
                        {"name": "Scientific Name", "id": "organism_link", "presentation": "markdown"},
                        {"name": "Common Name", "id": "common_name"},
                        {"name": "Current Status", "id": "current_status"},
                        {"name": "Symbionts Status", "id": "symbionts_status"}
                    ],
                    style_cell={"textAlign": "center"},
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


@callback(
    Output("checklist-selection-order", "data"),
    Input("organism-checklist", "value"),
    Input("common-name-checklist", "value"),
    Input("current-status-checklist", "value"),
    Input("experiment-type-checklist", "value"),
    State("checklist-selection-order", "data"),
    prevent_initial_call=True
)
def update_selection_order(org_val, common_val, status_val, exp_val, current_order):
    if current_order is None:
        current_order = []

    triggered_id = ctx.triggered_id
    triggered_value = {
        "organism-checklist": org_val,
        "common-name-checklist": common_val,
        "current-status-checklist": status_val,
        "experiment-type-checklist": exp_val,
    }.get(triggered_id)

    if triggered_value:
        # add if not already in the list (preserve original position)
        if triggered_id not in current_order:
            current_order.append(triggered_id)
    else:
        # remove if deselected
        if triggered_id in current_order:
            current_order.remove(triggered_id)

    return current_order


@callback(
    Output("organism-checklist", "value"),
    Output("common-name-checklist", "value"),
    Output("current-status-checklist", "value"),
    Output("experiment-type-checklist", "value"),
    Output("search-organism", "value"),
    Output("search-common-name", "value"),
    Output("search-current-status", "value"),
    Output("search-experiment-type", "value"),
    Input("clear-organism", "n_clicks"),
    Input("clear-common-name", "n_clicks"),
    Input("clear-current-status", "n_clicks"),
    Input("clear-experiment-type", "n_clicks"),
    prevent_initial_call=True
)
def clear_checklists(clear_org_clicks, clear_common_clicks, clear_status_clicks, clear_experiment_type_clicks):
    from dash import callback_context

    if not callback_context.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    organism_value = no_update
    common_name_value = no_update
    current_status_value = no_update
    experiment_type_value = no_update

    search_organism = no_update
    search_common_name = no_update
    search_current_status = no_update
    search_experiment_type = no_update

    if triggered_id == "clear-organism":
        organism_value = []
        search_organism = ""
    elif triggered_id == "clear-common-name":
        common_name_value = []
        search_common_name = ""
    elif triggered_id == "clear-current-status":
        current_status_value = []
        search_current_status = ""
    elif triggered_id == "clear-experiment-type":
        experiment_type_value = []
        search_experiment_type = ""

    return (
        organism_value,
        common_name_value,
        current_status_value,
        experiment_type_value,
        search_organism,
        search_common_name,
        search_current_status,
        search_experiment_type

    )


@callback(
    Output("current-status-checklist", "options"),
    Output("common-name-checklist", "options"),
    Output("organism-checklist", "options"),
    Output("experiment-type-checklist", "options"),

    Input("organism-checklist", "value"),
    Input("common-name-checklist", "value"),
    Input("current-status-checklist", "value"),
    Input("experiment-type-checklist", "value"),
    Input("search-organism", "value"),
    Input("search-common-name", "value"),
    Input("search-current-status", "value"),
    Input("search-experiment-type", "value"),
    Input("checklist-selection-order", "data"),
    Input("project-store", "data")
)
def update_checklists(selected_organisms, selected_common_names, selected_current_status, selected_experiment_type,
                      search_organism, search_common_name, search_current_status, search_experiment_type,
                      checklist_selection_order, project_name):
    user_selection = False

    data = load_data(project_name)["df_data"]

    all_organisms = sorted(data["organisms.organism"].unique())
    all_common_names = sorted(data["organisms.common_name"].unique())
    all_current_status = sorted(data["current_status"].unique())
    # all_experiment_type = sorted(filter(None, data["experiment_type"].unique()))
    all_experiment_type = sorted(
        str(v).strip() for v in data["experiment_type"].unique() if pd.notna(v)
    )

    # filter by search
    if search_organism:
        all_organisms = [org for org in all_organisms if search_organism.lower() in org.lower()]

    if search_common_name:
        all_common_names = [name for name in all_common_names if search_common_name.lower() in name.lower()]

    if search_current_status:
        all_current_status = [status for status in all_current_status if
                              search_current_status.lower() in status.lower()]

    if search_experiment_type:
        all_experiment_type = [exp_type for exp_type in all_experiment_type if search_experiment_type.lower()
                               in exp_type.lower()]

    grouped = {
        col: data.groupby(col)[["organisms.organism", "organisms.common_name", "current_status", "experiment_type"]].agg(
            lambda x: set(x))
        for col in ["organisms.organism", "organisms.common_name", "current_status", "experiment_type"]
    }

    org_to_common = grouped["organisms.organism"]["organisms.common_name"].to_dict()
    org_to_current_status = grouped["organisms.organism"]["current_status"].to_dict()
    org_to_experiment_type = grouped["organisms.organism"]["experiment_type"].to_dict()

    common_to_org = grouped["organisms.common_name"]["organisms.organism"].to_dict()
    common_to_current_status = grouped["organisms.common_name"]["current_status"].to_dict()
    common_to_experiment_type = grouped["organisms.common_name"]["experiment_type"].to_dict()

    status_to_org = grouped["current_status"]["organisms.organism"].to_dict()
    status_to_common_name = grouped["current_status"]["organisms.common_name"].to_dict()
    status_to_experiment_type = grouped["current_status"]["experiment_type"].to_dict()

    experiment_type_to_org = grouped["experiment_type"]["organisms.organism"].to_dict()
    experiment_type_to_common_name = grouped["experiment_type"]["organisms.common_name"].to_dict()
    experiment_type_to_status = grouped["experiment_type"]["current_status"].to_dict()

    organism_options = [{"label": org, "value": org, "disabled": False}
                        for org in all_organisms]

    common_name_options = [{"label": name, "value": name, "disabled": False}
                           for name in all_common_names]

    current_status_options = [{"label": status, "value": status, "disabled": False}
                              for status in all_current_status]

    experiment_type_options = [{"label": exp_type, "value": exp_type, "disabled": False}
                               for exp_type in all_experiment_type]

    print("checklist_selection_order ---> ", checklist_selection_order)

    checklist_mappings = {
        "organism-checklist": "selected_organisms",
        "common-name-checklist": "selected_common_names",
        "current-status-checklist": "selected_current_status",
        "experiment-type-checklist": "selected_experiment_type"
    }

    selections_list = [checklist_mappings[chk_id] for chk_id in checklist_selection_order]

    if selections_list and len(selections_list) > 0:
        user_selection = True

    allowed_orgs = set(all_organisms)
    allowed_common = set(all_common_names)
    allowed_current_status = set(all_current_status)
    allowed_experiment_type = set(all_experiment_type)

    for selection in selections_list:
        if selection == 'selected_organisms':
            for i, org in enumerate(selected_organisms):
                common = org_to_common.get(org, set())
                current_status = org_to_current_status.get(org, set())
                exp_type = org_to_experiment_type.get(org, set())
                if i == 0:
                    allowed_common = allowed_common & common
                    allowed_current_status = allowed_current_status & current_status
                    allowed_experiment_type = allowed_experiment_type & exp_type
                else:
                    allowed_common.update(common)
                    allowed_current_status.update(current_status)
                    allowed_experiment_type.update(exp_type)

        elif selection == 'selected_common_names':
            for i, name in enumerate(selected_common_names):
                orgs = common_to_org.get(name, set())
                current_status = common_to_current_status.get(name, set())
                exp_type = common_to_experiment_type.get(name, set())
                if i == 0:
                    allowed_orgs = allowed_orgs & orgs
                    allowed_current_status = allowed_current_status & current_status
                    allowed_experiment_type = allowed_experiment_type & exp_type
                else:
                    allowed_orgs.update(orgs)
                    allowed_current_status.update(current_status)
                    allowed_experiment_type.update(exp_type)

        elif selection == 'selected_current_status':
            for i, status in enumerate(selected_current_status):
                orgs = status_to_org.get(status, set())
                common = status_to_common_name.get(status, set())
                exp_type = status_to_experiment_type.get(status, set())

                if i == 0:
                    allowed_orgs = allowed_orgs & orgs
                    allowed_common = allowed_common & common
                    allowed_experiment_type = allowed_experiment_type & exp_type
                else:
                    allowed_orgs.update(orgs)
                    allowed_common.update(common)
                    allowed_experiment_type.update(exp_type)

        elif selection == 'selected_experiment_type':
            for i, exp_type in enumerate(selected_experiment_type):
                orgs = experiment_type_to_org.get(exp_type, set())
                common = experiment_type_to_common_name.get(exp_type, set())
                status = experiment_type_to_status.get(exp_type, set())

                if i == 0:
                    allowed_orgs = allowed_orgs & orgs
                    allowed_common = allowed_common & common
                    allowed_current_status = allowed_current_status & status
                else:
                    allowed_orgs.update(orgs)
                    allowed_common.update(common)
                    allowed_current_status.update(status)

    if user_selection:
        organism_options = [
            {"label": org, "value": org, "disabled": org not in allowed_orgs} if allowed_orgs
            else {"label": org, "value": org, "disabled": True}
            for org in all_organisms
        ]

        common_name_options = [
            {"label": name, "value": name, "disabled": name not in allowed_common} if allowed_common
            else {"label": name, "value": name, "disabled": True}
            for name in all_common_names
        ]

        current_status_options = [
            {"label": status, "value": status,
             "disabled": status not in allowed_current_status} if allowed_current_status
            else {"label": status, "value": status, "disabled": True}
            for status in all_current_status
        ]

        experiment_type_options = [
            {"label": exp_type, "value": exp_type,
             "disabled": exp_type not in allowed_experiment_type} if allowed_experiment_type
            else {"label": exp_type, "value": exp_type, "disabled": True}
            for exp_type in all_experiment_type
        ]

    organism_options.sort(key=lambda x: (x["value"] not in (selected_organisms or []), x["disabled"]))
    common_name_options.sort(key=lambda x: (x["value"] not in (selected_common_names or []), x["disabled"]))
    current_status_options.sort(key=lambda x: (x["value"] not in (selected_current_status or []), x["disabled"]))
    experiment_type_options.sort(key=lambda x: (x["value"] not in (selected_experiment_type or []), x["disabled"]))

    return current_status_options, common_name_options, organism_options, experiment_type_options


@callback(
    Output("sampling-map", "figure"),
    Input("organism-checklist", "value"),
    Input("common-name-checklist", "value"),
    Input("current-status-checklist", "value"),
    Input("experiment-type-checklist", "value"),
    Input("project-store", "data")
)
def build_map(selected_organisms, selected_common_names, selected_current_status, selected_experiment_types,
              project_name):
    filtered_map_data = load_data(project_name)["grouped_data"]

    if selected_organisms:
        filtered_map_data = filtered_map_data[filtered_map_data["organisms.organism"].apply(
            lambda organism_str: any(org in organism_str.split(", ") for org in selected_organisms)
        )]

    if selected_common_names:
        filtered_map_data = filtered_map_data[filtered_map_data["organisms.common_name"].apply(
            lambda common_name_str: any(name in common_name_str.split(", ") for name in selected_common_names)
        )]

    if selected_current_status:
        filtered_map_data = filtered_map_data[filtered_map_data["current_status"].apply(
            lambda current_status_str: any(
                status in current_status_str.split(", ") for status in selected_current_status)
        )]

    if selected_experiment_types:
        filtered_map_data = filtered_map_data[filtered_map_data["experiment_type"].apply(
            lambda experiment_type_str: any(
                exp_type in experiment_type_str.split(", ") for exp_type in selected_experiment_types)
        )]

    # marker sizes
    min_marker_size = 3
    max_marker_size = 50

    min_count = filtered_map_data["Record Count"].min() if not filtered_map_data.empty else 1
    max_count = filtered_map_data["Record Count"].max() if not filtered_map_data.empty else 1

    if max_count > min_count:
        filtered_map_data["scaled_size"] = filtered_map_data["Record Count"].apply(
            lambda x: min_marker_size + ((x - min_count) / (max_count - min_count)) * (
                max_marker_size - min_marker_size)
        )
    else:
        filtered_map_data["scaled_size"] = min_marker_size

    # create map
    map_fig = px.scatter_map(
        filtered_map_data,
        lat="lat",
        lon="lon",
        color="Kingdom",
        size="scaled_size",
        zoom=3,
        hover_name="geotag",
        hover_data={"lat": False, "lon": False, "scaled_size": False, "Kingdom": True, "Record Count": True},
        height=800
    )

    map_fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            title="Kingdom",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.07,
            yanchor="bottom"
        )
    )

    return map_fig


@callback(
    Output("map-click-flag", "data"),
    Output("prev-click-data", "data"),
    Input("sampling-map", "clickData"),
    Input("organism-checklist", "value"),
    Input("common-name-checklist", "value"),
    Input("current-status-checklist", "value"),
    Input("experiment-type-checklist", "value"),
    State("prev-click-data", "data"),
    prevent_initial_call=True
)
def update_click_flag(click_data, selected_organisms, selected_common_names, selected_current_status,
                      selected_experiment_types, prev_click_data):
    if selected_organisms or selected_common_names or selected_current_status or selected_experiment_types:
        print("Checklist selected, resetting flag to False.")
        return False, prev_click_data

    if click_data and click_data != prev_click_data:
        print("New map click detected, setting flag to True.")
        return True, click_data

    return no_update, prev_click_data


@callback(
    Output("datatable-paging", "data"),
    Output("datatable-paging", "page_count"),
    Output("datatable-paging", "page_current"),
    Input("map-click-flag", "data"),  # determines if triggered by map click or checklist
    Input("sampling-map", "selectedData"),
    Input("sampling-map", "clickData"),
    Input("organism-checklist", "value"),
    Input("common-name-checklist", "value"),
    Input("current-status-checklist", "value"),
    Input("experiment-type-checklist", "value"),
    Input('datatable-paging', "page_current"),
    Input('datatable-paging', "page_size"),
    Input("project-store", "data")
)
def build_table(click_flag_value, selected_data, click_data, selected_organisms, selected_common_names,
                selected_current_status, selected_experiment_types, page_current, page_size, project_name):
    data = load_data(project_name)["df_data"]
    print(f"click_flag_value: {click_flag_value}")
    filtered_df = data

    # map click
    if click_flag_value:
        selected_geotags = set()
        if selected_data:
            selected_geotags.update(point["hovertext"] for point in selected_data["points"])
        if click_data:
            selected_geotags.update(point["hovertext"] for point in click_data["points"])

        # filter data based on geotags
        if selected_geotags:
            filtered_df = data[data["geotag"].isin(selected_geotags)]

    # checklist selection
    if not click_flag_value:
        if selected_organisms:
            filtered_df = filtered_df[filtered_df["organisms.organism"].isin(selected_organisms)]

        if selected_common_names:
            filtered_df = filtered_df[filtered_df["organisms.common_name"].isin(selected_common_names)]

        if selected_current_status:
            filtered_df = filtered_df[filtered_df["current_status"].isin(selected_current_status)]

        if selected_experiment_types:
            filtered_df = filtered_df[filtered_df["experiment_type"].isin(selected_experiment_types)]

    # we have to do the gouping again because of the unnesting of raw_data which is a repeated record - array of structs
    # see BigQuery metadata schema
    grouped_df = filtered_df.groupby(['geotag', 'organisms.biosample_id']).agg({
        "experiment_type": lambda x: ", ".join(set(str(i).strip() for i in x if pd.notna(i))),
        'common_name': 'first',
        'current_status': 'first',
        'symbionts_status': 'first',
        'organisms.organism': 'first',
    }).reset_index()

    grouped_df["biosample_link"] = grouped_df["organisms.biosample_id"].apply(
        lambda x: f"[{x}](https://portal.darwintreeoflife.org/organism/{x})")
    grouped_df["organism_link"] = grouped_df["organisms.organism"].apply(
        lambda x: f"[{x}](https://portal.darwintreeoflife.org/data/{urllib.parse.quote(x)})")

    # pagination
    total_pages = max(math.ceil(len(grouped_df) / page_size), 1)
    page_current = min(page_current, total_pages - 1)

    return (
        grouped_df.iloc[page_current * page_size:(page_current + 1) * page_size].to_dict("records"),
        total_pages,
        page_current
    )
