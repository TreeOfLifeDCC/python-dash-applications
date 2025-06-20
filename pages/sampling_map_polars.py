import math
import urllib
import numpy as np
import plotly.express as px
import google.auth
from gcsfs import GCSFileSystem
import dash
import fsspec
from dash import dcc, callback, Output, Input, dash_table, State, no_update, html, ctx, callback_context
import dash_bootstrap_components as dbc
import polars as pl # Add this import


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


def extract_unique_protocols(raw_data_entry):
    if isinstance(raw_data_entry, (list, np.ndarray)):
        protocols = {
            item.get('library_construction_protocol')
            for item in raw_data_entry
            if isinstance(item, dict) and item.get('library_construction_protocol')
        }
        return ','.join(protocols) if protocols else None
    return None


def load_data(project_name):
    if project_name not in DATASETS:
        gcs_pattern = PROJECT_PARQUET_MAP.get(project_name, PROJECT_PARQUET_MAP["dtol"])

        fs = fsspec.filesystem("gcs")

        matching_files = fs.glob(gcs_pattern)

        if not matching_files:
            raise FileNotFoundError(f"No Parquet files found for pattern: {gcs_pattern}")

        lazy_chunks = []
        for file in matching_files:
            file_path = f"gs://{file}"
            # scan the file lazily and select only the columns needed for this page
            lf = pl.scan_parquet(file_path).select([
                pl.col("organisms").list.eval(
                    pl.struct([
                        pl.element().struct.field("biosample_id").alias("biosample_id"),
                        pl.element().struct.field("organism").alias("organism"),
                        pl.element().struct.field("latitude").alias("latitude"),
                        pl.element().struct.field("longitude").alias("longitude")
                    ])
                ).alias("organisms"),
                pl.col("raw_data").list.eval(
                    pl.element().struct.field("library_construction_protocol")).alias("raw_data_protocols"),
                # keep other fields as they are
                "raw_data",
                "current_status",
                "common_name",
                "tax_id",
                "symbionts_status",
                "phylogenetic_tree"
            ])
            lazy_chunks.append(lf)

        final_lf = pl.concat(lazy_chunks)

        final_lf = (
            final_lf
            .explode("organisms")
            .unnest("organisms")
            .with_columns(
                pl.when(pl.col("raw_data_protocols").list.len() > 0)
                .then(
                    pl.col("raw_data_protocols")
                    .list.unique()
                    .list.filter(
                        (pl.element().is_not_null()) & (pl.element() != "")
                    )
                )
                .otherwise([])
                .alias("experiment_type")
            )

            .explode("experiment_type")
            .with_columns(
                pl.col("latitude").cast(pl.Float64, strict=False).alias("lat"),
                pl.col("longitude").cast(pl.Float64, strict=False).alias("lon"),
                pl.col("phylogenetic_tree").struct.field("kingdom").struct.field("scientific_name").alias("Kingdom"),
            )
            # remove invalid geo-coordinates
            .filter(
                pl.col("lat").is_between(-90, 90) & pl.col("lon").is_between(-180, 180)
            )
            # create geotag and organism link
            .with_columns(
                (pl.col("lat").cast(pl.Utf8) + "," + pl.col("lon").cast(pl.Utf8)).alias("geotag"),
                pl.when(pl.col("organism").is_not_null())
                .then(pl.format(
                    "[{}](https://portal.darwintreeoflife.org/data/{})",
                    pl.col("organism"),
                    pl.col("organism").map_elements(lambda o: urllib.parse.quote(str(o)), return_dtype=pl.Utf8)
                ))
                .otherwise(None)
                .alias("organism_link")
            )
        )

        df_data = (
            final_lf
            .rename({
                "biosample_id": "organisms.biosample_id",
                "organism": "organisms.organism",
                "latitude": "organisms.latitude",
                "longitude": "organisms.longitude",
            })
            .collect()
        )

        grouped_data = (
            final_lf
            .group_by("geotag")
            .agg(
                pl.col("lat").first(),
                pl.col("lon").first(),
                pl.col("biosample_id").drop_nulls().unique().str.concat(", "),
                pl.col("organism").drop_nulls().unique().str.concat(", "),
                pl.col("Kingdom").drop_nulls().unique(),
                pl.col("common_name").drop_nulls().unique().str.concat(", "),
                pl.col("current_status").drop_nulls().unique().str.concat(", "),
                pl.col("experiment_type").drop_nulls().unique().str.concat(", "),
                pl.col("organism").n_unique().alias("Record Count"),
            )
            .explode("Kingdom")
            .rename({
                "biosample_id": "organisms.biosample_id",
                "organism": "organisms.organism",
            })
            .collect()
        )

        DATASETS[project_name] = {"df_data": df_data, "grouped_data": grouped_data}

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

    return dbc.Container([
        dcc.Store(id="map-click-flag", data=False),
        dcc.Store(id="map-selection-flag", data=False),
        dcc.Store(id="prev-click-data", data=None),
        dcc.Store(id="checklist-selection-order", data=[]),
        dcc.Store(id="project-store", data=project_name),
        dcc.Store(id="selected-geotags", data=[]),
        dcc.Store(id="selected-organisms", data=[]),

        # checklist section
        html.Div([
            # Horizontal checklists
            html.Div([
                # Scientific Name Checklist
                html.Div([
                    html.Div([
                        dbc.InputGroup([
                            dbc.Input(
                                id="search-organism",
                                type="text",
                                placeholder="Scientific Name",
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
                ], style={
                    "border": "1px solid #dee2e6",
                    "borderRadius": "5px",
                    "padding": "10px",
                    "backgroundColor": "#f8f9fa",
                    "flex": "1"
                }),

                # spacer
                html.Div(style={"width": "20px"}),

                # Common Name Checklist
                html.Div([
                    html.Div([
                        dbc.InputGroup([
                            dbc.Input(
                                id="search-common-name",
                                type="text",
                                placeholder="Common Name",
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
                ], style={
                    "border": "1px solid #dee2e6",
                    "borderRadius": "5px",
                    "padding": "10px",
                    "backgroundColor": "#f8f9fa",
                    "flex": "1"
                })
            ], style={
                "display": "flex",
                "flexDirection": "row",
                "gap": "20px"
            })
        ], style={
            "marginTop": "20px",
            "marginBottom": "20px"
        }),

        html.Div(
            id="map-selection-alert",
            children=[],
            style={"marginBottom": "20px"}
        ),

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
                        ], style={"overflowY": "auto", "maxHeight": "420px"})
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

        # Reset Button
        html.Div(
            dbc.Button(
                "Reset All",
                id="reset-all-button",
                color="danger",
                className="mb-3"
            ),
            style={"display": "flex", "justifyContent": "flex-end"}
        ),

        # data table
        dbc.Row(dbc.Col(
            dcc.Loading(
                id="loading-datatable",
                type="circle",
                color="#0d6efd",
                children=dash_table.DataTable(
                    id="datatable-paging",
                    columns=[
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
    Output("map-selection-alert", "children"),
    Input("selected-organisms", "data"),
)
def update_species_selection_alert(selected_organisms):
    if selected_organisms:
        return dbc.Alert(
            f"{len(selected_organisms)} unique species selected from the map — checklist filters apply within this "
            f"selection.",
            color="info",
            dismissable=True,
            is_open=True,
            style={"marginTop": "10px"}
        )
    return []


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


def create_options(options_list, display_all=True, allowed_options=set()):
    if display_all:
        return_list = [{"label": option, "value": option, "disabled": False}
                       for option in options_list]
    else:
        return_list = [{"label": option, "value": option}
                       for option in options_list if option in allowed_options
                       ]
    return return_list


@callback(
    Output("organism-checklist", "value", allow_duplicate=True),
    Output("common-name-checklist", "value", allow_duplicate=True),
    Output("current-status-checklist", "value", allow_duplicate=True),
    Output("experiment-type-checklist", "value", allow_duplicate=True),
    Output("search-organism", "value", allow_duplicate=True),
    Output("search-common-name", "value", allow_duplicate=True),
    Output("search-current-status", "value", allow_duplicate=True),
    Output("search-experiment-type", "value", allow_duplicate=True),
    Output("map-click-flag", "data", allow_duplicate=True),
    Output("map-selection-flag", "data", allow_duplicate=True),
    Output("prev-click-data", "data", allow_duplicate=True),
    Output("sampling-map", "clickData", allow_duplicate=True),
    Output("sampling-map", "selectedData", allow_duplicate=True),
    Output("checklist-selection-order", "data", allow_duplicate=True),
    Output("selected-geotags", "data", allow_duplicate=True),
    Output("selected-organisms", "data", allow_duplicate=True),
    Output("datatable-paging", "page_current", allow_duplicate=True),

    Input("reset-all-button", "n_clicks"),
    prevent_initial_call=True
)
def reset_all(n_clicks):
    return (
        [], [], [], [],  # checklist values
        "", "", "", "",  # search inputs
        False, False, None, None, None,  # map flags, clickData
        [], [], [],  # selected chklist, geotags & organisms stores
        0  # reset page
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
    Input("project-store", "data"),

    Input("map-click-flag", "data"),
    Input("map-selection-flag", "data"),
    Input("sampling-map", "selectedData"),
    Input("sampling-map", "clickData"),
    Input("selected-geotags", "data"),
)
def update_checklists(selected_organisms, selected_common_names, selected_current_status, selected_experiment_type,
                      search_organism, search_common_name, search_current_status, search_experiment_type,
                      checklist_selection_order, project_name,
                      click_flag_value, selection_flag_value, selected_data, click_data, selected_geotags):

    df_lazy = DATASETS[project_name]["df_data"].lazy()

    if selected_geotags:
        df_lazy = df_lazy.filter(pl.col("geotag").is_in(selected_geotags))

    df_processed = df_lazy.with_columns([
        # split and prepare organism column
        pl.when(pl.col("organisms.organism").is_not_null())
        .then(pl.concat_list([pl.col("organisms.organism").cast(pl.Utf8)]))
        .otherwise(pl.lit([], dtype=pl.List(pl.Utf8)))
        .alias("organisms_split"),

        # split and prepare experiment_type column
        pl.when(pl.col("experiment_type").is_not_null())
        .then(pl.col("experiment_type").cast(pl.Utf8).str.split(", "))
        .otherwise(pl.lit(None, dtype=pl.List(pl.Utf8)))
        .alias("experiment_type_split"),

        pl.col("common_name").cast(pl.Utf8),
        pl.col("current_status").cast(pl.Utf8)
    ])

    df_org_exploded = df_processed.explode("organisms_split").filter(pl.col("organisms_split").is_not_null())
    df_exp_exploded = df_processed.explode("experiment_type_split").filter(
        pl.col("experiment_type_split").is_not_null())

    all_organisms = sorted(
        df_processed.select(pl.col("organisms_split").flatten().drop_nulls().unique())
        .collect()
        .to_series()
        .to_list()
    )

    all_common_names = sorted(
        df_processed.select(pl.col("common_name").drop_nulls().unique())
        .collect()
        .to_series()
        .to_list()
    )

    all_current_status = sorted(
        df_processed.select(pl.col("current_status").drop_nulls().unique())
        .collect()
        .to_series()
        .to_list()
    )

    all_experiment_type = sorted(
        df_processed.select(pl.col("experiment_type_split").flatten().drop_nulls().unique())
        .collect()
        .to_series()
        .to_list()
    )

    all_values = {
        "all_organisms": all_organisms,
        "all_common_names": all_common_names,
        "all_current_status": all_current_status,
        "all_experiment_type": all_experiment_type,
    }

    # Apply search filters
    search_map = {
        "all_organisms": search_organism,
        "all_common_names": search_common_name,
        "all_current_status": search_current_status,
        "all_experiment_type": search_experiment_type,
    }

    for key, search in search_map.items():
        if search:
            all_values[key] = [v for v in all_values[key] if search.lower() in v.lower()]


    lookup_tables = {}

    #only build lookup tables if we have user selections
    user_selection = False
    checklist_mappings = {
        "organism-checklist": "selected_organisms",
        "common-name-checklist": "selected_common_names",
        "current-status-checklist": "selected_current_status",
        "experiment-type-checklist": "selected_experiment_type"
    }

    selections_list = [checklist_mappings[chk_id] for chk_id in checklist_selection_order if
                       chk_id in checklist_mappings]

    if selections_list and len(selections_list) > 0:
        user_selection = True

        lookup_queries = []

        # Organism to other fields
        if any("organisms" in sel for sel in selections_list):
            lookup_queries.extend([
                df_org_exploded.group_by("organisms_split").agg([
                    pl.col("common_name").drop_nulls().unique().alias("common_names")
                ]).select([
                    pl.col("organisms_split").alias("source"),
                    pl.lit("organism_to_common").alias("lookup_type"),
                    pl.col("common_names").alias("targets")
                ]),
                df_org_exploded.group_by("organisms_split").agg([
                    pl.col("current_status").drop_nulls().unique().alias("statuses")
                ]).select([
                    pl.col("organisms_split").alias("source"),
                    pl.lit("organism_to_status").alias("lookup_type"),
                    pl.col("statuses").alias("targets")
                ]),
                df_org_exploded.explode("experiment_type_split").filter(
                    pl.col("experiment_type_split").is_not_null()
                ).group_by("organisms_split").agg([
                    pl.col("experiment_type_split").unique().alias("exp_types")
                ]).select([
                    pl.col("organisms_split").alias("source"),
                    pl.lit("organism_to_experiment").alias("lookup_type"),
                    pl.col("exp_types").alias("targets")
                ])
            ])

        # Common Name to other fields
        if any("common" in sel for sel in selections_list):
            lookup_queries.extend([
                df_processed.explode("organisms_split").filter(
                    pl.col("organisms_split").is_not_null()
                ).group_by("common_name").agg([
                    pl.col("organisms_split").unique().alias("organisms")
                ]).select([
                    pl.col("common_name").alias("source"),
                    pl.lit("common_to_organism").alias("lookup_type"),
                    pl.col("organisms").alias("targets")
                ]),
                df_processed.group_by("common_name").agg([
                    pl.col("current_status").drop_nulls().unique().alias("statuses")
                ]).select([
                    pl.col("common_name").alias("source"),
                    pl.lit("common_to_status").alias("lookup_type"),
                    pl.col("statuses").alias("targets")
                ]),
                df_processed.explode("experiment_type_split").filter(
                    pl.col("experiment_type_split").is_not_null()
                ).group_by("common_name").agg([
                    pl.col("experiment_type_split").unique().alias("exp_types")
                ]).select([
                    pl.col("common_name").alias("source"),
                    pl.lit("common_to_experiment").alias("lookup_type"),
                    pl.col("exp_types").alias("targets")
                ])
            ])

        # Current Status to other fields
        if any("status" in sel for sel in selections_list):
            lookup_queries.extend([
                df_processed.explode("organisms_split").filter(
                    pl.col("organisms_split").is_not_null()
                ).group_by("current_status").agg([
                    pl.col("organisms_split").unique().alias("organisms")
                ]).select([
                    pl.col("current_status").alias("source"),
                    pl.lit("status_to_organism").alias("lookup_type"),
                    pl.col("organisms").alias("targets")
                ]),
                df_processed.group_by("current_status").agg([
                    pl.col("common_name").drop_nulls().unique().alias("common_names")
                ]).select([
                    pl.col("current_status").alias("source"),
                    pl.lit("status_to_common").alias("lookup_type"),
                    pl.col("common_names").alias("targets")
                ]),
                df_processed.explode("experiment_type_split").filter(
                    pl.col("experiment_type_split").is_not_null()
                ).group_by("current_status").agg([
                    pl.col("experiment_type_split").unique().alias("exp_types")
                ]).select([
                    pl.col("current_status").alias("source"),
                    pl.lit("status_to_experiment").alias("lookup_type"),
                    pl.col("exp_types").alias("targets")
                ])
            ])

        # Experiment Type to other fields
        if any("experiment" in sel for sel in selections_list):
            lookup_queries.extend([
                df_exp_exploded.explode("organisms_split").filter(
                    pl.col("organisms_split").is_not_null()
                ).group_by("experiment_type_split").agg([
                    pl.col("organisms_split").unique().alias("organisms")
                ]).select([
                    pl.col("experiment_type_split").alias("source"),
                    pl.lit("experiment_to_organism").alias("lookup_type"),
                    pl.col("organisms").alias("targets")
                ]),
                df_exp_exploded.group_by("experiment_type_split").agg([
                    pl.col("common_name").drop_nulls().unique().alias("common_names")
                ]).select([
                    pl.col("experiment_type_split").alias("source"),
                    pl.lit("experiment_to_common").alias("lookup_type"),
                    pl.col("common_names").alias("targets")
                ]),
                df_exp_exploded.group_by("experiment_type_split").agg([
                    pl.col("current_status").drop_nulls().unique().alias("statuses")
                ]).select([
                    pl.col("experiment_type_split").alias("source"),
                    pl.lit("experiment_to_status").alias("lookup_type"),
                    pl.col("statuses").alias("targets")
                ])
            ])

        # collect all lookup queries
        if lookup_queries:
            combined_lookups = pl.concat(lookup_queries).collect()

            # build lookup dictionary from results
            for row in combined_lookups.iter_rows(named=True):
                lookup_type = row["lookup_type"]
                source = row["source"]
                targets = set(row["targets"]) if row["targets"] is not None else set()

                if lookup_type not in lookup_tables:
                    lookup_tables[lookup_type] = {}
                lookup_tables[lookup_type][source] = targets

    print("checklist_selection_order ---> ", checklist_selection_order)

    # allowed sets
    allowed_orgs = set(all_values['all_organisms'])
    allowed_common = set(all_values['all_common_names'])
    allowed_current_status = set(all_values['all_current_status'])
    allowed_experiment_type = set(all_values['all_experiment_type'])

    def calculate_allowed_options_optimized(selected_values, lookup_type_prefix, allowed_sets):
        if not selected_values:
            return

        target_types = ["organism", "common", "status", "experiment"]
        target_keys = ["organisms.organism", "common_name", "current_status", "experiment_type"]

        for target_type, target_key in zip(target_types, target_keys):
            if target_type == lookup_type_prefix.split("_")[0]:
                continue

            lookup_key = f"{lookup_type_prefix}_to_{target_type}"
            if lookup_key not in lookup_tables:
                continue

            if selected_values:
                result_targets = set()
                for value in selected_values:
                    value_targets = lookup_tables[lookup_key].get(value, set())
                    result_targets.update(value_targets)  # UNION = OR logic

                # Apply AND logic between different checklists
                allowed_sets[target_key].intersection_update(result_targets)

    selection_configs = {
        "selected_organisms": ("organism", selected_organisms),
        "selected_common_names": ("common", selected_common_names),
        "selected_current_status": ("status", selected_current_status),
        "selected_experiment_type": ("experiment", selected_experiment_type),
    }

    allowed_sets = {
        "organisms.organism": allowed_orgs.copy(),
        "common_name": allowed_common.copy(),
        "current_status": allowed_current_status.copy(),
        "experiment_type": allowed_experiment_type.copy(),
    }

    # apply filters based on selection order
    for selection_name in selections_list:
        if selection_name in selection_configs:
            lookup_prefix, values = selection_configs[selection_name]
            calculate_allowed_options_optimized(values, lookup_prefix, allowed_sets)

    # update allowed sets
    allowed_orgs = allowed_sets["organisms.organism"]
    allowed_common = allowed_sets["common_name"]
    allowed_current_status = allowed_sets["current_status"]
    allowed_experiment_type = allowed_sets["experiment_type"]

    # create options
    if user_selection:
        organism_options = create_options(all_values['all_organisms'], False, allowed_orgs)
        common_name_options = create_options(all_values['all_common_names'], False, allowed_common)
        current_status_options = create_options(all_values['all_current_status'], False, allowed_current_status)
        experiment_type_options = create_options(all_values['all_experiment_type'], False, allowed_experiment_type)
    else:
        organism_options = create_options(all_values['all_organisms'])
        common_name_options = create_options(all_values['all_common_names'])
        current_status_options = create_options(all_values['all_current_status'])
        experiment_type_options = create_options(all_values['all_experiment_type'])

    # sort options
    organism_options.sort(key=lambda x: (x["value"] not in (selected_organisms or []), x["label"].lower()))
    common_name_options.sort(key=lambda x: (x["value"] not in (selected_common_names or []), x["label"].lower()))
    current_status_options.sort(key=lambda x: (x["value"] not in (selected_current_status or []), x["label"].lower()))
    experiment_type_options.sort(key=lambda x: (x["value"] not in (selected_experiment_type or []), x["label"].lower()))

    return current_status_options, common_name_options, organism_options, experiment_type_options


@callback(
    Output("sampling-map", "figure"),
    Input("organism-checklist", "value"),
    Input("common-name-checklist", "value"),
    Input("current-status-checklist", "value"),
    Input("experiment-type-checklist", "value"),
    Input("project-store", "data"),
    Input("selected-geotags", "data")
)
def build_map(
    selected_organisms,
    selected_common_names,
    selected_current_status,
    selected_experiment_types,
    project_name,
    selected_geotags
):
    df_lazy = DATASETS[project_name]["grouped_data"].lazy()

    if selected_geotags:
        df_lazy = df_lazy.filter(pl.col("geotag").is_in(selected_geotags))

    filter_configs = [
        ("organisms.organism", selected_organisms),
        ("common_name", selected_common_names),
        ("current_status", selected_current_status),
        ("experiment_type", selected_experiment_types),
    ]

    for field, values in filter_configs:
        if values:
            df_lazy = df_lazy.filter(
                pl.col(field)
                .str.split(", ")
                .list.eval(pl.element().is_in(values))
                .list.any()
            )

    # get min/max counts for scaling (minimal collection - just aggregates)
    count_stats = df_lazy.select([
        pl.col("Record Count").min().alias("min_count"),
        pl.col("Record Count").max().alias("max_count"),
        pl.len().alias("total_rows")
    ]).collect()

    if count_stats.height > 0:
        min_count = count_stats["min_count"][0] or 1
        max_count = count_stats["max_count"][0] or 1
        total_rows = count_stats["total_rows"][0]
    else:
        min_count, max_count, total_rows = 1, 1, 0

    min_size, max_size = 3, 50

    if total_rows > 0:
        if max_count > min_count:
            processed_df = df_lazy.with_columns([
                (((pl.col("Record Count") - min_count) / (max_count - min_count))
                 * (max_size - min_size)
                 + min_size
                 ).alias("scaled_size")
            ])
        else:
            # use minimum size
            processed_df = df_lazy.with_columns([
                pl.lit(min_size).alias("scaled_size")
            ])

        # collect final data needed for visualization
        df_final = processed_df.collect()

        # convert to pandas
        pdf = df_final.to_pandas()

        # build map figure
        fig = px.scatter_map(
            pdf,
            lat="lat",
            lon="lon",
            color="Kingdom",
            size="scaled_size",
            zoom=3,
            hover_name="geotag",
            hover_data={
                "lat": False,
                "lon": False,
                "scaled_size": False,
                "Kingdom": True,
                "Record Count": True
            },
            height=800
        )
    else:
        fig = px.scatter_map(
            lat=[],
            lon=[],
            zoom=3,
            height=800
        )

    fig.update_layout(
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

    return fig


@callback(
    Output("map-click-flag", "data"),
    Output("prev-click-data", "data"),
    Output("map-selection-flag", "data"),
    Output("selected-geotags", "data"),
    Output("selected-organisms", "data"),

    Input("sampling-map", "clickData"),
    Input("sampling-map", "selectedData"),
    Input("organism-checklist", "value"),
    Input("common-name-checklist", "value"),
    Input("current-status-checklist", "value"),
    Input("experiment-type-checklist", "value"),
    State("prev-click-data", "data"),
    Input("clear-organism", "n_clicks"),
    Input("clear-common-name", "n_clicks"),
    Input("clear-current-status", "n_clicks"),
    Input("clear-experiment-type", "n_clicks"),
    State("project-store", "data"),
    prevent_initial_call=True
)
def update_click_flag(click_data, selected_data, selected_organisms, selected_common_names, selected_current_status,
                      selected_experiment_types, prev_click_data,
                      clear_org_clicks, clear_common_clicks, clear_status_clicks, clear_experiment_type_clicks,
                      project_name):

    df_data: pl.DataFrame = DATASETS[project_name]["df_data"]

    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    checklist_ids = {
        "organism-checklist",
        "common-name-checklist",
        "current-status-checklist",
        "experiment-type-checklist",
    }

    # check if the callback was triggered by a checklist interaction
    checklist_interaction = triggered_id in checklist_ids

    if triggered_id.startswith("clear-"):
        return False, None, False, [], []  # clear selected geotags and organisms

    if checklist_interaction:
        return False, None, False, no_update, no_update

    # map click event
    if click_data and click_data != prev_click_data:
        selected_geotags = [point["hovertext"] for point in click_data["points"]]

        organisms = df_data.filter(
            pl.col("geotag").is_in(selected_geotags)
        ).select(
            pl.col("organisms.organism")
            .cast(pl.Utf8)
            .str.split(r',\s+(?![^()]*\))', inclusive=False) # only splits on commas outside parentheses
            .flatten()
            .drop_nulls()
        ).unique().to_series().to_list()

        return True, click_data, False, selected_geotags, organisms

    # if map selection (box/lasso select) data is available, it overrides map click.
    if selected_data and selected_data != prev_click_data:
        selected_geotags = [point["hovertext"] for point in selected_data["points"]]

        organisms = df_data.filter(
            pl.col("geotag").is_in(selected_geotags)
        ).select(
            pl.col("organisms.organism")
            .cast(pl.Utf8)
            .str.split(r',\s+(?![^()]*\))', inclusive=False) # only splits on commas outside parentheses
            .flatten()
            .drop_nulls()
        ).unique().to_series().to_list()

        return False, selected_data, True, selected_geotags, organisms

    # Default case: no relevant trigger, or data unchanged
    return no_update, prev_click_data, no_update, no_update, no_update




@callback(
    Output("datatable-paging", "data"),
    Output("datatable-paging", "page_count"),
    Output("datatable-paging", "page_current"),

    Input("map-click-flag", "data"),  # determines if triggered by map click or checklist
    Input("map-selection-flag", "data"),  # determines if triggered by map selection or checklist
    Input("sampling-map", "selectedData"),
    Input("sampling-map", "clickData"),
    Input("organism-checklist", "value"),
    Input("common-name-checklist", "value"),
    Input("current-status-checklist", "value"),
    Input("experiment-type-checklist", "value"),
    Input('datatable-paging', "page_current"),
    Input('datatable-paging', "page_size"),
    Input("project-store", "data"),
    Input("selected-geotags", "data")
)
def build_table(click_flag_value, selection_flag_value, selected_data, click_data, selected_organisms,
                selected_common_names, selected_current_status, selected_experiment_types, page_current, page_size,
                project_name, selected_geotags):
    print(f"click_flag_value: {click_flag_value}")

    filtered_df = DATASETS[project_name]["df_data"].lazy()

    if selected_geotags:
        filtered_df = filtered_df.filter(pl.col("geotag").is_in(selected_geotags))

    if selected_organisms:
        filtered_df = filtered_df.filter(pl.col("organisms.organism").is_in(selected_organisms))

    if selected_common_names:
        filtered_df = filtered_df.filter(pl.col("common_name").is_in(selected_common_names))

    if selected_current_status:
        filtered_df = filtered_df.filter(pl.col("current_status").is_in(selected_current_status))

    if selected_experiment_types:
        filtered_df = filtered_df.filter(pl.col("experiment_type").is_in(selected_experiment_types))

    link_prefix = PORTAL_URL_PREFIX.get(project_name, "")

    if project_name == "erga" or project_name == "gbdp":
        url_param = "tax_id"
    else:
        url_param = "organisms.organism"

    # build the complete lazy query pipeline
    processed_df = filtered_df.group_by(['geotag', 'organisms.biosample_id']).agg([
        pl.col("experiment_type")
        .drop_nulls()
        .cast(pl.Utf8)
        .str.strip_chars()
        .unique()
        .str.concat(", ")
        .alias("experiment_type"),

        # take first values for other columns
        pl.col('common_name').first(),
        pl.col('current_status').first(),
        pl.col('symbionts_status').first(),
        pl.col('tax_id').first(),
        pl.col('organisms.organism').first(),
    ]).with_columns([
        # add organism link column
        pl.format(
            f"[{{}}]({link_prefix}{{}})",
            pl.col("organisms.organism"),
            pl.col(url_param)
            .map_elements(lambda x: urllib.parse.quote(str(x)), return_dtype=pl.Utf8)
        ).alias("organism_link")
    ]).unique(subset=[
        # remove duplicates
        "organism_link",
        "common_name",
        "current_status",
        "symbionts_status"
    ]).sort("current_status")

    # get the count for pagination calculation
    total_records = processed_df.select(pl.len()).collect().item()

    # pagination
    total_pages = max(math.ceil(total_records / page_size), 1)
    page_current = min(page_current, total_pages - 1)

    start_idx = page_current * page_size
    paginated_df = processed_df.slice(start_idx, page_size).collect()

    # convert to records for the datatable
    records = paginated_df.to_dicts()

    return (
        records,
        total_pages,
        page_current
    )