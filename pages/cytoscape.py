import uuid
import dash
from dash import dcc, html, Output, Input, State, dash_table, callback_context, no_update
from dash.exceptions import PreventUpdate
import dash_cytoscape as cyto
from dash_extensions import EventListener
import io
import dash_bootstrap_components as dbc
from urllib.parse import quote, parse_qs
import polars as pl
import json
from google.cloud import bigquery
from google.oauth2 import service_account

dash.register_page(__name__, path="/cytoscape", name="Cytoscape Tree")

HEADER_COLOURS = {
    "dtol": "#8fbc45",
    "erga": "#e0efea",
    "asg": "#add8e6",
    "gbdp": "#d0d0ce"
}

PORTAL_URL_PREFIX = {
    "dtol": "https://portal.darwintreeoflife.org/data/",
    "erga": "https://portal.erga-biodiversity.eu/data_portal/",
    "asg": "https://portal.aquaticsymbiosisgenomics.org/data/root/details/",
    "gbdp": "https://www.ebi.ac.uk/biodiversity/data_portal/"
}

DATASETS = {}


# BigQuery view mapping for each project
VIEW_MAPPINGS = {
        "dtol": "prj-ext-prod-biodiv-data-in.dtol.phylogeny_tree_metadata_aggregated",
        "erga": "prj-ext-prod-biodiv-data-in.erga.phylogeny_tree_metadata_aggregated",
        "asg": "prj-ext-prod-biodiv-data-in.asg.phylogeny_tree_metadata_aggregated",
        "gbdp": "prj-ext-prod-biodiv-data-in.gbdp.phylogeny_tree_metadata_aggregated"

}


client = bigquery.Client(
    project="prj-ext-prod-biodiv-data-in"
)
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def make_link(text: str, project_name: str, target: str) -> str:
    prefix = PORTAL_URL_PREFIX.get(project_name, "")
    if not prefix:
        return text
    quoted = quote(str(target))
    full_url = f"{prefix}{quoted}"
    return f"[{text}]({full_url})"


def build_taxonomy_tree(flat_records, ranks_order, max_depth=None, target_node=None):
    """
    Build a taxonomy tree from flat records with an option to limit the depth.

    Args:
        flat_records: List of record dictionaries
        ranks_order: List of taxonomic ranks in order
        max_depth: Maximum depth to build the tree (None for full tree)
        target_node: If specified, only expand this node and its children

    Returns:
        Root node of the tree
    """
    root = {"id": str(uuid.uuid4()), "name": "Eukaryota", "parent": None, "children": []}
    for rec in flat_records:
        branch = []
        tree = rec.get("phylogenetic_tree", {})
        for rank in ranks_order:
            node = tree.get(rank, {})
            sci = node.get("scientific_name", "")
            branch.append(
                f"{rank}: {sci}"
                if sci and sci.lower() != "not specified"
                else "Not Specified"
            )
        sci_leaf = rec.get("scientific_name", "")
        branch.append(
            sci_leaf if sci_leaf and sci_leaf.lower() != "not specified" else "Not Specified"
        )

        # Check if we're targeting a specific node
        target_index = -1
        if target_node is not None:
            try:
                target_index = branch.index(target_node)
            except ValueError:
                # Target node not in this branch, skip to next record
                continue

        # Only process up to max_depth if specified and not targeting a specific node
        if max_depth is not None and target_node is None:
            branch = branch[:max_depth + 1]  # +1 for root level

        # If targeting a specific node, ensure we process at least up to that node plus one level
        if target_index >= 0:
            # We want to process the target node and its immediate children
            min_depth = target_index + 1
            if max_depth is not None:
                max_depth = max(max_depth, min_depth)

        current = root
        for i, name in enumerate(branch):
            # Stop if we've reached max_depth and not targeting this specific node
            if max_depth is not None and i > max_depth and (target_node is None or i != target_index + 1):
                break

            for child in current["children"]:
                if child["name"] == name:
                    current = child
                    break
            else:
                new_node = {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "parent": current["id"],
                    "children": []
                }
                current["children"].append(new_node)
                current = new_node
    return root


def build_node_dict(node, node_dict=None):
    if node_dict is None:
        node_dict = {}
    node_dict[node["id"]] = node
    for child_node in node.get("children", []):
        build_node_dict(child_node, node_dict)
    return node_dict


def make_full_elements(tree):
    nodes, edges = [], []

    def dfs(node):
        nodes.append({"data": {"id": node["id"], "label": node["name"]}, "classes": "node"})
        for child_node in node.get("children", []):
            edges.append({"data": {"source": node["id"], "target": child_node["id"]}, "classes": "edge"})
            dfs(child_node)

    dfs(tree)
    return nodes + edges


def tree_to_table_rows(node, path=None):
    if path is None:
        path = []
    current = path + [node["name"]]
    rows = [
        {
            "Scientific name": node["name"],
            "Phylogeny": " → ".join(current),
            "node_id": node["id"],
        }
    ]
    for child_node in node.get("children", []):
        rows.extend(tree_to_table_rows(child_node, current))
    return rows


def tree_to_elements(node, expanded):
    elems = [{"data": {"id": node["id"], "label": node["name"]}, "classes": "node"}]
    if node["id"] in expanded:
        for child_node in node.get("children", []):
            elems.append(
                {"data": {"source": node["id"], "target": child_node["id"]}, "classes": "edge"}
            )
            elems.extend(tree_to_elements(child_node, expanded))
    return elems


def get_subtree_nodes(node_id, elements):
    children_map = {}
    for el in elements:
        data = el["data"]
        if "source" in data and "target" in data:
            children_map.setdefault(data["source"], []).append(data["target"])
    subtree_ids, stack, visited = [], [node_id], set()
    while stack:
        curr = stack.pop()
        if curr in visited:
            continue
        visited.add(curr)
        subtree_ids.append(curr)
        for child in children_map.get(curr, []):
            if child not in visited:
                stack.append(child)
    return subtree_ids


def load_data_polars(project_name):
    if project_name in DATASETS:
        return DATASETS[project_name]

    # Get the BigQuery table name for the project
    view_name = VIEW_MAPPINGS[project_name]

    # SQL query to select the required columns
    query = f"""
    SELECT
        scientific_name,
        common_name,
        current_status,
        symbionts_status,
        phylogenetic_tree,
        tax_id
    FROM
        `{view_name}`
    """


    job_config = bigquery.QueryJobConfig(
        use_query_cache=True,
        use_legacy_sql=False,
        maximum_bytes_billed=10 ** 12
    )

    # Execute the query
    query_job = client.query(query, job_config=job_config)

    # Convert the results to a pandas DataFrame
    pandas_df = query_job.to_dataframe(create_bqstorage_client=True)

    # Convert pandas DataFrame to polars DataFrame
    df = pl.from_pandas(pandas_df)

    # Process the phylogenetic_tree column if it's a string
    if df["phylogenetic_tree"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("phylogenetic_tree")
            .apply(json.loads)
            .alias("phylogenetic_tree")
        )

    # Extract taxonomic ranks
    for rank in RANKS:
        df = df.with_columns([
            pl.col("phylogenetic_tree")
              .map_elements(lambda d: d.get(rank, {}).get("scientific_name", "Not Specified"),
                            return_dtype=pl.Utf8)
              .alias(f"{rank}_sci"),
            pl.col("phylogenetic_tree")
              .map_elements(lambda d: d.get(rank, {}).get("common_name", "Not Specified"),
                            return_dtype=pl.Utf8)
              .alias(f"{rank}_com"),
        ])

    # Create concatenated lists for scientific and common names
    df = df.with_columns([
        pl.concat_list([pl.col(f"{r}_sci") for r in RANKS])
        .alias("phylogenetic_tree_scientific_names"),

        pl.concat_list([pl.col(f"{r}_com") for r in RANKS])
        .alias("phylogenetic_tree_common_names"),
    ])

    # Cache the result
    DATASETS[project_name] = df
    return DATASETS[project_name]


def init_project(project_name):
    df = load_data_polars(project_name)
    records = df.to_dicts()

    all_sci = sorted({
        name
        for subtree in df["phylogenetic_tree_scientific_names"]
        for name in subtree
        if name and name.lower() != "not specified"
    })
    all_com = sorted({
        name
        for subtree in df["phylogenetic_tree_common_names"]
        for name in subtree
        if name and name.lower() != "not specified"
    })
    sci_options = [{"label": name, "value": name} for name in all_sci]
    com_options = [{"label": name, "value": name} for name in all_com]

    # Build the full tree
    tree = build_taxonomy_tree(records, RANKS)
    full_tree_dict = tree  # Store the full tree
    initial_expanded = [tree["id"]]

    # Load all data into the table
    rows_for_table = []
    for rec in records:
        sci = rec.get("scientific_name", "")
        if sci and sci not in ("Not Specified", "Eukaryota"):
            # Create phylogeny path for this record
            phylogeny_path = ["Eukaryota"]
            for rank in RANKS:
                rank_sci = rec.get("phylogenetic_tree", {}).get(rank, {}).get("scientific_name", "")
                if rank_sci and rank_sci.lower() != "not specified":
                    phylogeny_path.append(f"{rank}: {rank_sci}")
            if sci and sci.lower() != "not specified":
                phylogeny_path.append(sci)

            # Create table row
            link_target = rec.get("tax_id", sci) if project_name in ("erga", "gbdp") else sci
            md_link = make_link(sci, project_name, link_target)
            rows_for_table.append({
                "Scientific name": md_link,
                "Common name": rec.get("common_name", ""),
                "Current Status": rec.get("current_status", ""),
                "Symbionts Status": rec.get("symbionts_status", ""),
                "Phylogeny": " → ".join(phylogeny_path),
                "node_id": ""  # No direct node ID since it might not be in the partial tree
            })

    # Only load the root node and its immediate children for the tree visualization
    elements = tree_to_elements(tree, initial_expanded)

    color = HEADER_COLOURS.get(project_name, "#cccccc")
    stylesheet = [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "background-color": color,
                "width": 60,
                "height": 60,
                "font-size": "14px",
                "text-valign": "center",
                "text-halign": "center"
            }
        },
        {
            "selector": "edge",
            "style": {
                "line-color": "#A3C4BC",
                "width": 1
            }
        }
    ]

    table_header_style = {"backgroundColor": color, "color": "black", "textAlign": "center"}
    sci_filter_header_style = {"backgroundColor": color, "padding": "10px", "borderRadius": "5px"}
    common_filter_header_style = {"backgroundColor": color, "padding": "10px", "borderRadius": "5px"}

    return (
        com_options,
        sci_options,
        [],
        [],
        full_tree_dict,
        rows_for_table,
        initial_expanded,
        elements,
        stylesheet,
        table_header_style,
        sci_filter_header_style,
        common_filter_header_style
    )


layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="project-store", data="dtol"),
    dbc.Container(
        fluid=True,
        style={"maxWidth": "1600px", "paddingTop": "20px", "paddingBottom": "20px"},
        children=[
            dbc.Row(
                align="center",
                className="mb-3",
                children=[
                    dbc.Col(
                        html.Div(
                            [
                                "• Single click: expand or collapse the node",
                                html.Br(),
                                "• Double click: display the leaves of the selected node’s branch in the table and show the corresponding filters"
                            ],
                            style={
                                "fontSize": "16px",
                                "fontWeight": "bold",
                                "color": "#333",
                                "backgroundColor": "#f0f0f0",
                                "padding": "8px 12px",
                                "borderRadius": "4px"
                            }
                        ),
                        width=12
                    )
                ]
            ),
            dbc.Row(
                className="mb-4",
                children=[
                    dbc.Col(
                        html.Div(
                            id="sci-filter-container",
                            style={
                                "border": "1px solid #dee2e6",
                                "borderRadius": "5px",
                                "backgroundColor": "#f8f9fa",
                                "padding": "10px"
                            },
                            children=[
                                html.Div(
                                    id="sci-filter-header",
                                    style={"backgroundColor": "#ffffff", "padding": "10px", "borderRadius": "5px"},
                                    children=[
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(
                                                    id="search-sci",
                                                    type="text",
                                                    placeholder="Phylogeny Scientific Name",
                                                    style={
                                                        "borderRadius": "20px 0 0 20px",
                                                        "backgroundColor": "#f1f3f4"
                                                    }
                                                ),
                                                dbc.Button(
                                                    "Clear",
                                                    id="clear-sci",
                                                    color="dark",
                                                    size="sm",
                                                    style={
                                                        "borderRadius": "0 20px 20px 0",
                                                        "color": "#fff",
                                                        "boxShadow": "0 0 4px rgba(0,0,0,0.3)"
                                                    }
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                dcc.Loading(
                                    type="circle",
                                    children=html.Div(
                                        dcc.Checklist(
                                            id="chk-sci",
                                            options=[],
                                            value=[],
                                            inline=False,
                                            style={"display": "flex", "flexDirection": "column", "gap": "7px"},
                                            labelStyle={"display": "flex", "alignItems": "center", "gap": "5px"}
                                        ),
                                        style={"overflowY": "auto", "maxHeight": "260px", "marginTop": "8px"}
                                    )
                                )
                            ]
                        ),
                        md=6
                    ),
                    dbc.Col(
                        html.Div(
                            id="common-filter-container",
                            style={
                                "border": "1px solid #dee2e6",
                                "borderRadius": "5px",
                                "backgroundColor": "#f8f9fa",
                                "padding": "10px"
                            },
                            children=[
                                html.Div(
                                    id="common-filter-header",
                                    style={"backgroundColor": "#ffffff", "padding": "10px", "borderRadius": "5px"},
                                    children=[
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(
                                                    id="search-common",
                                                    type="text",
                                                    placeholder="Phylogeny Common Name",
                                                    style={
                                                        "borderRadius": "20px 0 0 20px",
                                                        "backgroundColor": "#f1f3f4"
                                                    }
                                                ),
                                                dbc.Button(
                                                    "Clear",
                                                    id="clear-common",
                                                    color="dark",
                                                    size="sm",
                                                    style={
                                                        "borderRadius": "0 20px 20px 0",
                                                        "color": "#fff",
                                                        "boxShadow": "0 0 4px rgba(0,0,0,0.3)"
                                                    }
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                dcc.Loading(
                                    type="circle",
                                    children=html.Div(
                                        dcc.Checklist(
                                            id="chk-common",
                                            options=[],
                                            value=[],
                                            inline=False,
                                            style={"display": "flex", "flexDirection": "column", "gap": "7px"},
                                            labelStyle={"display": "flex", "alignItems": "center", "gap": "5px"}
                                        ),
                                        style={"overflowY": "auto", "maxHeight": "260px", "marginTop": "8px"}
                                    )
                                )
                            ]
                        ),
                        md=6
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        cyto.Cytoscape(
                            id="cytoscape-tree",
                            elements=[],
                            layout={
                                "name": "breadthfirst",
                                "directed": True,
                                "padding": 10,
                                "animate": True,
                                "animationDuration": 500
                            },
                            style={"width": "100%", "height": "568px"},
                            stylesheet=[]
                        ),
                        width=12
                    ),
                    dbc.Col(
                        EventListener(
                            id="cytoscape-listener",
                            events=[
                                {"event": "tap", "props": ["type", "target.id"]},
                                {"event": "dblclick", "props": ["type", "target.id"]}
                            ],
                            logging=True
                        ),
                        width=0
                    )
                ],
                className="mb-3"
            ),
            dbc.Row(
                dbc.Col(
                    html.Button(
                        "Reset All",
                        id="reset-all",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#F28265",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "8px",
                            "padding": "8px 16px",
                            "fontSize": "14px",
                            "cursor": "pointer",
                            "marginBottom": "10px"
                        }
                    ),
                    width=12,
                    style={"textAlign": "right"}
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                id="node-info",
                                style={
                                    "fontWeight": "bold",
                                    "marginBottom": "10px",
                                    "textAlign": "left",
                                    "color": "#333"
                                }
                            ),
                            html.Div(
                                dash_table.DataTable(
                                    id="tree-table",
                                    columns=[
                                        {"name": "Scientific name", "id": "Scientific name", "presentation": "markdown"},
                                        {"name": "Common name", "id": "Common name"},
                                        {"name": "Current Status", "id": "Current Status"},
                                        {"name": "Symbionts Status", "id": "Symbionts Status"},
                                        {"name": "Phylogeny", "id": "Phylogeny"},
                                    ],
                                    data=[],
                                    page_action="native",
                                    sort_action="none",
                                    page_current=0,
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_header={"textAlign": "center"},
                                    style_cell={
                                        "padding": "5px",
                                        "textAlign": "center",
                                        "verticalAlign": "middle",
                                        "whiteSpace": "normal"
                                    },
                                    css=[{"selector": "a", "rule": "text-decoration: none !important;"}]
                                ),
                                style={
                                    "background": "#fff",
                                    "border": "1px solid #ddd",
                                    "borderRadius": "6px",
                                    "boxShadow": "0 2px 4px rgba(0,0,0,0.08)",
                                    "padding": "16px",
                                    "marginBottom": "20px",
                                }
                            )
                        ]
                    ),
                    width=12
                )
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Store(id="full-tree-store", data={}), width=0),
                    dbc.Col(dcc.Store(id="expanded-store", data=[]), width=0)
                ]
            )
        ]
    )
])


@dash.callback(
    Output("project-store", "data"),
    Input("url", "search")
)
def update_project_from_url(search_string):
    if not search_string:
        return "dtol"
    parsed = parse_qs(search_string.lstrip("?"))
    project_name = parsed.get("projectName", ["dtol"])[0]
    if project_name not in VIEW_MAPPINGS:
        return "dtol"
    return project_name


@dash.callback(
    Output("node-info", "children"),
    Input("cytoscape-tree", "tapNodeData"),
    State("full-tree-store", "data"),
    prevent_initial_call=True
)
def update_node_info(data, tree_data):
    if not data or "id" not in data:
        raise PreventUpdate
    nd = build_node_dict(tree_data, {})
    path, cur = [], data["id"]
    while cur:
        path.insert(0, nd[cur]["name"])
        cur = nd[cur].get("parent")
    return " → ".join(path)


@dash.callback(
    Output("chk-common", "options"),
    Output("chk-sci", "options"),
    Output("chk-common", "value"),
    Output("chk-sci", "value"),
    Output("full-tree-store", "data"),
    Output("tree-table", "data"),
    Output("expanded-store", "data"),
    Output("cytoscape-tree", "elements"),
    Output("cytoscape-tree", "stylesheet"),
    Output("tree-table", "style_header"),
    Output("sci-filter-header", "style"),
    Output("common-filter-header", "style"),
    Input("project-store", "data"),
    Input("chk-sci", "value"),
    Input("chk-common", "value"),
    Input("search-sci", "value"),
    Input("search-common", "value"),
    Input("search-sci", "n_submit"),
    Input("search-common", "n_submit"),
    Input("clear-sci", "n_clicks"),
    Input("clear-common", "n_clicks"),
    Input("cytoscape-tree", "tapNodeData"),
    Input("reset-all", "n_clicks"),
    Input("cytoscape-listener", "n_events"),
    State("expanded-store", "data"),
    State("full-tree-store", "data"),
    State("cytoscape-listener", "event"),
    prevent_initial_call=True
)
def master(
        project_name,
        sci_sel, common_sel, search_sci, search_common,
        sub_sci, sub_com, clr_sci, clr_com,
        tap_node, reset_all, dbl_count,
        expanded_nodes, full_tree_data, listener_event
):
    triggered = {t["prop_id"] for t in callback_context.triggered}

    if "project-store.data" in triggered or "reset-all.n_clicks" in triggered:
        return init_project(project_name)

    if "clear-sci.n_clicks" in triggered or "clear-common.n_clicks" in triggered:
        if "clear-sci.n_clicks" in triggered:
            sci_sel = []
        if "clear-common.n_clicks" in triggered:
            common_sel = []
        if not sci_sel and not common_sel:
            return init_project(project_name)

    records_df = DATASETS.get(project_name)
    if records_df is None:
        records_df = load_data_polars(project_name)
    records = records_df.to_dicts()
    tree_data = full_tree_data

    sci_sel = [s for s in (sci_sel or []) if s != "Not Specified"]
    common_sel = [c for c in (common_sel or []) if c != "Not Specified"]
    search_sci = search_sci or ""
    search_common = search_common or ""

    if "cytoscape-listener.n_events" in triggered and listener_event.get("type") == "dblclick":
        if not tap_node or "id" not in tap_node:
            raise PreventUpdate
        node_id = tap_node["id"]
        elems = make_full_elements(tree_data)
        subs = get_subtree_nodes(node_id, elems)
        children_map = {}
        for el in elems:
            data = el["data"]
            if "source" in data and "target" in data:
                children_map.setdefault(data["source"], []).append(data["target"])
        leaf_ids = [nid for nid in subs if not children_map.get(nid)]
        leaf_rows = [r for r in tree_to_table_rows(tree_data) if r["node_id"] in leaf_ids]

        visible = []
        for row in leaf_rows:
            sci_plain = row["Scientific name"]
            rec = next((x for x in records if x["scientific_name"] == sci_plain), {})
            link_target = rec.get("tax_id", sci_plain) if project_name in ("erga", "gbdp") else sci_plain
            md_link = make_link(sci_plain, project_name, link_target)
            visible.append({
                "Scientific name": md_link,
                "Common name": rec.get("common_name", ""),
                "Current Status": rec.get("current_status", ""),
                "Symbionts Status": rec.get("symbionts_status", ""),
                "Phylogeny": row["Phylogeny"]
            })

        subtree_recs = [
            rec for rec in records
            if rec["scientific_name"] in {row["Scientific name"] for row in leaf_rows}
        ]
        sci_vals = sorted({
            sci_val
            for record_data in subtree_recs
            for sci_val in record_data["phylogenetic_tree_scientific_names"]
            if sci_val.lower() != "not specified"
        })
        com_vals = sorted({
            common_val
            for record_data in subtree_recs
            for common_val in record_data["phylogenetic_tree_common_names"]
            if common_val.lower() != "not specified"
        })
        sci_opts = [{"label": s, "value": s} for s in sci_vals]
        com_opts = [{"label": c, "value": c} for c in com_vals]

        return (
            com_opts,
            sci_opts,
            com_vals,
            sci_vals,
            no_update,
            visible,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update
        )

    if "cytoscape-tree.tapNodeData" in triggered and tap_node:
        nd = build_node_dict(tree_data, {})
        nid = tap_node["id"]

        # Check if this node is at the maximum depth and has no children loaded yet
        node = nd.get(nid, {})
        if not node.get("children") and node.get("name") != "Not Specified":
            # This node might be at the maximum depth, try to expand it
            # Find the corresponding record in the original data
            node_name = node.get("name", "")

            # If this is a leaf node (scientific name without rank prefix)
            if ":" not in node_name and node_name not in ("Eukaryota", "Not Specified"):
                # Find the record for this scientific name
                matching_record = next((r for r in records if r.get("scientific_name") == node_name), None)

                if matching_record:
                    # Build a subtree for this record, focusing on expanding this specific node
                    subtree = build_taxonomy_tree([matching_record], RANKS, target_node=node_name)

                    # Find the path to this node in the current tree
                    path = []
                    current = node
                    while current.get("parent"):
                        parent_id = current.get("parent")
                        parent = nd.get(parent_id)
                        if parent:
                            path.insert(0, parent)
                            current = parent
                        else:
                            break

                    # Find the corresponding node in the subtree
                    current_subtree = subtree
                    for p in path:
                        for child in current_subtree.get("children", []):
                            if child.get("name") == p.get("name"):
                                current_subtree = child
                                break

                    # Add children from the subtree to the current node
                    node["children"] = current_subtree.get("children", [])
                    for child in node["children"]:
                        child["parent"] = node["id"]

        if nid in expanded_nodes:
            to_remove = set()
            def collect(x):
                for c in nd[x]["children"]:
                    to_remove.add(c["id"])
                    collect(c["id"])
            collect(nid)
            new_expanded = [e for e in expanded_nodes if e not in to_remove and e != nid]
        else:
            new_expanded = expanded_nodes + [nid]
        new_elems = tree_to_elements(tree_data, new_expanded)
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            new_expanded,
            new_elems,
            no_update,
            no_update,
            no_update,
            no_update
        )

    ctx = callback_context.triggered[0]
    comp, prop = ctx["prop_id"].split(".")
    if comp == "search-sci" and prop in ("value", "n_submit"):
        allowed = {
            sci
            for rec in records
            for sci, cm in zip(
                rec["phylogenetic_tree_scientific_names"],
                rec["phylogenetic_tree_common_names"]
            )
            if (not common_sel or cm in common_sel) and sci and sci.lower() != "not specified"
        }
        if search_sci:
            ss = search_sci.lower()
            allowed = {s for s in allowed if ss in s.lower()}
        return (
            no_update,
            [{"label": s, "value": s} for s in sorted(allowed)],
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update
        )

    if comp == "search-common" and prop in ("value", "n_submit"):
        allowed = {
            cm
            for rec in records
            for sci, cm in zip(
                rec["phylogenetic_tree_scientific_names"],
                rec["phylogenetic_tree_common_names"]
            )
            if (not sci_sel or sci in sci_sel) and cm and cm.lower() != "not specified"
        }
        if search_common:
            sc = search_common.lower()
            allowed = {c for c in allowed if sc in c.lower()}
        return (
            [{"label": c, "value": c} for c in sorted(allowed)],
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update
        )

    if not sci_sel and not common_sel:
        return init_project(project_name)

    filtered_records = [
        rec for rec in records
        if (not sci_sel or any(s in rec["phylogenetic_tree_scientific_names"] for s in sci_sel))
           and (not common_sel or any(c in rec["phylogenetic_tree_common_names"] for c in common_sel))
    ]

    # Build a partial tree for filtered results too
    new_tree = build_taxonomy_tree(filtered_records, RANKS)
    new_elems = tree_to_elements(new_tree, [new_tree["id"]])

    # Create table rows directly from filtered records
    visible = []
    for rec in filtered_records:
        sci = rec.get("scientific_name", "")
        if sci and sci not in ("Not Specified", "Eukaryota"):
            # Create phylogeny path for this record
            phylogeny_path = ["Eukaryota"]
            for rank in RANKS:
                rank_sci = rec.get("phylogenetic_tree", {}).get(rank, {}).get("scientific_name", "")
                if rank_sci and rank_sci.lower() != "not specified":
                    phylogeny_path.append(f"{rank}: {rank_sci}")
            if sci and sci.lower() != "not specified":
                phylogeny_path.append(sci)

            # Create table row
            link_target = rec.get("tax_id", sci) if project_name in ("erga", "gbdp") else sci
            md_link = make_link(sci, project_name, link_target)
            visible.append({
                "Scientific name": md_link,
                "Common name": rec.get("common_name", ""),
                "Current Status": rec.get("current_status", ""),
                "Symbionts Status": rec.get("symbionts_status", ""),
                "Phylogeny": " → ".join(phylogeny_path)
            })

    allowed_sci = {
        sci for rec in records for sci, cm in zip(
            rec["phylogenetic_tree_scientific_names"],
            rec["phylogenetic_tree_common_names"]
        ) if (not common_sel or cm in common_sel) and sci and sci.lower() != "not specified"
    }
    allowed_com = {
        cm for rec in records for sci, cm in zip(
            rec["phylogenetic_tree_scientific_names"],
            rec["phylogenetic_tree_common_names"]
        ) if (not sci_sel or sci in sci_sel) and cm and cm.lower() != "not specified"
    }

    sci_options = [{"label": s_val, "value": s_val} for s_val in sorted(allowed_sci)]
    com_options = [{"label": c_val, "value": c_val} for c_val in sorted(allowed_com)]

    return (
        com_options,
        sci_options,
        common_sel,
        sci_sel,
        new_tree,
        visible,
        [new_tree["id"]],  # Only root node is expanded initially
        new_elems,
        no_update,
        no_update,
        no_update,
        no_update
    )
