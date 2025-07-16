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


def build_taxonomy_tree(flat_records, ranks_order, max_depth=None):
    """
    Build a taxonomy tree from flat records with optional depth limit.

    Args:
        flat_records: List of record dictionaries
        ranks_order: List of taxonomic ranks in order
        max_depth: Maximum depth to build the tree (None for full tree)

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

        # Apply depth limit if specified
        if max_depth is not None:
            branch = branch[:max_depth + 1]  # +1 because we start from root

        current = root
        for name in branch:
            for child in current["children"]:
                if child["name"] == name:
                    current = child
                    break
            else:
                new_node = {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "parent": current["id"],
                    "children": [],
                    "has_unexpanded_children": max_depth is not None and branch.index(name) == len(branch) - 1
                                              and branch.index(name) < len(ranks_order)
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


def expand_node(tree, node_id, records, ranks_order):
    """
    Expand a specific node in the tree by loading its children.

    Args:
        tree: The current tree structure
        node_id: ID of the node to expand
        records: Original records to use for building the subtree
        ranks_order: List of taxonomic ranks in order

    Returns:
        Updated tree with the specified node expanded
    """
    # Build a dictionary of nodes for easy access
    node_dict = build_node_dict(tree)

    # Find the node to expand
    node_to_expand = node_dict.get(node_id)
    if not node_to_expand or not node_to_expand.get("has_unexpanded_children", False):
        return tree  # Node not found or already fully expanded

    # Determine the current depth of the node
    depth = 0
    current = node_to_expand
    while current.get("parent"):
        depth += 1
        current = node_dict.get(current["parent"])

    # Filter records that belong to this branch
    branch_path = []
    current = node_to_expand
    while current:
        branch_path.insert(0, current["name"])
        current = node_dict.get(current.get("parent"))

    # Remove "Eukaryota" from the path if it exists
    if branch_path and branch_path[0] == "Eukaryota":
        branch_path = branch_path[1:]

    # Filter records that match this branch
    filtered_records = []
    for rec in records:
        matches = True
        tree_data = rec.get("phylogenetic_tree", {})

        for i, name in enumerate(branch_path):
            if ":" in name:  # It's a taxonomic rank
                rank, value = name.split(": ", 1)
                if rank.lower() in tree_data:
                    node_data = tree_data[rank.lower()]
                    if node_data.get("scientific_name", "") != value and value != "Not Specified":
                        matches = False
                        break
            elif i == len(branch_path) - 1:  # It's the leaf node (species)
                if rec.get("scientific_name", "") != name and name != "Not Specified":
                    matches = False

        if matches:
            filtered_records.append(rec)

    # Build a subtree for this node with one more level of depth
    subtree = build_taxonomy_tree(filtered_records, ranks_order, max_depth=depth + 1)

    # Replace the node's children with the new subtree's children
    if subtree.get("children"):
        # Find the corresponding node in the subtree
        current_path = branch_path.copy()
        current_subtree = subtree

        while current_path and current_subtree.get("children"):
            name_to_find = current_path[0]
            for child in current_subtree["children"]:
                if child["name"] == name_to_find:
                    current_subtree = child
                    current_path = current_path[1:]
                    break
            else:
                break

        # If we found the node, update its children
        if not current_path:
            node_to_expand["children"] = current_subtree.get("children", [])
            # Update parent references
            for child in node_to_expand["children"]:
                child["parent"] = node_to_expand["id"]

            # Mark as expanded if all children are fully expanded
            node_to_expand["has_unexpanded_children"] = False

            # Check if any children need to be marked as unexpanded
            for child in node_to_expand["children"]:
                if depth + 1 < len(ranks_order):
                    child["has_unexpanded_children"] = True

    return tree


def make_full_elements(tree):
    nodes, edges = [], []

    def dfs(node):
        node_class = "node"
        if node.get("has_unexpanded_children", False):
            node_class = "node unexpanded"

        nodes.append({
            "data": {
                "id": node["id"],
                "label": node["name"],
                "has_unexpanded_children": node.get("has_unexpanded_children", False)
            },
            "classes": node_class
        })

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
    node_class = "node"
    if node.get("has_unexpanded_children", False):
        node_class = "node unexpanded"

    elems = [{
        "data": {
            "id": node["id"],
            "label": node["name"],
            "has_unexpanded_children": node.get("has_unexpanded_children", False)
        },
        "classes": node_class
    }]

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

    # Get the BigQuery view name for the project
    view_name = VIEW_MAPPINGS[project_name]

    # SQL query to select the required columns
    query = f"""
    SELECT
        *
    FROM `{view_name}`
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


def init_project(project_name, initial_depth=2):
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

    # Build tree with limited depth for initial load
    tree = build_taxonomy_tree(records, RANKS, max_depth=initial_depth)
    full_tree_dict = tree
    initial_expanded = [tree["id"]]

    # Store the original records for later use when expanding nodes
    if "original_records" not in DATASETS:
        DATASETS["original_records"] = records

    all_rows = tree_to_table_rows(tree)
    rows_for_table = []
    for row in all_rows:
        sci = row["Scientific name"]
        if ":" not in sci and sci not in ("Not Specified", "Eukaryota"):
            rec = next((r for r in records if r.get("scientific_name") == sci), {})
            link_target = rec.get("tax_id", sci) if project_name in ("erga", "gbdp") else sci
            md_link = make_link(sci, project_name, link_target)
            rows_for_table.append({
                "Scientific name": md_link,
                "Common name": rec.get("common_name", ""),
                "Current Status": rec.get("current_status", ""),
                "Symbionts Status": rec.get("symbionts_status", ""),
                "Phylogeny": row["Phylogeny"],
                "node_id": row["node_id"]
            })

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
            "selector": "node.unexpanded",
            "style": {
                "border-width": 3,
                "border-color": "#FF5733",
                "border-style": "dashed"
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
        return init_project(project_name, initial_depth=2)

    if "clear-sci.n_clicks" in triggered or "clear-common.n_clicks" in triggered:
        if "clear-sci.n_clicks" in triggered:
            sci_sel = []
        if "clear-common.n_clicks" in triggered:
            common_sel = []
        if not sci_sel and not common_sel:
            return init_project(project_name, initial_depth=2)

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

        # Check if this node has unexpanded children
        node = nd.get(nid, {})
        has_unexpanded = node.get("has_unexpanded_children", False)

        if has_unexpanded:
            # Expand this node by loading its children
            original_records = DATASETS.get("original_records", [])
            updated_tree = expand_node(tree_data, nid, original_records, RANKS)

            # Add this node to expanded nodes if not already there
            if nid not in expanded_nodes:
                new_expanded = expanded_nodes + [nid]
            else:
                new_expanded = expanded_nodes

            # Generate new elements with the updated tree
            new_elems = tree_to_elements(updated_tree, new_expanded)

            return (
                no_update,
                no_update,
                no_update,
                no_update,
                updated_tree,  # Update the full tree store with the expanded tree
                no_update,
                new_expanded,
                new_elems,
                no_update,
                no_update,
                no_update,
                no_update
            )
        elif nid in expanded_nodes:
            # Collapse this node
            to_remove = set()
            def collect(x):
                for c in nd[x]["children"]:
                    to_remove.add(c["id"])
                    collect(c["id"])
            collect(nid)
            new_expanded = [e for e in expanded_nodes if e not in to_remove and e != nid]
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
        else:
            # Expand this node (it's already fully loaded)
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
        return init_project(project_name, initial_depth=2)

    filtered_records = [
        rec for rec in records
        if (not sci_sel or any(s in rec["phylogenetic_tree_scientific_names"] for s in sci_sel))
           and (not common_sel or any(c in rec["phylogenetic_tree_common_names"] for c in common_sel))
    ]

    new_tree = build_taxonomy_tree(filtered_records, RANKS)
    new_rows = tree_to_table_rows(new_tree)
    nd_new = build_node_dict(new_tree, {})
    exp_ids = [new_tree["id"]]
    for rec in filtered_records:
        leaf_name = rec["scientific_name"]
        for row in new_rows:
            if row["Scientific name"] == leaf_name:
                cur = row["node_id"]
                while cur:
                    exp_ids.append(cur)
                    cur = nd_new[cur].get("parent")
                break
    new_elems = tree_to_elements(new_tree, exp_ids)

    visible = []
    for row in new_rows:
        if row["node_id"] in exp_ids and not nd_new[row["node_id"]]["children"]:
            sci_plain = row["Scientific name"]
            rec = next((x for x in filtered_records if x["scientific_name"] == sci_plain), {})
            link_target = rec.get("tax_id", sci_plain) if project_name in ("erga", "gbdp") else sci_plain
            md_link = make_link(sci_plain, project_name, link_target)
            visible.append({
                "Scientific name": md_link,
                "Common name": rec.get("common_name", ""),
                "Current Status": rec.get("current_status", ""),
                "Symbionts Status": rec.get("symbionts_status", ""),
                "Phylogeny": row["Phylogeny"]
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
        exp_ids,
        new_elems,
        no_update,
        no_update,
        no_update,
        no_update
    )
