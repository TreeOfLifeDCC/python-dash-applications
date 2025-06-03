import uuid
import dash
from dash import dcc, html, Output, Input, State, dash_table, callback_context, no_update
from dash.exceptions import PreventUpdate
import dash_cytoscape as cyto
from dash_extensions import EventListener
import pandas as pd
from google.cloud import storage
import io
import dash_bootstrap_components as dbc
from urllib.parse import quote

dash.register_page(__name__, path="/cytoscape", name="Cytoscape Tree")

PROJECT = "dtol"

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
PROJECT_PARQUET_MAP = {
    "dtol": "python_dash_data_bucket/metadata_dtol*.parquet",
    "erga": "python_dash_data_bucket/metadata_erga*.parquet",
    "asg": "python_dash_data_bucket/metadata_asg*.parquet",
    "gbdp": "python_dash_data_bucket/metadata_gbdp*.parquet"
}

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def make_link(text: str, target: str) -> str:
    prefix = PORTAL_URL_PREFIX.get(PROJECT, "")
    if not prefix:
        return text
    quoted = quote(str(target))
    full_url = f"{prefix}{quoted}"
    return f"[{text}]({full_url})"


def build_taxonomy_tree(flat_records, ranks_order):
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
                    "children": []
                }
                current["children"].append(new_node)
                current = new_node
    return root


def build_node_dict(node, node_dict=None):
    if node_dict is None:
        node_dict = {}
    node_dict[node["id"]] = node
    for c in node.get("children", []):
        build_node_dict(c, node_dict)
    return node_dict


def make_full_elements(tree):
    nodes, edges = [], []

    def dfs(n):
        nodes.append({"data": {"id": n["id"], "label": n["name"]}, "classes": "node"})
        for c in n.get("children", []):
            edges.append({"data": {"source": n["id"], "target": c["id"]}, "classes": "edge"})
            dfs(c)

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
    for c in node.get("children", []):
        rows.extend(tree_to_table_rows(c, current))
    return rows


def tree_to_elements(node, expanded):
    elems = [{"data": {"id": node["id"], "label": node["name"]}, "classes": "node"}]
    if node["id"] in expanded:
        for child in node.get("children", []):
            elems.append(
                {"data": {"source": node["id"], "target": child["id"]}, "classes": "edge"}
            )
            elems.extend(tree_to_elements(child, expanded))
    return elems


def get_subtree_nodes(node_id, elements):
    children_map = {}
    for el in elements:
        d = el["data"]
        if "source" in d and "target" in d:
            children_map.setdefault(d["source"], []).append(d["target"])
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


def load_data(project_name):
    if project_name not in DATASETS:
        gcs_pattern = PROJECT_PARQUET_MAP[project_name]
        bucket, rest = gcs_pattern.split("/", 1)
        prefix = rest.replace("*", "")
        client = storage.Client.create_anonymous_client()
        blobs = client.list_blobs(bucket, prefix=prefix)

        pieces = []
        for blob in blobs:
            if not blob.name.endswith(".parquet"):
                continue
            data = blob.download_as_bytes()
            df_chunk = pd.read_parquet(
                io.BytesIO(data),
                engine="pyarrow",
                columns=[
                    "scientific_name",
                    "common_name",
                    "current_status",
                    "symbionts_status",
                    "phylogenetic_tree",
                    "tax_id",
                ]
            )
            pieces.append(df_chunk)

        if not pieces:
            raise FileNotFoundError(f"No parquet files in gs://{bucket}/{rest}")
        df = pd.concat(pieces, ignore_index=True)

        def extract_names(tree, field):
            return [
                tree.get(rank, {}).get(field, "Not Specified")
                for rank in RANKS
            ]

        df["phylogenetic_tree_scientific_names"] = df["phylogenetic_tree"].apply(
            lambda t: extract_names(t, "scientific_name")
        )
        df["phylogenetic_tree_common_names"] = df["phylogenetic_tree"].apply(
            lambda t: extract_names(t, "common_name")
        )
        DATASETS[project_name] = df

    return DATASETS[project_name]


df = load_data(PROJECT)
records = df.to_dict(orient="records")

all_sci = sorted(
    {
        n
        for sub in df["phylogenetic_tree_scientific_names"]
        for n in sub
        if n and n.lower() != "not specified"
    }
)
all_com = sorted(
    {
        n
        for sub in df["phylogenetic_tree_common_names"]
        for n in sub
        if n and n.lower() != "not specified"
    }
)
checklist_options = [{"label": n, "value": n} for n in all_sci]
common_options = [{"label": n, "value": n} for n in all_com]

tree = build_taxonomy_tree(records, RANKS)
initial_expanded = [tree["id"]]
initial_elements = tree_to_elements(tree, initial_expanded)

all_rows = tree_to_table_rows(tree)
table_rows = []
for row in all_rows:
    sci = row["Scientific name"]
    if ":" not in sci and sci not in ("Not Specified", "Eukaryota"):
        rec = next((r for r in records if r.get("scientific_name") == sci), {})
        if PROJECT in ("erga", "gbdp"):
            link_target = rec.get("tax_id", sci)
        else:
            link_target = sci
        md_link = make_link(sci, link_target)
        table_rows.append({
            "Scientific name": md_link,
            "Common name": rec.get("common_name", ""),
            "Current Status": rec.get("current_status", ""),
            "Symbionts Status": rec.get("symbionts_status", ""),
            "Phylogeny": row["Phylogeny"],
            "node_id": row["node_id"]
        })

header_colour = HEADER_COLOURS.get(PROJECT, "#f1f3f4")

layout = html.Div(className="page-container", children=[

    html.Div(
        style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "10px"},
        children=[
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
                    "cursor": "pointer"
                }
            )
        ]
    ),

    html.Div(
        className="filters-container",
        style={"display": "flex", "gap": "20px", "marginBottom": "20px"},
        children=[

            html.Div(
                style={
                    "border": "1px solid #dee2e6",
                    "borderRadius": "5px",
                    "padding": "10px",
                    "backgroundColor": "#f8f9fa",
                    "flex": "1"
                },
                children=[
                    html.Div(
                        style={"backgroundColor": header_colour, "padding": "10px", "borderRadius": "5px"},
                        children=[
                            dbc.InputGroup([
                                dbc.Input(
                                    id="search-sci",
                                    type="text",
                                    placeholder="Phylogeny Scientific Name",
                                    style={"borderRadius": "20px 0 0 20px", "backgroundColor": "#f1f3f4"}
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
                            ])
                        ]
                    ),
                    dcc.Loading(
                        type="circle",
                        children=html.Div(
                            [
                                dcc.Checklist(
                                    id="chk-sci",
                                    options=checklist_options,
                                    value=[],
                                    inline=False,
                                    style={"display": "flex", "flexDirection": "column", "gap": "7px"},
                                    labelStyle={"display": "flex", "alignItems": "center", "gap": "5px"}
                                )
                            ],
                            style={"overflowY": "auto", "maxHeight": "260px"}
                        )
                    )
                ]
            ),

            html.Div(
                style={
                    "border": "1px solid #dee2e6",
                    "borderRadius": "5px",
                    "padding": "10px",
                    "backgroundColor": "#f8f9fa",
                    "flex": "1"
                },
                children=[
                    html.Div(
                        style={"backgroundColor": header_colour, "padding": "10px", "borderRadius": "5px"},
                        children=[
                            dbc.InputGroup([
                                dbc.Input(
                                    id="search-common",
                                    type="text",
                                    placeholder="Phylogeny Common Name",
                                    style={"borderRadius": "20px 0 0 20px", "backgroundColor": "#f1f3f4"}
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
                            ])
                        ]
                    ),
                    dcc.Loading(
                        type="circle",
                        children=html.Div(
                            [
                                dcc.Checklist(
                                    id="chk-common",
                                    options=common_options,
                                    value=[],
                                    inline=False,
                                    style={"display": "flex", "flexDirection": "column", "gap": "7px"},
                                    labelStyle={"display": "flex", "alignItems": "center", "gap": "5px"}
                                )
                            ],
                            style={"overflowY": "auto", "maxHeight": "260px"}
                        )
                    )
                ]
            ),
        ]
    ),

    dcc.Store(id="filtered-tree", data=tree),
    dcc.Store(id="expanded-store", data=initial_expanded),

    cyto.Cytoscape(
        id="cytoscape-tree",
        elements=initial_elements,
        layout={"name": "breadthfirst", "directed": True, "padding": 10, "animate": True, "animationDuration": 500},
        style={"width": "100%", "height": "568px"},
        stylesheet=[
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "background-color": header_colour,
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
    ),

    EventListener(
        id="cytoscape-listener",
        events=[
            {"event": "tap", "props": ["type", "target.id"]},
            {"event": "dblclick", "props": ["type", "target.id"]}
        ],
        logging=True
    ),

    html.Div(id="node-info", style={"marginBottom": "10px", "fontWeight": "bold"}),

    html.Div(
        className="table-container",
        children=[
            dash_table.DataTable(
                id="tree-table",
                columns=[
                    {
                        "name": "Scientific name",
                        "id": "Scientific name",
                        "presentation": "markdown"
                    },
                    {"name": "Common name", "id": "Common name"},
                    {"name": "Current Status", "id": "Current Status"},
                    {"name": "Symbionts Status", "id": "Symbionts Status"},
                    {"name": "Phylogeny", "id": "Phylogeny"},
                ],
                data=table_rows,
                page_action="native",
                sort_action="native",
                sort_mode="single",
                page_current=0,
                page_size=10,
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#404040", "fontWeight": "bold", "color": "white"},
                style_cell={"textAlign": "left", "padding": "5px"}
            )
        ]
    )
])


@dash.callback(
    Output("node-info", "children"),
    Input("cytoscape-tree", "tapNodeData"),
    State("filtered-tree", "data"),
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
    Output("filtered-tree", "data"),
    Output("tree-table", "data"),
    Output("expanded-store", "data"),
    Output("cytoscape-tree", "elements"),
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
    State("filtered-tree", "data"),
    State("cytoscape-listener", "event"),
    prevent_initial_call=True
)
def master(
        sci_sel, common_sel, search_sci, search_common,
        sub_sci, sub_com, clr_sci, clr_com,
        tap_node, reset_all, dbl_count,
        expanded, tree_data, listener_event
):
    triggered = {t["prop_id"] for t in callback_context.triggered}

    if "cytoscape-listener.n_events" in triggered and listener_event.get("type") == "dblclick":
        if not tap_node or "id" not in tap_node:
            raise PreventUpdate
        node_id = tap_node["id"]
        elems = make_full_elements(tree_data)
        subs = get_subtree_nodes(node_id, elems)
        children_map = {}
        for e in elems:
            d = e["data"]
            if "source" in d and "target" in d:
                children_map.setdefault(d["source"], []).append(d["target"])
        leaf_ids = [n for n in subs if not children_map.get(n)]
        leaf_rows = [r for r in tree_to_table_rows(tree_data) if r["node_id"] in leaf_ids]
        visible = []
        for r in leaf_rows:
            sci_plain = r["Scientific name"]
            rec = next((x for x in records if x["scientific_name"] == sci_plain), {})
            if PROJECT in ("erga", "gbdp"):
                link_target = rec.get("tax_id", sci_plain)
            else:
                link_target = sci_plain
            md_link = make_link(sci_plain, link_target)
            visible.append({
                "Scientific name": md_link,
                "Common name": rec.get("common_name", ""),
                "Current Status": rec.get("current_status", ""),
                "Symbionts Status": rec.get("symbionts_status", ""),
                "Phylogeny": r["Phylogeny"]
            })
        recs = [
            r for r in records
            if r["scientific_name"] in {row["Scientific name"] for row in leaf_rows}
        ]
        sci_vals = sorted(
            {s for r in recs for s in r["phylogenetic_tree_scientific_names"] if s.lower() != "not specified"}
        )
        com_vals = sorted(
            {c for r in recs for c in r["phylogenetic_tree_common_names"] if c.lower() != "not specified"}
        )
        sci_opts = [{"label": s, "value": s} for s in sci_vals]
        com_opts = [{"label": c, "value": c} for c in com_vals]
        return com_opts, sci_opts, com_vals, sci_vals, no_update, visible, no_update, no_update

    ctx = callback_context.triggered[0]
    comp, prop = ctx["prop_id"].split(".")
    sci_sel = [s for s in (sci_sel or []) if s != "Not Specified"]
    common_sel = [c for c in (common_sel or []) if c != "Not Specified"]
    search_sci = search_sci or ""
    search_common = search_common or ""

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
            no_update
        )

    if comp == "clear-sci" and prop == "n_clicks":
        sci_sel = []

    if comp == "clear-common" and prop == "n_clicks":
        common_sel = []

    if comp == "cytoscape-tree" and prop == "tapNodeData" and tap_node:
        nd = build_node_dict(tree_data, {})
        nid = tap_node["id"]
        if nid in expanded:
            to_remove = set()

            def collect(x):
                for c in nd[x]["children"]:
                    to_remove.add(c["id"])
                    collect(c["id"])

            collect(nid)
            new_expanded = [e for e in expanded if e not in to_remove and e != nid]
        else:
            new_expanded = expanded + [nid]
        new_elems = tree_to_elements(tree_data, new_expanded)
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            new_expanded,
            new_elems
        )

    if not sci_sel and not common_sel:
        return (
            common_options,
            checklist_options,
            [],
            [],
            tree,
            table_rows,
            initial_expanded,
            initial_elements
        )

    if comp == "reset-all" and prop == "n_clicks":
        return (
            common_options,
            checklist_options,
            [],
            [],
            tree,
            table_rows,
            initial_expanded,
            initial_elements
        )

    filtered = [
        rec for rec in records
        if (not sci_sel or any(s in rec["phylogenetic_tree_scientific_names"] for s in sci_sel))
           and (not common_sel or any(c in rec["phylogenetic_tree_common_names"] for c in common_sel))
    ]

    new_tree = build_taxonomy_tree(filtered, RANKS)
    new_rows = tree_to_table_rows(new_tree)
    nd_new = build_node_dict(new_tree, {})
    exp_ids = {new_tree["id"]}
    for rec in filtered:
        leaf = rec["scientific_name"]
        for r in new_rows:
            if r["Scientific name"] == leaf:
                cur = r["node_id"]
                while cur:
                    exp_ids.add(cur)
                    cur = nd_new[cur].get("parent")
                break
    new_elems = tree_to_elements(new_tree, exp_ids)

    visible = []
    for r in new_rows:
        if r["node_id"] in exp_ids and not nd_new[r["node_id"]]["children"]:
            sci_plain = r["Scientific name"]
            rec = next((x for x in filtered if x["scientific_name"] == sci_plain), {})
            if PROJECT in ("erga", "gbdp"):
                link_target = rec.get("tax_id", sci_plain)
            else:
                link_target = sci_plain
            md_link = make_link(sci_plain, link_target)
            visible.append({
                "Scientific name": md_link,
                "Common name": rec.get("common_name", ""),
                "Current Status": rec.get("current_status", ""),
                "Symbionts Status": rec.get("symbionts_status", ""),
                "Phylogeny": r["Phylogeny"]
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

    return (
        [{"label": c, "value": c} for c in sorted(allowed_com)],
        [{"label": s, "value": s} for s in sorted(allowed_sci)],
        common_sel,
        sci_sel,
        new_tree,
        visible,
        list(exp_ids),
        new_elems
    )
