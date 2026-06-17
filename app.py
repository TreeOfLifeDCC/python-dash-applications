import os

import dash
from dash import html
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.MINTY],
    use_pages=True,
    suppress_callback_exceptions=True
)

app.layout = html.Div([
    dash.page_container
])
server = app.server

if __name__ == "__main__":
    # Debug mode exposes the Werkzeug interactive debugger (RCE via the console).
    # Keep it off by default; enable explicitly with DASH_DEBUG=1 for local dev only.
    debug = os.getenv("DASH_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run_server(debug=debug)
