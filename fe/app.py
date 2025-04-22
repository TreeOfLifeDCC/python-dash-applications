import dash
from dash import html
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.MINTY],
    use_pages=True)

app.layout = html.Div([
    dash.page_container
])
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
