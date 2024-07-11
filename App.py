import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

data = pd.read_csv('data/data_clean.csv', delimiter=',', decimal='.')
descriptions = pd.read_csv('data/data_descriptions.csv', delimiter=';')

descriptions_dict = dict(zip(descriptions['Variable'], descriptions['Significado']))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

exclude_columns = ['Customer_ID']

color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

dropdown_options = [{'label': col.capitalize(), 'value': col} for col in sorted(data.columns) if col not in exclude_columns]

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Dataset Viz",), width=12)),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='graph-type',
            options=dropdown_options,
            value='churn'
            ), width=6)]),
    
    dbc.Row(dbc.Col(dcc.Graph(id='main-graph'), width=12)),
    dbc.Row(dbc.Col(html.H2(id='description'), width=6)),
    dbc.Row(dbc.Col(html.H3(id='column-description'), width=12))
], fluid=True)

@app.callback(
    Output('main-graph', 'figure'),
    Output('description', 'children'),
    Output('column-description', 'children'),
    Input('graph-type', 'value')
)

def update_graph(selected_column):
    unique_values = data[selected_column].nunique()
    type = data[selected_column].dtype
    
    column_description = descriptions_dict.get(selected_column, " ")

    if unique_values <= 5:
        fig = px.pie(data, names=selected_column, title=f'Distribución de {selected_column.capitalize()}', hole=0.5, color_discrete_sequence=color_palette)
        description = html.P(f"Este histograma muestra la distribución de la variable {selected_column} in the dataset.", className="text-white")
        
    elif type == 'object':
        fig = px.histogram(data, x=selected_column, title=f'Distribución de {selected_column.capitalize()}')
        description = html.P(f"Este histograma muestra la distribución de la variable {selected_column} en el dataset.", className="text-white")
    
    elif type == 'float64':   
        fig = px.box(data, x=selected_column, orientation = 'h', title=f'Distribución de {selected_column.capitalize()}')
        description = html.P(f"Este boxplot muestra la distribución de la variable {selected_column} en el dataset.", className="text-white")
    
    else:
        fig = px.histogram(data, x=selected_column, title=f'Distribución de {selected_column.capitalize()}', marginal= "box")
        if unique_values > 50:
            low_bound = data[selected_column].min()
            up_bound = data[selected_column].max()
        else:
            Q1 = data[selected_column].quantile(0.25)
            Q3 = data[selected_column].quantile(0.75)
            IQR = Q3 - Q1
            low_bound = Q1 - 1.5 * IQR
            up_bound = Q3 + 1.5 * IQR
        fig.update_layout(
             xaxis=dict(range=[low_bound, up_bound])
        )
        description = html.P(f"Este histograma muestra la distribución de la variable {selected_column} en el dataset.", className="text-white")
    
        
    return fig, description, column_description


if __name__ == '__main__':
    app.run_server(debug=True)