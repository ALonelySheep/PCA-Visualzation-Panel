import random
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np


def _graph_resize(graph):
    graph = graph.reshape((len(graph)//2, 2))
    return graph


def pca(app, model, dataset, latent_options, pre_process=None, prefix=""):
    prefix += '-pca-'

    header = dbc.Row([html.Div(html.H5("Airfoil Parameter Design Visualization Demo"), className="col-md-6")])

    dataset_unscaled = pre_process.inverse_transform(dataset)
    data_pca = model.transform(dataset)

    # Left input Image Display Area
    input_div = dbc.Col(
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Span(children="", id=prefix +
                              'input-content-id', className='p-2 d-grid gap-2'),

                    dbc.Button('Sample', color="primary", id=prefix +
                               'sample-input', className="mr-1 float-right", n_clicks=0),
                ], className="d-grid gap-2 d-md-flex justify-content-between",)
            ]),
            dbc.CardBody(
                [
                    dcc.Graph(
                        # style={ 'width': 200},
                        id=prefix + 'input-content',
                    ),

                ], className="d-flex justify-content-center"
            ),
        ]
        )
    )

    # Middle Slider Area
    latent_size = latent_options['n']
    latent_space = []
    # just used to fill initial space in the html
    init_hidden_space = [str(latent_options['min'])
                         for _ in range(latent_size)]

    for _ in range(latent_size):
        id = prefix + 'latent-slider-' + str(_)
        latent_space.append(dcc.Slider(min=latent_options['min'],
                                       max=latent_options['max'],
                                       marks={latent_options['min']: f"{latent_options['min']}",
                                              0: '0',
                                              latent_options['max']: f"{latent_options['max']}"},
                                       step=latent_options['step'],
                                       updatemode='drag',
                                       id=id,
                                       value=latent_options['min'],
                                       tooltip={"placement": "bottom",
                                                "always_visible": False},
                                       className="mt-3 mb-3"))
        
    latent_div = dbc.Col(dbc.Card([dbc.CardHeader("Parameters"),
                                   html.Span(id=prefix + "hidden-latent-space", children=init_hidden_space,
                                             className='d-none'),
                                   dbc.CardBody([html.Div(children=latent_space, id=prefix + 'output-latent')]), ]))

    # Right Output Display area
    output_div = dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader("Output"),
                dbc.CardBody(
                    [
                        dcc.Graph(
                            # style={ 'width': 200},
                            id=prefix + 'output-content'
                        ),
                        # html.Div(children=[], id=prefix + 'output-content'),
                    ],
                    className="d-flex justify-content-center"
                ),
            ]

        ), class_name="mt-2"
    )

    # '''Image Display'''
    # Click to sample input image

    @app.callback(
        [
            Output(component_id=prefix + 'input-content-id',
                   component_property='children'),
            Output(prefix + 'input-content', 'figure'),
            Output(component_id=prefix + "hidden-latent-space",
                   component_property='children')
        ],
        [Input(component_id=prefix + 'sample-input', component_property='n_clicks')])
    def sample_input(n_clicks):
        # Randomly sample a data from the dataset
        input_id = random.randint(0, len(dataset_unscaled) - 1)
        img = dataset_unscaled[input_id]
        print(f'Airfoil: No. {input_id}')

        # Display the original airfoil
        img = _graph_resize(img)
        graph_img = px.line(x=img[:, 0],
                            y=img[:, 1],
                            # width=300,
                            height=300,
                            template='plotly_white',
                            title='Original Airfoil')
        # Set the x and y axis to be the same scale
        graph_img.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
        )

        hx = data_pca[input_id]

        return "Input ID: " + str(input_id), graph_img, hx

    # Sliders
    #
    for slider_id in range(latent_size):
        @app.callback(
            Output(component_id=prefix + 'latent-slider-' +
                   str(slider_id), component_property="value"),
            [Input(component_id=prefix + "hidden-latent-space",
                   component_property='children')],
            [State(component_id=prefix + 'latent-slider-' + str(slider_id), component_property="id")])
        def set_latent_slider(latent_space, slider_id):
            # print(f"Latent: {latent_space}")
            # print(f"Latent: {type(latent_space)}")
            # print(f"Slider ID: {slider_id}")
            # print(f"Slider ID: {type(slider_id)}")
            slider_id = int(slider_id.split("-")[-1])
            return float(latent_space[slider_id])

    #
    @app.callback(
        Output(component_id=prefix + 'output-content',
               component_property='figure'),
        [Input(component_id=prefix + 'latent-slider-' + str(slider_id), component_property='value')
         for slider_id in range(latent_size)])
    def predicted_output(*latent_space):
        print("Latent Space: ", [f"{i:.2f}" for i in latent_space])
        result_data = np.dot(latent_space, model.components_) + model.mean_
        # print("Latent Space 1: ", result_data.shape)
        result_data = result_data.reshape(1, -1)
        # result_data.reshape(1, -1)
        # print("Latent Space 2: ", result_data.shape)
        result_data = pre_process.inverse_transform(result_data)
        # print("Latent Space 3: ", result_data.shape)
        result_data = result_data.reshape(-1)
        # print("Latent Space 4: ", result_data.shape)

        result_img = _graph_resize(result_data)
        graph_img = px.line(x=result_img[:, 0],
                            y=result_img[:, 1],
                            # width=300,
                            height=300,
                            template='plotly_white',
                            title='Result')
        # Set the x and y axis to be the same scale
        graph_img.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
            ),
        )
        return graph_img

    '''Refresh the model when selected model changes'''
    # @app.callback(
    #     Output(component_id=prefix + 'sample-input',
    #            component_property='n_clicks'),
    #     [Input(component_id=prefix + 'selected-airfoil', component_property='value'),
    #      Input(component_id=prefix + 'reload-model', component_property='n_clicks')],
    #     [State(component_id=prefix + 'sample-input', component_property='n_clicks')])
    # def refresh_model(n_component, reload, n_clicks):
    #     print(f"n_component: {n_component}")
    # #     model.load_state_dict(torch.load(model_path))
    #     return int(n_clicks) + 1

    pca_div = dbc.Card(
        [dbc.CardHeader(header),
         dbc.CardBody(dbc.Row([input_div, latent_div, output_div], className='m-auto'))],
        className="mt-4 mb-4 border-secondary autoencoder-box"
    )
    return pca_div
