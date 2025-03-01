import pandas as pd
import plotly.graph_objects as go


def plot_for_sample(df_sample: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_sample["token_position"],
        y=df_sample["loss"],
        mode='lines+markers',
        name='Loss',
        hovertemplate='Token: %{customdata[0]}<br>Loss: %{y:.4f}<br>Pos: %{x}',
        customdata=df_sample[['token']].values
    ))

    fig.add_trace(go.Scatter(
        x=df_sample["token_position"],
        y=df_sample["entropy"],
        mode='lines+markers',
        name='Entropy',
        yaxis='y2',
        hovertemplate='Token: %{customdata[0]}<br>Entropy: %{y:.4f}<br>Pos: %{x}',
        customdata=df_sample[['token']].values
    ))

    fig.update_layout(
        title='Loss and Entropy for each token',
        xaxis=dict(title='Token Position'),
        yaxis=dict(title='Loss', side='left'),
        yaxis2=dict(title='Entropy', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99)
    )

    fig.show()