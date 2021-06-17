
import os
import csv
import pandas as pd
from PIL import Image
import terato.display as Tox
import torch
from torchvision import transforms

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import numpy as np
import altair as alt
import prince
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Save a dose vs death probability chart for each compound


# Save a dose vs phenotype for each compound
def doseresponse(df, path):
  path = Path(path)
  feno_names = ['bodycurvature', 'yolkedema', 'necrosis',
                           'tailbending', 'craniofacialedema', 'finabsence', 'scoliosis',
                           'snoutjawdefects', 'otolithsdefects']
  alt.data_transformers.disable_max_rows()
  selector  = alt.selection_single(fields = ['feno'], bind = "legend")

  compounds = df['compound'].drop_duplicates()
  compounds = compounds.to_numpy()

  baseline = alt.Chart(df).mark_rule(color = "black").transform_filter(alt.datum.compound == "DMSO").transform_fold(
        fold = ['bodycurvature', 'yolkedema', 'necrosis',
                           'tailbending', 'craniofacialedema', 'finabsence', 'scoliosis',
                           'snoutjawdefects', 'otolithsdefects'],
         as_ = ['feno', 'value']
  ).transform_aggregate(
       count = "count()",
       ill = 'sum(value)',
       groupby=["feno"]
  ).transform_calculate(
       prob_ill = "datum.ill/datum.count"
  ).encode(
       y = alt.Y("prob_ill:Q", title = 'Illness probability', axis=alt.Axis(format='.2%')),
       # color = alt.Color("feno:N", scale = alt.Scale(domain= feno_names))
  ).transform_filter(selector)


  compounds = np.delete(compounds, 0)
  for c in compounds:
    dosevsdeath = alt.Chart(df).mark_bar().transform_filter(alt.datum.compound == c).transform_fold(
        fold = ['dead24', 'dead120'],
        as_ = ['type_death', 'dead']
        ).transform_aggregate(
           dosecount = 'count()',
           death = 'sum(dead)',
           groupby=["dose",  "type_death"]
        ).transform_calculate(
           prob_death = '(datum.death / datum.dosecount)'
        ).encode(
           x = alt.X("dose:N", title = 'Applied doses'),
           y = alt.Y("prob_death:Q", scale=alt.Scale(domain=[0, 1]), title = 'Death probability'),
           color = alt.Color("type_death:N", title = 'Time of death'),
           tooltip = ["dosecount:Q", "death:Q", "prob_death:Q", "type_death:N"]
        ).properties(title = c)

    doseresponse = alt.Chart(df).mark_bar().transform_filter(alt.datum.compound == c).transform_fold(
        fold = ['bodycurvature', 'yolkedema', 'necrosis',
                'tailbending', 'craniofacialedema', 'finabsence', 'scoliosis',
                'snoutjawdefects', 'otolithsdefects'],
        as_ = ['feno', 'value']
        ).transform_aggregate(
            dosecount = 'count()',
            ill = 'sum(value)',
            groupby=["dose",  "feno"]
        ).transform_calculate(
            prob_ill = '(datum.ill / datum.dosecount)'
        ).encode(
            x = alt.X("dose:N", title = 'Applied doses'),
            y = alt.Y("prob_ill:Q", title = 'Illness probability', axis=alt.Axis(format='.2%'), stack = None),
            color = alt.Color("feno:N", scale = alt.Scale(domain= feno_names)),
            tooltip = ["dosecount:Q", "feno:N", "prob_ill:Q"]
        ).transform_filter(selector).add_selection(selector).properties(title = c)
    chart = doseresponse + baseline
    graphic = alt.hconcat(dosevsdeath, chart).resolve_scale(color='independent').configure_view(stroke=None)
    graphic.save(str(path / (c + ".html")))


def plot_coordinates(mca, X, ax=None, figsize=(8, 8), x_component=0, y_component=1, show_column_points=True, column_points_size=30, show_column_labels=True, legend_n_cols=1):

    mca._check_is_fitted()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Add style
    ax = prince.plot.stylize_axis(ax)
    # Plot column principal coordinates
    if show_column_points or show_column_labels:

        col_coords = mca.column_coordinates(X)
        x = col_coords[x_component]
        y = col_coords[y_component]

        prefixes = col_coords.index.str.split('_').map(lambda x: x[0])

        for prefix in prefixes.unique():
            mask = prefixes == prefix

            if show_column_points:
                ax.scatter(x[mask], y[mask],
                           s=column_points_size, label=prefix)

            if show_column_labels:
                for i, label in enumerate(col_coords[mask].index):
                    ax.annotate(label[-1], (x[mask][i], y[mask][i]))

        ax.legend(ncol=legend_n_cols)

    ax.set_title('Row and column principal coordinates')
    ei = mca.explained_inertia_
    ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(
        x_component, 100 * ei[x_component]))
    ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(
        y_component, 100 * ei[y_component]))

    return ax


def Mca(df,path):
    features = ['bodycurvature','yolkedema','necrosis','tailbending','notochorddefects','craniofacialedema',
                'finabsence','scoliosis','snoutjawdefects']
    x = df.loc[:,features].values
    dataframe = pd.DataFrame(data = x, columns = features)
    dataframe = dataframe.dropna()
    dataframe = dataframe.replace([False],'F')
    dataframe = dataframe.replace([True],'T')
    dataframe = dataframe.astype(str)
    mca = prince.MCA()
    mca.fit(dataframe)
    ax = plot_coordinates(mca, dataframe, column_points_size=100)
    ax.get_figure().savefig(path, bbox_inches='tight')


def plots(df, path):
    path = Path(path)
    df = df.dropna()
    areas = ['area_eyes', 'area_heart', 'area_out_dor', 'area_out_lat',
             'area_ov', 'area_yolk']
    feno_names = ['bodycurvature', 'yolkedema', 'necrosis', 'tailbending', 'notochorddefects', 'craniofacialedema',
                  'finabsence', 'scoliosis', 'snoutjawdefects']
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    for f in feno_names:
        i = 0
        j = 0
        for a in areas:
            if j == 3:
                j = 0
                i = 1
            scatter = ax[i, j].scatter(df['dose'], df[a], c=df[f], alpha=1)
            ax[i, j].set_xlabel("Applied doses")
            ax[i, j].set_ylabel(a)
            ax[i, j].set_title(f + ' & ' + a[5:])
            if i == 1 and j == 2:
                legend1 = ax[i, j].legend(*scatter.legend_elements(),
                                          loc="lower right", title=f)
            j = j + 1
        fig.tight_layout(pad=0.5)
        plt.savefig(path / (f + 'scatter.png'))
