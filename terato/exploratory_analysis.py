

import os
import csv
import cv2
import pandas as pd
from PIL import Image
import terato.display as Tox
import torch
from torchvision import transforms
from pathlib import Path

import matplotlib.pyplot as plt
import altair as alt

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import numpy as np
from bioinfokit.visuz import cluster
import prince

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def pca(df):
    plt.clf()
    df = df.dropna()
    features = ['dose','area_eyes', 'area_heart', 'area_out_dor', 'area_out_lat',
               'area_ov', 'area_yolk', 'length_eyes', 'length_heart', 'length_out_dor',
               'length_out_lat', 'length_ov', 'length_yolk']
    x = df.loc[:,features].values
    data = StandardScaler().fit_transform(x)
    dataframe = pd.DataFrame(data = x, columns = features)
    pca = PCA()
    pca_scores = pca.fit_transform(data)
    loadings = pca.components_
    compounds = df['compound'].to_numpy()
    return cluster.biplot(cscore=pca_scores, loadings=loadings, labels=dataframe.columns.values, var1=round(pca.explained_variance_ratio_[0]*100, 2),
                    var2=round(pca.explained_variance_ratio_[1]*100, 2), colorlist=compounds)



def plot_coordinates(mca, X, ax=None, figsize=(10,8), x_component=0, y_component=1, show_column_points=True, column_points_size=100, show_column_labels=True, legend_n_cols=1):


    #mca._check_is_fitted()

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
                ax.scatter(x[mask], y[mask], s=column_points_size, label=prefix)

            if show_column_labels:
                for i, label in enumerate(col_coords[mask].index):
                    ax.annotate(label[-3], (x[mask][i], y[mask][i]))

        ax.legend(ncol=legend_n_cols,fontsize=9)


    ei = mca.explained_inertia_
    ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]), fontsize=12, fontname='Arial')
    ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]), fontsize=12 , fontname='Arial')

    return ax


def Mca(df, path):
    df = df.dropna()
    features =['bodycurvature','yolkedema','necrosis','tailbending',
               'notochorddefects','craniofacialedema',
                'finabsence','scoliosis','snoutjawdefects']
    x = df.loc[:,features].values
    dataframe = pd.DataFrame(data = x, columns = features)
    mca = prince.MCA()
    mask = dataframe.applymap(type) != bool
    d = {True: '1', False: '0'}
    dataframe = dataframe.where(mask, dataframe.replace(d))
    print(type(dataframe.iloc[3,3]))
    print(dataframe)
    mca = mca.fit(dataframe)
    ax = plot_coordinates(mca, dataframe)
    ax.get_figure().savefig(path, bbox_inches='tight')

#Save a dose vs death probability chart for each compound
def compoundvsdeathvsdose(df):
    alt.data_transformers.disable_max_rows()
    compounds = df['compound'].drop_duplicates()
    compounds = compounds.to_numpy()

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
        dosevsdeath.save(c + '.html')


#Save a dose vs phenotype for each compound
def doseperresponse(df, path):
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
           #color = alt.Color("feno:N", scale = alt.Scale(domain= feno_names))
      ).transform_filter(selector)


    compounds = np.delete(compounds, 0)
    for c in compounds:
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
        chart = (doseresponse + baseline).properties(height=700,width=500)
        chart.save(str(Path(path) / (c + "feno" + ".html")))
