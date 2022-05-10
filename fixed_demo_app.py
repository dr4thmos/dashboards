import streamlit as st
import json
import requests
import os

import numpy as np
import pandas as pd

import pacmap
import umap
import hdbscan
from sklearn.cluster import KMeans, DBSCAN

from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Category20, Category10
from bokeh.themes import built_in_themes
from bokeh.io import output_notebook, curdoc
from bokeh.resources import settings


import pathlib 
import shutil

EXPERIMENT_FOLDER = "20220503-094002-wide-ld64-15k"
EMBEDDINGS_FILE = "embeddings.json"
LABELS_FILE = "labels.json"
IMAGES_FOLDER = "images"
GENERATED_FOLDER = "generated"

# HACK This only works when we've installed streamlit with pipenv, so the
# permissions during install are the same as the running process

STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
print(STREAMLIT_STATIC_PATH)
# We create a videos directory within the streamlit static asset directory
# and we write output files to it
STATIC_IMAGES_PATH = (os.path.join(STREAMLIT_STATIC_PATH, IMAGES_FOLDER))
if not os.path.isdir(STATIC_IMAGES_PATH):
    os.mkdir(STATIC_IMAGES_PATH)

for image in os.listdir(os.path.join(EXPERIMENT_FOLDER, IMAGES_FOLDER)):
    shutil.copy(os.path.join(EXPERIMENT_FOLDER, IMAGES_FOLDER, image), STATIC_IMAGES_PATH)  # For newer Python.

STATIC_GENERATED_PATH = (os.path.join(STREAMLIT_STATIC_PATH, GENERATED_FOLDER))
if not os.path.isdir(STATIC_GENERATED_PATH):
    os.mkdir(STATIC_GENERATED_PATH)

for image in os.listdir(os.path.join(EXPERIMENT_FOLDER, GENERATED_FOLDER)):
    shutil.copy(os.path.join(EXPERIMENT_FOLDER, GENERATED_FOLDER, image), STATIC_GENERATED_PATH)  # For newer Python.

st.set_page_config(layout="wide")

visualization_column, clustering_column = st.columns(2)

visualization_column.header("Visualization pipeline")
clustering_column.header("Clustering pipeline")

with visualization_column:
    
    with st.container():
        visualization_prereduction_check = st.checkbox("Execute Pre-Reduction", value=False, key="prereduction", help=None, on_change=None)
        
        if visualization_prereduction_check:
            """
            ---
            """
            st.write('Pre-Reduction')
            visualization_prereduction_method = st.selectbox(
                'Reduction type',
                ('UMAP', ''),
                key="prereduction method",
                disabled=not(visualization_prereduction_check)
            )

            #vis_pre_reduction_components = st.number_input('Output dimensions', key="vis_pre_reduction_components", min_value = 4, max_value = 32)
            vis_pre_reduction_components = st.slider('Output dimensions', key="vis_pre_reduction_components", step = 1, min_value = 4, max_value = 32, value = 5)

            with st.container():
                if visualization_prereduction_method == "UMAP":
                    #vis_pre_reduction_n_neighbors = st.number_input('Number of neighbors', key="vis_pre_reduction_n_neighbors", step = 5, min_value = 5, max_value = 100)
                    vis_pre_reduction_n_neighbors = st.slider('Number of neighbors', key="vis_pre_reduction_n_neighbors", step = 5, min_value = 5, max_value = 100, value = 15)

                elif visualization_prereduction_method == "PACMAP":
                    st.write('No parameter for PACMAP for now')
    """
    ---
    """

    with st.container():
        st.write('Reduction')

        visualization_reduction_method = st.selectbox(
            'Reduction type',
            ('UMAP', 'PACMAP'),
            key="reduction method"
        )

        vis_reduction_components = st.number_input('Output dimensions', key="vis_reduction_components", min_value = 2, max_value = 2)

        with st.container():
            if visualization_reduction_method == "UMAP":
                #vis_reduction_n_neighbors = st.number_input('Number of neighbors', key="vis_reduction_n_neighbors", step = 5, min_value = 5, max_value = 100)
                vis_reduction_UMAP_n_neighbors = st.slider('Number of neighbors', key="vis_reduction_UMAP_n_neighbors", min_value = 5, max_value = 100, step = 5, value = 15)
                #vis_reduction_min_distance = st.number_input('Minimum distance between points', key="vis_reduction_min_distance", step = 0.1, min_value = 0.0, max_value = 1.0)
                vis_reduction_UMAP_min_distance = st.slider('Minimum distance between points', key="vis_reduction_UMAP_min_distance", step = 0.05, min_value = 0.0, max_value = 1.0, value = 0.1)
                

            elif visualization_reduction_method == "PACMAP":
                st.write('No parameter for PACMAP for now')
                vis_reduction_PACMAP_n_neighbors = st.slider('Number of neighbors', key="vis_reduction_PACMAP_n_neighbors", min_value = 5, max_value = 100, step = 5, value = 15)
                vis_reduction_PACMAP_MN_ratio = st.slider('Attraction between near points', key="vis_reduction_PACMAP_MN_ratio", step = 0.1, min_value = 0.1, max_value = 2.0, value = 0.5)
                vis_reduction_PACMAP_FP_ratio = st.slider('Repulsion between distance points', key="vis_reduction_PACMAP_FP_ratio", step = 0.5, min_value = 0.5, max_value = 5.0, value = 2.0)
                


with clustering_column:
    
    with st.container():
        clustering_prereduction_check = st.checkbox("Execute Pre-Reduction", value=False, key="clustering_prereduction", help=None, on_change=None)
        
        if clustering_prereduction_check:
            """
            ---
            """
            st.write('Clustering Pre-Reduction')
            clustering_prereduction_method = st.selectbox(
                'Reduction type',
                ('UMAP', ''),
                key="clustering_prereduction_method"
            )

            #clustering_prereduction_components = st.number_input('Output dimensions', key="clustering_prereduction_components", min_value = 4, max_value = 32)
            clustering_prereduction_components = st.slider('Output dimensions', key="clustering_prereduction_components", step = 1, value = 5, min_value = 4, max_value = 32)

            with st.container():
                if clustering_prereduction_method == "UMAP":
                    #clustering_prereduction_n_neighbors = st.number_input('Number of neighbors', key="clustering_prereduction_n_neighbors", step = 5, min_value = 5, max_value = 100)
                    clustering_prereduction_n_neighbors = st.slider('Number of neighbors', key="clustering_prereduction_n_neighbors", step = 5, min_value = 5, max_value = 100, value = 15)

                elif clustering_prereduction_method == "PACMAP":
                    st.write('No parameter for PACMAP for now')
    """
    ---
    """

    with st.container():
        st.write('Clustering')

        clustering_method = st.selectbox(
            'Clustering type',
            ('kmeans', 'dbscan', 'hdbscan'),
            key="clustering_method"
        )

        with st.container():
            if clustering_method == "kmeans":
                #K = st.number_input('Number of cluster', key="K", step = 1, min_value = 2, max_value = 20)
                K = st.slider('Number of cluster', key="K", step = 1, min_value = 2, max_value = 20, value = 5)
                clusterer = KMeans(n_clusters=K, random_state=0)

            elif clustering_method == "dbscan":
                #eps = st.number_input('Eps', key="eps", step = 0.0, min_value = 0.1, max_value = 1.0)
                eps = st.slider('Eps', key="eps", step = 0.1, min_value = 0.0, max_value = 1.0, value = 0.1)
                #min_samples = st.number_input('Min samples', key="min_samples", step = 1, min_value = 2, max_value = 50)
                min_samples = st.slider('Min samples', key="min_samples", step = 1, min_value = 2, max_value = 50, value = 5)
                clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                

            elif clustering_method == "hdbscan":
                #min_cluster_size = st.number_input('Min cluster size', key="min_cluster_size", step = 1, min_value = 2, max_value = 50)
                min_cluster_size = st.slider('Min cluster size', key="min_cluster_size", step = 1, min_value = 2, max_value = 50, value = 5)
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)


"""
---
"""
a = st.button("Compute", key="Compute", help="Compute all the pipeline and visualize")

@st.cache
def load_data(data_path):
    #print(data_path)
    with open(data_path, "r") as file:
        data = json.load(file)
        #print(data[0])
    return data

EXPERIMENT_FOLDER = "20220503-094002-wide-ld64-15k"
EMBEDDINGS_FILE = "embeddings.json"
LABELS_FILE = "labels.json"
IMAGES_FOLDER = "images"
GENERATED_FOLDER = "generated"

TRAIN_PATH = os.path.join(EXPERIMENT_FOLDER, EMBEDDINGS_FILE)
LABELS_PATH = os.path.join(EXPERIMENT_FOLDER, LABELS_FILE)
IMAGE_PATH = os.path.join(EXPERIMENT_FOLDER, IMAGES_FOLDER)
GEN_PATH = os.path.join(EXPERIMENT_FOLDER, GENERATED_FOLDER)

train_data = np.array(load_data(TRAIN_PATH))
labels_data = load_data(LABELS_PATH)

images = labels_data['columns']
labels = np.array(labels_data['data'])
labels = labels.flatten()

df_image_paths = pd.DataFrame(
    {
        'image_path' : map(
            lambda image: os.path.join(IMAGES_FOLDER,image), 
            images
            )
    })
print(df_image_paths.head())

df_images_filename = pd.DataFrame({'image': images})
df_images_filename = df_images_filename.join(df_image_paths)
print(df_images_filename.head())


df_gen_paths = pd.DataFrame(
    {
        'gen_path' : map(
            lambda image: os.path.join(GENERATED_FOLDER,image), 
            images
            )
    })
print(df_gen_paths.head())


viz_data = train_data
clus_data = train_data



if a:
    # Visualization
    if visualization_prereduction_check:
        st.write('Visualization Pre-Reduction: ', visualization_prereduction_method)
        reducer = umap.UMAP(n_neighbors=vis_pre_reduction_n_neighbors, min_dist=0, n_components=vis_pre_reduction_components)
        viz_data = reducer.fit_transform(viz_data)
    
    
    st.write('Reduction: ', visualization_reduction_method)
    if visualization_reduction_method == "UMAP":
        reducer = umap.UMAP(n_neighbors=vis_reduction_UMAP_n_neighbors, min_dist=vis_reduction_UMAP_min_distance, n_components=vis_reduction_components)
    elif visualization_reduction_method == "PACMAP":
        reducer = pacmap.PaCMAP(n_components=vis_reduction_components, n_neighbors=vis_reduction_PACMAP_n_neighbors, MN_ratio=vis_reduction_PACMAP_MN_ratio, FP_ratio=vis_reduction_PACMAP_FP_ratio)
        #umap.PACMAP(n_neighbors=vis_reduction_n_neighbors, min_dist=vis_reduction_min_distance, n_components=vis_reduction_components)
    embedding = reducer.fit_transform(viz_data)

    df_embedding = pd.DataFrame(embedding)
    
    if vis_reduction_components == 2:
        df_embedding = df_embedding.rename(columns={0:"x", 1:"y"})
    if vis_reduction_components == 3:
        df_embedding = df_embedding.rename(columns={0:"x", 1:"y", 2:"z"})

    # Clustering
    if clustering_prereduction_check:
        
        st.write('Clustering Pre-Reduction: ', clustering_prereduction_method)
        reducer = umap.UMAP(n_neighbors=clustering_prereduction_n_neighbors, min_dist=0, n_components=clustering_prereduction_components)
        clus_data = reducer.fit_transform(clus_data)
    
    st.write('Clustering:' , clustering_method)
    
    clusters = clusterer.fit_predict(clus_data)
    df_clusters = pd.DataFrame(clusters)
    df_clusters = df_clusters.rename(columns={0:"clusters"})

    df_embedding = df_embedding.join(df_images_filename)
    df_embedding = df_embedding.join(df_clusters)
    df_embedding = df_embedding.join(df_gen_paths)
    #df_embedding = df_embedding.join(df_image_paths)

    output_file('plot.html')
    curdoc().theme = 'dark_minimal'

    if np.unique(clusters).size < 10:
        color = [Category10[10][i+1] for i in df_embedding['clusters']]
    else:
        color = [Category20[20][i+1] for i in df_embedding['clusters']]

    datasource =  ColumnDataSource(data=dict(index=df_embedding.index,
                                            x=df_embedding.x,
                                            y=df_embedding.y,
                                            image=df_embedding.image,
                                            clusters=df_embedding.clusters,
                                            image_path=df_embedding.image_path,
                                            gen_path=df_embedding.gen_path,
                                            color=color))


    plot_figure = figure(plot_width=800, plot_height=800, tools=('pan, wheel_zoom, reset, save'))
    #color_mapping = CategoricalColorMapper(factors=[(x) for x in 'clusters'], palette=Category20[3])

    plot_figure.add_tools(HoverTool(tooltips="""
    <div style='text-align:center; border: 2px solid; border-radius: 2px'>
    <div style='display:flex'> 
        <div>
            <img src='@image_path' width="192" style='display: block; margin: 2px auto auto auto;'/>
        </div>
        <div>
            <img src='@gen_path' width="192" style='display: block; margin: 2px auto auto auto;'/>
        </div>
        </div>
        <div style='padding: 2px; font-size: 12px; color: #000000'>
            <span>Cluster:</span>
            <span>@clusters</span><br>
            <span>X:</span>
            <span>@x</span><br>
            <span>Y:</span>
            <span>@y</span><br>
            <span>Image:</span>
            <span>@image</span>
        </div>
    </div>
    """))

    plot_figure.circle('x', 'y', source=datasource, color='color', legend_field='clusters', fill_alpha=0.5, size=12)
    plot_figure.legend.title = "Clusters"
    plot_figure.legend.label_text_color = "black"
    plot_figure.legend.background_fill_color = 'white'
    plot_figure.legend.background_fill_alpha = 0.5

    #show(plot_figure)

    st.bokeh_chart(plot_figure, use_container_width=True)


    from PIL import Image
    
    for cluster, col in zip(np.unique(clusters), st.columns(np.unique(clusters).size)):
        with col:
            st.title('#' + str(cluster))
            for _, row in df_embedding.iterrows():
                if row.clusters == cluster:
                    image = Image.open(os.path.join(EXPERIMENT_FOLDER, row.image_path))
                    st.image(image, caption=row.image)