from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Category20
from bokeh.themes import built_in_themes
from bokeh.io import output_notebook, curdoc
from bokeh.resources import settings

import pandas as pd
import numpy as np
import umap
import json
import os
import glob
import pacmap
import hdbscan
from sklearn.cluster import KMeans, DBSCAN

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
IMAGE_PATH = os.path.join(EXPERIMENT_FOLDER, IMAGES_FOLDER, '*')
GEN_PATH = os.path.join(EXPERIMENT_FOLDER, GENERATED_FOLDER, '*')

train_data = np.array(load_data(TRAIN_PATH))

labels_data = load_data(LABELS_PATH)

images = labels_data['columns']
labels = np.array(labels_data['data'])
labels = labels.flatten()

imagepath = []
for file in sorted(glob.iglob(IMAGE_PATH, recursive=True)):
    imagepath.append(file)
df_image_paths = pd.DataFrame({'image_path' : imagepath[:]})
df_image_paths.head()

df_images_filename = pd.DataFrame({'image': images[:]})
df_images_filename = df_images_filename.join(df_image_paths)
print(df_images_filename.head())

genpath = []
for file in sorted(glob.iglob(GEN_PATH, recursive=True)):
    genpath.append(file)
df_image_paths = pd.DataFrame({'gen_path' : genpath[:]})
print(df_image_paths.head())

reducer = umap.UMAP(n_neighbors=10, min_dist=0, n_components=5)

reducted_space = reducer.fit_transform(train_data)

n_clusters = 5
random_state = 0
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
clusters = kmeans.fit_predict(reducted_space)
df_clusters = pd.DataFrame(clusters)
df_clusters = df_clusters.rename(columns={0:"clusters"})

n_components = 2
n_neighbors = 5
MN_ratio = 0.5
FP_ratio = 2

reducer = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio)
embedding = reducer.fit_transform(reducted_space)
embeddingx = embedding[:,0]
embeddingy = embedding[:,1]

df_embedding = pd.DataFrame(embedding)
df_embedding = df_embedding.rename(columns={0:"x", 1:"y"})

df_embedding = df_embedding.join(df_images_filename)
df_embedding = df_embedding.join(df_clusters)

df_embedding.head()

output_file('plot.html')
curdoc().theme = 'dark_minimal'
datasource =  ColumnDataSource(data=dict(index=df_embedding.index,
                                         x=df_embedding.x,
                                         y=df_embedding.y,
                                         image=df_embedding.image,
                                         clusters=df_embedding.clusters,
                                         image_path=imagepath,
                                         gen_path=genpath,
                                         color=[Category20[20][i+1] for i in df_embedding['clusters']]))

plot_figure = figure(plot_width=800, plot_height=800, tools=('pan, wheel_zoom, reset, save'))
color_mapping = CategoricalColorMapper(factors=[(x) for x in 'clusters'], palette=Category20[3])

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

plot_figure.circle('x', 'y', source=datasource, color='color', legend_field='clusters', fill_alpha=0.5, size=6)
plot_figure.legend.title = "Clusters"
plot_figure.legend.label_text_color = "black"
plot_figure.legend.background_fill_color = 'white'
plot_figure.legend.background_fill_alpha = 0.5

show(plot_figure)