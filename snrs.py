import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
import umap

st.title('Snub Nose Remenants analyzer')

EXPERIMENT_FOLDER = "20220503-094002-wide-ld64-15k"
EMBEDDINGS_FILE = "embeddings.json"
LABELS_FILE = "labels.json"
IMAGES_FOLDER = "images"
GENERATED_FOLDER = "generated"

@st.cache
def load_data(data_path):
    data = json.load(json_data)
    print(data[0])
    return np.array(data)

train_data = load_data(os.path.join(EXPERIMENT_FOLDER, EMBEDDINGS_FILE))

reduction_method = umap.UMAP(n_neighbors=10, min_dist=0, n_components=5)
train_data = reduction_method.fit_transform(train_data)
print(train_data.shape)




@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)

x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

p = figure(
     title='simple line example',
     x_axis_label='x',
     y_axis_label='y')

p.line(x, y, legend_label='Trend', line_width=2)

st.bokeh_chart(p, use_container_width=True)