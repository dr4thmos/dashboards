import streamlit as st

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
                ('UMAP', 'PACMAP'),
                key="prereduction method",
                disabled=not(visualization_prereduction_check)
            )

            vis_pre_reduction_components = st.number_input('Output dimensions', key="vis_pre_reduction_components", min_value = 4, max_value = 32)

            with st.container():
                if visualization_prereduction_method == "UMAP":
                    vis_pre_reduction_n_neighbors = st.number_input('Number of neighbors', key="vis_pre_reduction_n_neighbors", step = 5, min_value = 5, max_value = 100)

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

        vis_reduction_components = st.number_input('Output dimensions', key="vis_reduction_components", min_value = 4, max_value = 32)

        with st.container():
            if visualization_reduction_method == "UMAP":
                vis_reduction_n_neighbors = st.number_input('Number of neighbors', key="vis_reduction_n_neighbors", step = 5, min_value = 5, max_value = 100)

            elif visualization_reduction_method == "PACMAP":
                st.write('No parameter for PACMAP for now')


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
                ('UMAP', 'PACMAP'),
                key="clustering_prereduction_method",
                disabled=not(visualization_prereduction_check)
            )

            clustering_prereduction_components = st.number_input('Output dimensions', key="clustering_prereduction_components", min_value = 4, max_value = 32)

            with st.container():
                if clustering_prereduction_method == "UMAP":
                    clustering_prereduction_n_neighbors = st.number_input('Number of neighbors', key="clustering_prereduction_n_neighbors", step = 5, min_value = 5, max_value = 100)

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
                K = st.number_input('Number of cluster', key="K", step = 1, min_value = 2, max_value = 20)

            elif clustering_method == "dbscan":
                eps = st.number_input('Eps', key="eps", step = 0.0, min_value = 0.1, max_value = 1.0)
                min_samples = st.number_input('Min samples', key="min_samples", step = 1, min_value = 2, max_value = 50)

            elif clustering_method == "hdbscan":
                min_cluster_size = st.number_input('Min cluster size', key="min_cluster_size", step = 1, min_value = 2, max_value = 50)


"""
---
"""
a = st.button("Compute", key="Compute", help="Compute all the pipeline and visualize")

if a:    
    # Visualization
    if visualization_prereduction_check:
        st.write('Visualization Pre-Reduction: ', visualization_prereduction_method)
    
    st.write('Reduction: ', visualization_reduction_method)

    # Clustering
    if clustering_prereduction_check:
        st.write('Clustering Pre-Reduction: ', clustering_prereduction_method)
    
    st.write('Clustering:' , clustering_method)
