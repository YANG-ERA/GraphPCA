import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scanpy as sc


def match_cluster_labels(true_labels, est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i + 1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j - 1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr == org_cat[i]) * (est_labels_arr == est_cat[j]))
            B.add_edge(i + 1, -j - 1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
    #     match = minimum_weight_full_matching(B)
    if len(org_cat) >= len(est_cat):
        return np.array([match[-est_cat.index(c) - 1] - 1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c) - 1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c) - 1) in match:
                l.append(match[-est_cat.index(c) - 1] - 1)
            else:
                l.append(len(org_cat) + unmatched.index(c))
        return np.array(l)


def refine(sample_id, pred, dis, shape="hexagon", neighbor_num=None):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    global num_nbs
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    elif shape == "generic":
        if neighbor_num is None:
            raise ValueError("The parameter cannot be empty, please enter a valid value.")
        else:
            num_nbs = neighbor_num
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0:num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred


def make_scatterplot(adata, column_name=None, color_list=None, coord_x="x", coord_y="y", use_title=True, size=200,
                     only_point=True, figsize_width=7, figsize_height=4, plot_name=None):
    if adata is None:
        raise ValueError("Please specify an adata!")
    if column_name not in adata.obs.columns:
        raise ValueError(f"Error: Column '{column_name}' does not exist in the adata.obs.")

    # Get the number of unique categories in the specified column
    category_length = adata.obs[column_name].nunique()

    if color_list is not None:

        # Check if the length of the color_list matches the category length
        if len(color_list) != category_length:
            raise ValueError("Length of color_list must be the same as the category length of the specified column.")

        adata.uns[column_name + "_colors"] = color_list

    else:
        print("We use the default color scheme by scanpy")

    if only_point:
        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height), constrained_layout=True)
        if use_title:
            sc.pl.scatter(adata, x=coord_x, y=coord_y, color=column_name, title=column_name, ax=ax, show=False,
                          size=size)
        else:
            sc.pl.scatter(adata, x=coord_x, y=coord_y, color=column_name, title="", ax=ax, show=False, size=size)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height), constrained_layout=True)
        if use_title:
            sc.pl.scatter(adata, x=coord_x, y=coord_y, color=column_name, title=column_name, ax=ax, show=False,
                          size=size)
        else:
            sc.pl.scatter(adata, x=coord_x, y=coord_y, color=column_name, title="", ax=ax, show=False, size=size)

    if plot_name is None:
        plt.savefig("scatterplot_" + column_name + ".pdf")
    else:
        plt.savefig(plot_name)
