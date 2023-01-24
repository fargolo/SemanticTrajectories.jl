"""
    rqa_metrics(dist_matrix)

Return RQA measures from a distance matrix.

Return RQA measures from a t-SNE based 2D trajectory. 
Threshold for recurrence is the average Euclidian distance.

"""

function rqa_metrics(dist_matrix)
    tsne_output = TSne.tsne(dist_matrix,distance=true)
    tsne_dists = Distances.pairwise(Distances.euclidean,tsne_output')
    mu_dists = Statistics.mean(tsne_dists)
    rec_mat = DynamicalSystems.RecurrenceMatrix(tsne_output,mu_dists)
    rqa_res = DynamicalSystems.rqa(Dict,rec_mat)

    return rqa_res

end


"""
    windowed_rqa_metrics(dist_matrix)

Return RQA measures from a distance matrix.

Return RQA measures from a t-SNE based 2D trajectory. 
Threshold for recurrence is the average Euclidian distance.

"""

function windowed_rqa_metrics(dist_matrix,w_width=10,w_step=1)
    
    tsne_output = TSne.tsne(dist_matrix,distance=true)
    tsne_dists = Distances.pairwise(Distances.euclidean,tsne_output')
    mu_dists = Statistics.mean(tsne_dists)
    rec_mat = DynamicalSystems.RecurrenceMatrix(tsne_output,mu_dists; metric = Euclidean())
    rqa_res = DynamicalSystems.rqa(Dict,rec_mat)
    window_props = @windowed rqa(rec_mat) width=w_width step=w_step
    label_keys = keys(window_props)
    window_summ = Dict(zip(label_keys,map(mean,values(window_props))))

    return window_summ

end