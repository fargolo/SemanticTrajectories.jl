using Distances
using Statistics 
using DynamicalSystems

sm  = semantic_space_trajectory(colors_txt,ft_embtable;skipmiss=true)



# Change cosine (semi-metric does not respect triangle inequality) for another 
## 
cosine_dists = Distances.pairwise(Distances.cosine_dist,sm["trajectory"])
mu_dists = Statistics.mean(cosine_dists)

rec_mat = DynamicalSystems.RecurrenceMatrix(sm["trajectory"],mu_dists;metric=Distances.Euclidean())
rqa_res = DynamicalSystems.rqa(Dict,rec_mat)



function rqa_cosine(raw_text,embtable)

    vector_reps_skipmiss = permutedims(hcat(latent_space_reps_skipmiss...))        
    word_dists = Distances.pairwise(Distances.cosine_dist, vector_reps, dims=1)
    word_dists_skipmiss = Distances.pairwise(Distances.cosine_dist, vector_reps_skipmiss, dims=1)
   
    word_dists_df = DataFrame(hcat(labels,word_dists),:auto)
    word_dists_skipmiss_df = DataFrame(hcat(labels_skipmiss,word_dists_skipmiss),:auto)
    rename!(word_dists_skipmiss_df, ["HEADER",labels_skipmiss...], makeunique=true)
    rename!(word_dists_df, ["HEADER",labels...], makeunique=true)

end
