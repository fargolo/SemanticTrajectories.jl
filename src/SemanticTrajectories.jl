module SemanticTrajectories

using WordTokenizers 
using Statistics , Distances , DataFrames , NaNStatistics
using DynamicalSystems , TSne

export 
    get_embedding, semantic_space_trajectory,
    semantic_space_dists , semantic_space_dists_phrase ,
    rqa_metrics , windowed_rqa_metrics

include("semantic_dists.jl")
include("RQA.jl")

end # module SemanticTrajectories
