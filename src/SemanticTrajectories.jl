module SemanticTrajectories

using WordTokenizers 
using Statistics , Distances , DataFrames
using DynamicalSystems , TSne

export 
    get_embedding, semantic_space_dists , 
    rqa_metrics , windowed_rqa_metrics

include("semantic_dists.jl")
include("RQA.jl")

end # module SemanticTrajectories
