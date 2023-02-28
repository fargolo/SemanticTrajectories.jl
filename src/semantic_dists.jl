"""
    get_embeddings(word,embtable)

Get real vector embedding of a word in an Embeddings.jl embtable. 
"""
# Get embeddings form corpus in embtable
function get_embedding(word,embtable)
    
    word_inds = Dict(word => ii for (ii,word) in enumerate(embtable.vocab))
    try
        ind = word_inds[word]
        emb = embtable.embeddings[:,ind]
        return emb
    catch y
        if isa(y, KeyError)
            return missing
        end
    end
end


"""
    semantic_space_trajectory(raw_text, embtable;skipmiss=true)

Obtain trajectory (sequence of embeddings) using an Embeddings.jl embedding table. 

Inputs are the raw text and an embedding table.
It returns a Dict with trajectory (Vector{Float32}) and labels (Vector{String}).  
"""

function semantic_space_trajectory(raw_text,embtable;skipmiss=true)

    labels = WordTokenizers.punctuation_space_tokenize(lowercase(raw_text)) 

    latent_space_reps = map(x -> get_embedding(x,embtable),labels)
    embedded_traj = ifelse(skipmiss==false,
        [latent_space_reps,labels], # trajectory and labels
        [collect(skipmissing(latent_space_reps)), labels[.!(ismissing.(vec(latent_space_reps)))]]) # skip missing embeddings
    
  
    return Dict("trajectory" => embedded_traj[1],"labels" => map(String,embedded_traj[2]))

end


"""
    semantic_space_dists(raw_text, embtable)

Obtain distance matrices using an Embeddings.jl embedding table. 

Inputs are the raw text and an embedding table.
Return Dict with "skipmiss_distances","distances".  
"""

function semantic_space_dists(raw_text,embtable)

    labels = WordTokenizers.punctuation_space_tokenize(lowercase(raw_text)) 

    latent_space_reps = map(x -> get_embedding(x,embtable),labels)
    latent_space_reps_skipmiss = collect(skipmissing(latent_space_reps)) 
    labels_skipmiss = labels[.!(ismissing.(vec(latent_space_reps)))]

    # Add 0s to missing values befode cosine dist.
    latent_space_repsZ = map(latent_space_reps) do vec
        if ismissing(vec)
            return zeros(300)
        end
        return vec
    end

    vector_reps = permutedims(hcat(latent_space_repsZ...))
    vector_reps_skipmiss = permutedims(hcat(latent_space_reps_skipmiss...))        
    word_dists = Distances.pairwise(Distances.cosine_dist, vector_reps, dims=1)
    word_dists_skipmiss = Distances.pairwise(Distances.cosine_dist, vector_reps_skipmiss, dims=1)
   
    word_dists_df = DataFrame(hcat(labels,word_dists),:auto)
    word_dists_skipmiss_df = DataFrame(hcat(labels_skipmiss,word_dists_skipmiss),:auto)
    rename!(word_dists_skipmiss_df, ["HEADER",labels_skipmiss...], makeunique=true)
    rename!(word_dists_df, ["HEADER",labels...], makeunique=true)

    return Dict("skipmiss_distances" => word_dists_skipmiss_df,
    "skipmiss_raw" => Matrix{Float64}(word_dists_skipmiss_df[:,2:end]), 
    "distances" => word_dists_df,
    "distances_raw" => Matrix{Float64}(word_dists_df[:,2:end]))

end



"""
    semantic_space_dists_phrase(raw_text, embtable)

Obtain distance matrices using an Embeddings.jl embedding table and summing vectors in each sentence. 

Inputs are the raw text and an embedding table.
Return Dict with "distances" (DataFrame including original sentences),"distances_raw" (real valued matrix).  
"""

function semantic_space_dists_phrase(raw_text,embtable)
    
    tokenized_sentences = [x for x in WordTokenizers.split_sentences(raw_text)]
    sentences_emb = []
    for sentence in tokenized_sentences
        labels = WordTokenizers.punctuation_space_tokenize(lowercase(sentence))
        latent_space_reps = map(x -> get_embedding(x,embtable),labels)
        latent_space_reps_skipmiss = collect(skipmissing(latent_space_reps)) 
        labels_skipmiss = labels[.!(ismissing.(vec(latent_space_reps)))]
        
        sentences_sum = foldl(+,latent_space_reps_skipmiss)
        push!(sentences_emb,sentences_sum)

    end        

    vector_reps = permutedims(hcat(sentences_emb...))
    word_dists = Distances.pairwise(Distances.cosine_dist, vector_reps, dims=1)
    
    word_dists_df = DataFrame(hcat(tokenized_sentences,word_dists),:auto)
    rename!(word_dists_df, ["HEADER",tokenized_sentences...], makeunique=true)

    return Dict("distances" => word_dists_df,
    "distances_raw" => Matrix{Float64}(word_dists_df[:,2:end]))

end


