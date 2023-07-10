
"""
    semantic_space_dists_phrase(raw_text, embtable)

Obtain distance matrices using an Embeddings.jl embedding table and summing vectors in each sentence. 

Inputs are the raw text and an embedding table.
Return Dict with "distances" (DataFrame including original sentences),"distances_raw" (real valued matrix).  
"""

# Previously, 'foldl(+)' could not handle phrases
# made of 1 word. Added 'if' statement that pushed
# zeros(300) to the matrix. This results in NaN entries
# when calculating pairwise distances.
## Fix 1: Remove single phrases from sentence
## Fix 2: Remove NaNs from matrix 

## Update: added kwarg init=zeros(300) to foldl

function semantic_space_dists_phrase(raw_text,embtable)
    
    tokenized_sentences = [x for x in WordTokenizers.split_sentences(raw_text)]
    sentences_emb = []
    for sentence in tokenized_sentences
        labels = WordTokenizers.punctuation_space_tokenize(lowercase(sentence))        
        if length(labels) < 2
            continue
        end 
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


