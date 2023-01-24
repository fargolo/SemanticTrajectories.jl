using Embeddings
using SemanticTrajectories

#using WordTokenizers 

# Load embedding table
const ft_embtable = load_embeddings(FastText_Text{:en}; max_vocab_size=30000) 

colors_txt = "Colors are the central theme of this text .
Blue , green and yellow are the colors of Brazil .
Red and white are from Japan and Canada .
Does violet exist in any flags ?
Jamaica is yellow, black and green . Orange is another color ."

random_txt = "This text looks like a Haiku .
Frog can be pink, rocket is bigger than hat .
Every night there's ice in some bakery .
Even the Jew can run and fish ,
but socialism remains seated"

# Semantic distances
rnd_dists = semantic_space_dists(random_txt,ft_embtable)
colors_dists = semantic_space_dists(colors_txt,ft_embtable)

# RQA metrics
colors_props = rqa_metrics(colors_dists["skipmiss_raw"])
rnd_props = rqa_metrics(rnd_dists["skipmiss_raw"])

# Windowed RQA metrics
windowed_rqa_metrics(colors_dists["skipmiss_raw"],10,1)
windowed_rqa_metrics(rnd_dists["skipmiss_raw"],10,1)

