# SemanticTrajectories

[![Build Status](https://github.com/fargolo/SemanticTrajectories.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/fargolo/SemanticTrajectories.jl/actions/workflows/CI.yml?query=branch%3Amain)  


See: https://osf.io/preprints/psyarxiv/epgfy/

`SemanticTrajectories.jl` is package built upon `DynamicalSystems.jl` and `Embeddings.jl`. It is designed to analyze semantic coherence in text. A popular approach in literature is to calculate coherence among consecutive words ("first order coherence") or among two words having a third word in-between ("second order coherence"). Then, one evaluates properties such as average, minimum and maximum coherence.  

Semantic `SemanticTrajectories.jl` leverages `DynamicalSystems.jl` Recurrence Quantification Analysis (RQA) capabilities to analyze semantic coherence in the entire trajectory.  

# RQA

For the purpose of performing RQA, we first build a recurrence matrix that contains 0-valued entries if two states are far apart and 1-valued entries when states are close enough according to a threshold. Each $X_i$,j value corresponds to the result for states i and j. For instance, $x_{1,3}$ is 0-valued if the 1st and 3rd words have a large distance and it is 1-valued if it is smaller than a certain threshold.  

The cosine distance is widely used to evaluate distances in semantic embeddings. Unfortunately, it is not a metric in the strict sense. That is, it does not satisfy properties such as the triangle inequality (e.g. the sum of any two sides of a triangle is greater than or equal to the third side).  To circumvent that, obtain 2D trajectories from the original cosine based distance matrices using t-Distributed Stochastic Neighbor Embedding (t-SNE).  

The cosine distance matrix is transformed into a 2D trajectory through t-SNE and RQA is applied to this trajectory. The recurrence matrix considers the Euclidean distance as the metric and the average value as a threshold.   

---  

A total of 14 RQA metrics are divided in 4 groups: 1 of them quantifies all similar pairs (RR); 3 features are associated with the number of words between two similar words (MRT, RTE and NMPRT); 4 features are related to vertical lines and characterize sequences of words similar to the first one in the sequence (LAM, TT, Vmax and VENTR); 6 features are related to diagonals and describe similarities occurring at regular spaced intervals with non-similar words in-between (L, DET, Lmax, ENTR and TREND).  

---  

The recurrence rate (RR) is simply the relative frequency of states in which the trajectory returns to positions visited before, akin to a general measure of coherence, considering all similar word pairs in the text.  

---  

Recurrence times measure how many intermediary states exist between an arbitrary state and a recurrence. In other words, how long it takes for similarities to occur. They are related to the Poincaré recurrence theorem, which proves that certain systems will eventually return to a state arbitrarily close to their initial state. The Poincaré recurrence time is the length of time elapsed until a recurrence. A total of 3 features assess this: MRT, RTE and NMPRT. The mean recurrence time (MRT) is the average interval between similar words. Recurrence time entropy (RTE) checks whether recurrence times are heterogeneous (appear with different frequencies). The number of the most probable recurrence time (NMPRT) indicates how many times recurrences exist considering the recurrence time that appears most frequently.  

---  

Vertical lines relate a given word and the next n-words. For instance, a vertical line formed by points $x_{1,2} , x_{1,3}  and x_{1,4}$ means a small semantic distance between the 1st and the next 3 consecutive words. A total of 4 features are associated with vertical lines: TT, LAM, Vmax and VENTR. Trapping time (TT) is the average size of these consecutive coherent snippets. Laminarity (LAM) is the percentage of recurrence points which form vertical lines. That is, how many coherent pairs belong to a consecutive sequence as compared to random coherent pairs. The maximum length (Vmax) indicates the longest sequence. The Shannon entropy of vertical structures (VENTR) indicates if coherent sequences appear at any size or whether some sequence sizes are more frequent than others.  

---  

Diagonals suggest regular intervals with different delays. Each diagonal contains sequences of distances between words considering different step windows. For instance, the diagonal immediately next to the identity (main diagonal, line of identity, LOI) is called superdiagonal (or 1-diagonal) and contains all FOCs, while the 2-diagonal contains all SOCs.  
There are 6 features available from diagonals. The  average length of diagonal structures (L) represents the average size of coherent snippets of any order (e.g. FOC, SOC, 3rd order coherence…). Determinism (DET) is the ratio between diagonals (count weighted by length) and total recurrence; that is, how often coherent speech snippets with regular intervals (e.g. FOC and SOC sequences) appear as compared to random coherent word pairs. The maximum diagonal length (Lmax) and its inverse, Divergence (DIV, or 1/Lmax) shows the largest snippet with coherence in regular intervals.  We can also calculate the Shannon entropy (ENTR) of the probability distribution of the diagonal line lengths. Texts that present similar diagonal lengths for all intervals (FOC, SOC, 3rd order coherence,…) will have higher entropy (akin to a uniform distribution) while imbalances in the distribution of diagonal lengths will be associated with smaller values.  The trend (TREND) is the slope of a linear regression that relates the density of recurrent points in the diagonals parallel to the LOI and the distance between those diagonals and the LOI. If trend is positive, recurrence at large delays is more frequent than for close words. If negative, recurrence is more frequent at small delays. Zeroed values suggest that similar words occur regardless of time-window.  
