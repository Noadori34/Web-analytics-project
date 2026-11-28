# Tableau use

- Use centrality_results.csv for centrality rankings and skill demand (one row per skill).
- Use communities.csv to relate skill IDs to community membership if needed.
- Use community_summary.csv for group summaries. Jobs can belong to several communities.
- Add figures/skill_network.png as an image; label it Full-sample skill network.
- network_edges.csv contains one row per skill pair; do not join it directly to job rows and then sum job counts.
- Metrics are precomputed for the full sample. Filtering displayed skills does not recompute PageRank, betweenness or communities.
- Use corrected Phase 3 salary outputs for salary charts; these graph files do not contain salary estimates.
- Preserve C++ and C# as separate keys when relating to the salary results.
