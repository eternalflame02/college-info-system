import numpy as np
import json
import os
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.decomposition import PCA

def main():
    print("Loading data...")
    cache_path = os.path.join("data", "embeddings", "embedding_cache.npz")
    chunks_path = os.path.join("data", "chunks", "chunks.json")
    
    if not os.path.exists(cache_path) or not os.path.exists(chunks_path):
        print("Data files not found.")
        return

    # Load embeddings
    data = np.load(cache_path)
    embeddings = data['embeddings']
    chunk_ids_arr = data['chunk_ids']
    
    # Load chunks
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    print(f"Loaded {len(embeddings)} embeddings and {len(chunks)} chunks.")
    
    # Create a mapping for quick lookup
    chunk_dict = {c['chunk_id']: c for c in chunks}
    
    # Ensure they match order
    ordered_chunks = []
    for cid in chunk_ids_arr:
        # np arrays sometimes store strings strangely, ensure string type
        cid_str = str(cid)
        if cid_str in chunk_dict:
            ordered_chunks.append(chunk_dict[cid_str])
        else:
            ordered_chunks.append({"content_type": "unknown", "text": "unknown", "source_file": "unknown"})
            
    # Prepare metadata for hover text
    content_types = [c.get('content_type', 'unknown') for c in ordered_chunks]
    source_files = [os.path.basename(c.get('source_file', 'unknown')) for c in ordered_chunks]
    
    # Shorten text for preview
    texts = []
    for c in ordered_chunks:
        text = c.get('text', '')
        preview = text[:150].replace('\n', ' ') + ('...' if len(text) > 150 else '')
        texts.append(preview)
        
    print("Performing PCA reduction...")
    # Optional: reduce to 50 dims first with PCA to speed up t-SNE
    pca = PCA(n_components=min(50, embeddings.shape[1], embeddings.shape[0]))
    pca_result = pca.fit_transform(embeddings)

    print("Performing t-SNE dimensionality reduction (this may take a minute)...")
    # Perplexity must be less than n_samples
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(pca_result)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'type': content_types,
        'source': source_files,
        'text': texts
    })
    
    print("Generating interactive plot...")
    fig = px.scatter(
        df, x='x', y='y', 
        color='type', 
        hover_data=['source', 'text'],
        title='MBCET CSE Knowledge Base Embeddings - t-SNE Visualization',
        color_discrete_sequence=px.colors.qualitative.Bold,
        template='plotly_dark'
    )
    
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
        legend=dict(title='Content Type', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Save the interactive plot as HTML
    out_dir = os.path.join("data", "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "embedding_visualization.html")
    
    fig.write_html(out_file)
    print(f"✅ Visualization saved to {os.path.abspath(out_file)}")

if __name__ == "__main__":
    main()
