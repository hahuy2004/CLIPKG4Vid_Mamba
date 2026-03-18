"""
Farthest Query Selection (FQS) Algorithm
Phase 2: Select k most diverse queries from enriched set

Usage:
    python fqs_selector.py --k 2 --input_path datasets/MSRVTT/MSRVTT_JSFUSION_test_enriched.csv
"""

import numpy as np
import torch
import pandas as pd
import argparse
import os
from tqdm import tqdm
from collections import defaultdict


def load_clip_model():
    """
    Load CLIP model for text encoding.
    
    Returns:
        model, preprocess: CLIP model and preprocessing function
    """
    try:
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"✅ Loaded CLIP ViT-B/32 on {device}")
        return model, device
    except ImportError:
        print("❌ CLIP not found. Installing...")
        os.system("pip install git+https://github.com/openai/CLIP.git")
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, device


def encode_texts(texts, model, device, batch_size=32):
    """
    Encode texts into embeddings using CLIP.
    
    Args:
        texts: List of text strings
        model: CLIP model
        device: Device to use
        batch_size: Batch size for encoding
        
    Returns:
        np.ndarray: Embeddings of shape (n_texts, embed_dim)
    """
    import clip
    
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            tokens = clip.tokenize(batch_texts, truncate=True).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            embeddings.append(text_features.cpu().numpy())
    
    return np.vstack(embeddings)


def compute_distance_matrix(query_embeddings):
    """
    Compute pairwise distance matrix between query embeddings.
    
    Args:
        query_embeddings: torch.Tensor or np.ndarray of shape (n_queries, embed_dim)
        
    Returns:
        np.ndarray: Distance matrix of shape (n_queries, n_queries)
    """
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.cpu().numpy()
    
    # Normalize embeddings (already normalized from CLIP, but ensure)
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    normalized = query_embeddings / (norms + 1e-8)
    
    # Compute cosine similarity
    similarity = np.dot(normalized, normalized.T)
    
    # Convert to distance (1 - cosine similarity)
    distance = 1 - similarity
    
    return distance


def farthest_query_selection(query_embeddings, k=2, threshold=None, return_indices=True):
    """
    Farthest Query Selection (FQS) algorithm with optional similarity threshold.

    Args:
        query_embeddings: (n_queries, embed_dim), first embedding is original query
        k: Number of enriched queries to select (total selected = k+1)
        threshold: Minimum cosine similarity with original query (index 0)
        return_indices: If True, return indices; else return embeddings

    Returns:
        If return_indices=True: list of selected indices [0, idx1, idx2, ...]
        If return_indices=False: selected embeddings array
    """
    if isinstance(query_embeddings, torch.Tensor):
        embeddings_np = query_embeddings.cpu().numpy()
    else:
        embeddings_np = query_embeddings.copy()

    n_queries = embeddings_np.shape[0]
    if n_queries < k + 1:
        raise ValueError(f"Need at least {k+1} queries, but got {n_queries}")

    # Compute distance matrix and derive cosine similarity matrix.
    distance_matrix = compute_distance_matrix(embeddings_np)
    similarity_matrix = 1.0 - distance_matrix

    # Initialize with original query (index 0)
    selected_indices = [0]
    remaining_indices = list(range(1, n_queries))

    # Iteratively select k more queries
    for _ in range(k):
        max_min_distance = -1
        farthest_idx = -1

        # Keep only candidates that pass similarity threshold with original query.
        valid_candidates = remaining_indices
        if threshold is not None:
            valid_candidates = [
                idx for idx in remaining_indices if similarity_matrix[0, idx] >= threshold
            ]

        # If no candidate passes threshold, re-use original query (index 0).
        if not valid_candidates:
            selected_indices.append(0)
            continue

        # Farthest point sampling on valid candidates only.
        for idx in valid_candidates:
            distances_to_selected = [distance_matrix[idx, s_idx] for s_idx in selected_indices]
            min_distance = min(distances_to_selected)

            if min_distance > max_min_distance:
                max_min_distance = min_distance
                farthest_idx = idx

        selected_indices.append(farthest_idx)
        remaining_indices.remove(farthest_idx)

    if return_indices:
        return selected_indices
    else:
        return embeddings_np[selected_indices]


def batch_farthest_query_selection(batch_query_embeddings, k=2):
    """
    Apply FQS to a batch of query sets.
    
    Args:
        batch_query_embeddings: torch.Tensor of shape (batch_size, n_queries, embed_dim)
        k: Number of enriched queries to select per sample
        
    Returns:
        list of lists: [[selected_indices for sample 1], [selected_indices for sample 2], ...]
    """
    batch_size = batch_query_embeddings.shape[0]
    batch_selected = []
    
    for i in range(batch_size):
        selected = farthest_query_selection(batch_query_embeddings[i], k=k, return_indices=True)
        batch_selected.append(selected)
    
    return batch_selected


def select_queries_by_indices(query_embeddings, selected_indices):
    """
    Extract selected query embeddings by indices.
    
    Args:
        query_embeddings: torch.Tensor of shape (n_queries, embed_dim) or (batch, n_queries, embed_dim)
        selected_indices: list of indices or list of lists for batch
        
    Returns:
        Selected embeddings
    """
    if isinstance(selected_indices[0], list):
        # Batch mode
        selected = []
        for i, indices in enumerate(selected_indices):
            selected.append(query_embeddings[i, indices])
        return torch.stack(selected)
    else:
        # Single sample
        return query_embeddings[selected_indices]


def load_enriched_csv(csv_path):
    """
    Load enriched CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        pd.DataFrame: DataFrame with columns [key, vid_key, video_id, sentence]
    """
    print(f"\n📂 Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = ['key', 'vid_key', 'video_id', 'sentence']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"✅ Loaded {len(df)} rows")
    return df


def group_by_video(df):
    """
    Group captions by video.
    
    Args:
        df: DataFrame with enriched captions
        
    Returns:
        dict: {video_id: list of (key, vid_key, video_id, sentence)}
    """
    grouped = defaultdict(list)
    
    for _, row in df.iterrows():
        video_id = row['video_id']
        grouped[video_id].append({
            'key': row['key'],
            'vid_key': row['vid_key'],
            'video_id': row['video_id'],
            'sentence': row['sentence']
        })
    
    return dict(grouped)


def apply_fqs_per_video(video_data, k, model, device, threshold=None):
    """
    Apply FQS algorithm to select k+1 queries for a single video.
    
    Args:
        video_data: List of caption dicts for one video
        k: Number of enriched captions to select (total = k+1)
        model: CLIP model
        device: Device
        
    Returns:
        list: Selected caption dicts (k+1 items)
    """
    # Separate original and enriched
    original = None
    enriched = []
    
    for item in video_data:
        key = item['key']
        # Original caption has key without underscore (e.g., ret0)
        if '_' not in key:
            original = item
        else:
            enriched.append(item)
    
    if original is None:
        raise ValueError(f"No original caption found for video {video_data[0]['video_id']}")
    
    # If not enough enriched captions, return what we have
    if len(enriched) < k:
        print(f"⚠️  Warning: Video {original['video_id']} has only {len(enriched)} enriched captions, need {k}")
        return [original] + enriched
    
    # Encode all captions
    all_captions = [original] + enriched
    all_texts = [item['sentence'] for item in all_captions]
    embeddings = encode_texts(all_texts, model, device)
    
    # Apply FQS
    selected_indices = farthest_query_selection(
        embeddings, k=k, threshold=threshold, return_indices=True
    )
    
    # Return selected captions
    selected_captions = [all_captions[idx] for idx in selected_indices]
    return selected_captions


def main():
    parser = argparse.ArgumentParser(
        description="Apply Farthest Query Selection (FQS) to enriched captions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input_path", type=str, default=None,
                       help="Path to input enriched CSV file")
    parser.add_argument("--k", type=int, default=2,
                       help="Number of enriched captions to select per video (default: 2)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to output FQS CSV file")
    parser.add_argument("--threshold", "-s", type=float, default=None,
                       help="Minimum similarity threshold 's' (e.g., 0.5). Default is None (Vanilla FQS)")
    
    args = parser.parse_args()
    
    # Auto-detect input file if not specified
    if args.input_path is None:
        # Try to find enriched CSV
        possible_paths = [
            "datasets/MSRVTT/MSRVTT_JSFUSION_test_enriched_2_captions.csv",
            "datasets/MSRVTT/MSRVTT_JSFUSION_test_enriched.csv",
            "MSRVTT_JSFUSION_test_enriched_2_captions.csv",
            "MSRVTT_JSFUSION_test_enriched.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                args.input_path = path
                break
        
        if args.input_path is None:
            raise FileNotFoundError("No enriched CSV file found. Please specify --input_path")
    
    # Auto-generate output path if not specified
    if args.output_path is None:
        input_dir = os.path.dirname(args.input_path)
        args.output_path = os.path.join(input_dir, f"MSRVTT_JSFUSION_test_fqs_k_{args.k}.csv")
    
    print("=" * 70)
    print("FARTHEST QUERY SELECTION (FQS)")
    print("=" * 70)
    print(f"Input:  {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"k:      {args.k} (total {args.k + 1} queries per video)")
    threshold_str = args.threshold if args.threshold is not None else "None (Vanilla FQS)"
    print(f"Threshold (s): {threshold_str}")
    print("=" * 70)
    
    # Load CLIP model
    print("\n🔧 Loading CLIP model...")
    model, device = load_clip_model()
    
    # Load CSV
    df = load_enriched_csv(args.input_path)
    
    # Group by video
    print("\n📊 Grouping captions by video...")
    video_groups = group_by_video(df)
    print(f"✅ Found {len(video_groups)} unique videos")
    
    # Apply FQS to each video
    print(f"\n🎯 Applying FQS (k={args.k})...")
    selected_results = []
    
    for video_id in tqdm(sorted(video_groups.keys()), desc="Processing videos"):
        video_data = video_groups[video_id]
        selected = apply_fqs_per_video(
            video_data, args.k, model, device, threshold=args.threshold
        )
        selected_results.extend(selected)
    
    # Create output DataFrame
    print(f"\n💾 Saving results...")
    output_df = pd.DataFrame(selected_results)
    output_df = output_df[['key', 'vid_key', 'video_id', 'sentence']]
    
    # Sort by key: ret0, ret0_1, ret0_2, ret1, ret1_1, ...
    def sort_key(key_str):
        """Extract numeric parts for sorting: ret0_1 -> (0, 1), ret10 -> (10, -1)"""
        parts = key_str.replace('ret', '').split('_')
        main_num = int(parts[0])
        sub_num = int(parts[1]) if len(parts) > 1 else -1  # -1 for original (no underscore)
        return (main_num, sub_num)
    
    output_df['sort_key'] = output_df['key'].apply(sort_key)
    output_df = output_df.sort_values('sort_key')
    output_df = output_df.drop('sort_key', axis=1)
    
    output_df.to_csv(args.output_path, index=False)
    
    # Statistics
    print("\n" + "=" * 70)
    print("✨ FQS COMPLETED!")
    print("=" * 70)
    print(f"📊 Total videos processed:  {len(video_groups)}")
    print(f"📊 Total captions selected: {len(selected_results)}")
    print(f"📊 Expected per video:      {args.k + 1} (1 original + {args.k} enriched)")
    print(f"📊 Average per video:       {len(selected_results) / len(video_groups):.1f}")
    print(f"\n📄 Output saved to: {args.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
