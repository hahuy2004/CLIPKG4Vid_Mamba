"""
Query Selection Algorithms
Phase 2: Select k most diverse (FQS), similar (NQS), or random queries from enriched set

Usage FQS:
    python fqs_selector.py --k 2 --input_path datasets/MSRVTT/MSRVTT_JSFUSION_test_enriched.csv

Usage NQS:
    python fqs_selector.py --k 2 --nqs --input_path datasets/MSRVTT/MSRVTT_JSFUSION_test_enriched.csv

Usage Random:
    python fqs_selector.py --k 2 --random --seed 42 --input_path datasets/MSRVTT/MSRVTT_JSFUSION_test_enriched.csv
"""

import numpy as np
import torch
import pandas as pd
import argparse
import os
import random
from tqdm import tqdm
from collections import defaultdict


def load_clip_model():
    """Load CLIP model for text encoding."""
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
    """Encode texts into embeddings using CLIP."""
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
    """Compute pairwise distance matrix between query embeddings."""
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.cpu().numpy()
    
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    normalized = query_embeddings / (norms + 1e-8)
    
    similarity = np.dot(normalized, normalized.T)
    distance = 1 - similarity
    
    return distance, similarity


def farthest_query_selection(query_embeddings, k=2, threshold=None, return_indices=True):
    """Farthest Query Selection (FQS): Maximizes the minimum distance."""
    if isinstance(query_embeddings, torch.Tensor):
        embeddings_np = query_embeddings.cpu().numpy()
    else:
        embeddings_np = query_embeddings.copy()

    n_queries = embeddings_np.shape[0]
    if n_queries < k + 1:
        raise ValueError(f"Need at least {k+1} queries, but got {n_queries}")

    distance_matrix, similarity_matrix = compute_distance_matrix(embeddings_np)
    selected_indices = [0]
    remaining_indices = list(range(1, n_queries))

    for _ in range(k):
        valid_candidates = remaining_indices
        if threshold is not None:
            valid_candidates = [idx for idx in remaining_indices if similarity_matrix[0, idx] >= threshold]

        if not valid_candidates:
            selected_indices.append(0)
            continue

        max_min_distance = -1.0
        farthest_idx = -1

        for idx in valid_candidates:
            distances_to_selected = [distance_matrix[idx, s_idx] for s_idx in selected_indices]
            min_distance = min(distances_to_selected)

            if min_distance > max_min_distance:
                max_min_distance = min_distance
                farthest_idx = idx

        selected_indices.append(farthest_idx)
        remaining_indices.remove(farthest_idx)

    return selected_indices if return_indices else embeddings_np[selected_indices]


def nearest_query_sampling(query_embeddings, k=2, threshold=None, return_indices=True):
    """Nearest Query Sampling (NQS): Minimizes the minimum distance."""
    if isinstance(query_embeddings, torch.Tensor):
        embeddings_np = query_embeddings.cpu().numpy()
    else:
        embeddings_np = query_embeddings.copy()

    n_queries = embeddings_np.shape[0]
    if n_queries < k + 1:
        raise ValueError(f"Need at least {k+1} queries, but got {n_queries}")

    distance_matrix, similarity_matrix = compute_distance_matrix(embeddings_np)
    selected_indices = [0]
    remaining_indices = list(range(1, n_queries))

    for _ in range(k):
        valid_candidates = remaining_indices
        if threshold is not None:
            valid_candidates = [idx for idx in remaining_indices if similarity_matrix[0, idx] >= threshold]

        if not valid_candidates:
            selected_indices.append(0)
            continue

        min_min_distance = float('inf')
        nearest_idx = -1

        for idx in valid_candidates:
            distances_to_selected = [distance_matrix[idx, s_idx] for s_idx in selected_indices]
            min_distance = min(distances_to_selected)

            if min_distance < min_min_distance:
                min_min_distance = min_distance
                nearest_idx = idx

        selected_indices.append(nearest_idx)
        remaining_indices.remove(nearest_idx)

    return selected_indices if return_indices else embeddings_np[selected_indices]


def load_enriched_csv(csv_path):
    """Load enriched CSV file."""
    print(f"\n📂 Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = ['key', 'vid_key', 'video_id', 'sentence']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    print(f"✅ Loaded {len(df)} rows")
    return df


def group_by_video(df):
    """Group captions by video."""
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


def apply_selection_per_video(video_data, k, model=None, device=None, threshold=None, is_nqs=False, is_random=False):
    """Apply selected algorithm (Random, FQS, or NQS) to select queries for a single video."""
    original = None
    enriched = []
    
    for item in video_data:
        key = item['key']
        if '_' not in key:
            original = item
        else:
            enriched.append(item)
    
    if original is None:
        raise ValueError(f"No original caption found for video {video_data[0]['video_id']}")
    
    if len(enriched) < k:
        print(f"⚠️  Warning: Video {original['video_id']} has only {len(enriched)} enriched captions, need {k}")
        return [original] + enriched
    
    # Random Logic - No CLIP needed
    if is_random:
        selected_enriched = random.sample(enriched, k)
        return [original] + selected_enriched
    
    # FQS/NQS Logic - Requires CLIP embeddings
    all_captions = [original] + enriched
    all_texts = [item['sentence'] for item in all_captions]
    embeddings = encode_texts(all_texts, model, device)
    
    if is_nqs:
        selected_indices = nearest_query_sampling(embeddings, k=k, threshold=threshold, return_indices=True)
    else:
        selected_indices = farthest_query_selection(embeddings, k=k, threshold=threshold, return_indices=True)
    
    selected_captions = [all_captions[idx] for idx in selected_indices]
    return selected_captions


def main():
    parser = argparse.ArgumentParser(
        description="Apply Query Selection (FQS, NQS, or Random) to enriched captions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input_path", type=str, default=None, help="Path to input enriched CSV file")
    parser.add_argument("--k", type=int, default=2, help="Number of enriched captions to select per video")
    parser.add_argument("--output_path", type=str, default=None, help="Path to output CSV file")
    parser.add_argument("--threshold", "-s", type=float, default=None, help="Minimum similarity threshold 's'")
    
    # Algorithm flags
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--nqs", action="store_true", help="Use Nearest Query Sampling (NQS)")
    group.add_argument("--random", action="store_true", help="Use Random Query Sampling")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed if using random sampling
    if args.random:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    if args.input_path is None:
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
    
    if args.output_path is None:
        input_dir = os.path.dirname(args.input_path)
        if args.random:
            algo_prefix = "random"
        elif args.nqs:
            algo_prefix = "nqs"
        else:
            algo_prefix = "fqs"
        args.output_path = os.path.join(input_dir, f"MSRVTT_JSFUSION_test_{algo_prefix}_k_{args.k}.csv")
    
    if args.random:
        algo_name = f"RANDOM QUERY SAMPLING (Seed={args.seed})"
    elif args.nqs:
        algo_name = "NEAREST QUERY SAMPLING (NQS)"
    else:
        algo_name = "FARTHEST QUERY SELECTION (FQS)"
    
    print("=" * 70)
    print(algo_name)
    print("=" * 70)
    print(f"Input:  {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"k:      {args.k} (total {args.k + 1} queries per video)")
    if not args.random:
        print(f"Threshold (s): {args.threshold if args.threshold is not None else 'None'}")
    print("=" * 70)
    
    # Only load CLIP if we actually need it (FQS or NQS)
    model, device = None, None
    if not args.random:
        print("\n🔧 Loading CLIP model...")
        model, device = load_clip_model()
    else:
        print("\n⚡ Skipping CLIP loading (Random mode is fast!)...")
    
    df = load_enriched_csv(args.input_path)
    print("\n📊 Grouping captions by video...")
    video_groups = group_by_video(df)
    
    print(f"\n🎯 Applying Algorithm (k={args.k})...")
    selected_results = []
    
    for video_id in tqdm(sorted(video_groups.keys()), desc="Processing videos"):
        selected = apply_selection_per_video(
            video_groups[video_id], args.k, model, device, 
            threshold=args.threshold, is_nqs=args.nqs, is_random=args.random
        )
        selected_results.extend(selected)
    
    print(f"\n💾 Saving results...")
    output_df = pd.DataFrame(selected_results)[['key', 'vid_key', 'video_id', 'sentence']]
    
    def sort_key(key_str):
        parts = key_str.replace('ret', '').split('_')
        main_num = int(parts[0])
        sub_num = int(parts[1]) if len(parts) > 1 else -1 
        return (main_num, sub_num)
    
    output_df['sort_key'] = output_df['key'].apply(sort_key)
    output_df = output_df.sort_values('sort_key').drop('sort_key', axis=1)
    output_df.to_csv(args.output_path, index=False)
    
    print("\n" + "=" * 70)
    algo_short = "RANDOM" if args.random else ("NQS" if args.nqs else "FQS")
    print(f"✨ {algo_short} COMPLETED!")
    print("=" * 70)
    print(f"📊 Processed: {len(video_groups)} videos -> Selected: {len(selected_results)} captions")
    print(f"📄 Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()