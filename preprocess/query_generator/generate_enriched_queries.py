"""
Standalone script for generating enriched queries using GPT-5-mini
Run this script BEFORE evaluation to prepare enriched data

Usage for MSRVTT:
    python generate_enriched_queries.py \
        --datatype msrvtt \
        --data_path data/MSRVTT/MSRVTT_JSFUSION_test.csv \
        --output_csv data/MSRVTT/MSRVTT_JSFUSION_test_enriched.csv \
        --output_reference data/MSRVTT/MSRVTT_eval_enriched_reference_data.json \
        --output_json data/MSRVTT/MSRVTT_standard.json \
        --api_key "your-openai-api-key" \
        --n_variations 10

Usage for MSVD:
    python generate_enriched_queries.py \
        --datatype msvd \
        --data_path data/MSVD/test_list.txt \
        --raw_captions data/MSVD/raw-captions.pkl \
        --output_pkl data/MSVD/eval_enriched-caption-complete.pkl \
        --output_reference data/MSVD/enriched_eval_captions.json \
        --output_json data/MSVD/MSVD_standard.json \
        --api_key "your-openai-api-key" \
        --n_variations 10

Usage for DiDeMo:
    python generate_enriched_queries.py \
        --datatype didemo \
        --data_path data/DiDeMo/test_captions.json \
        --output_json data/DiDeMo/DiDeMo_standard.json \
        --api_key "your-openai-api-key" \
        --n_variations 10
"""

import sys
import os
import pickle

# Add parent directory to path to import enriched_eval module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_generator import generate_enriched_queries
import argparse
import json
import pandas as pd

# Default API key (can be overridden via command line)
DEFAULT_API_KEY = "YOUR_OPENAI_API_KEY"

def load_msrvtt_csv(csv_path):
    print(f"Loading MSRVTT CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    data = {}
    for _, row in df.iterrows():
        key = row['key']
        data[key] = {
            'vid_key': row['vid_key'],
            'video_id': row['video_id'],
            'sentence': row['sentence']
        }
    
    print(f"Loaded {len(data)} MSRVTT queries")
    return data

def load_msvd_data(test_list_path, raw_captions_path):
    print(f"Loading MSVD test list from: {test_list_path}")
    with open(test_list_path, 'r') as f:
        test_videos = [line.strip() for line in f if line.strip()]
    
    print(f"Loading MSVD captions from: {raw_captions_path}")
    with open(raw_captions_path, 'rb') as f:
        all_captions = pickle.load(f)
    
    test_data = {}
    for video_id in test_videos:
        if video_id in all_captions:
            original_captions = all_captions[video_id]
            first_caption_text = ' '.join(original_captions[0])
            test_data[video_id] = {
                'original_captions': original_captions,
                'first_caption_text': first_caption_text
            }
    
    print(f"Loaded {len(test_data)} MSVD test videos")
    return test_data

def save_msrvtt_enriched_csv(enriched_data, original_data, output_csv_path, n_variations=10):
    print(f"Saving enriched MSRVTT CSV to: {output_csv_path}")
    rows = []
    for key in sorted(original_data.keys()):
        vid_key = original_data[key]['vid_key']
        video_id = original_data[key]['video_id']
        
        if key in enriched_data:
            captions = enriched_data[key]
        else:
            captions = [original_data[key]['sentence']] * (n_variations + 1)
        
        rows.append({
            'key': key,
            'vid_key': vid_key,
            'video_id': video_id,
            'sentence': captions[0]
        })
        
        for j in range(1, n_variations + 1):
            enriched_key = f"{key}_{j}"
            rows.append({
                'key': enriched_key,
                'vid_key': vid_key,
                'video_id': video_id,
                'sentence': captions[j] if j < len(captions) else captions[0]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(rows)} rows to {output_csv_path}")

def save_msvd_enriched_pkl(enriched_data, original_captions_dict, output_pkl_path):
    print(f"Saving enriched MSVD pickle to: {output_pkl_path}")
    output_data = {}
    metadata = {}
    
    for video_id, enriched_captions in enriched_data.items():
        all_captions_tokenized = []
        video_metadata = []
        
        if video_id in original_captions_dict:
            original_caps = original_captions_dict[video_id]
            num_originals = len(original_caps)
            
            for idx, original_cap_tokens in enumerate(original_caps):
                all_captions_tokenized.append(original_cap_tokens)
                group_start = len(all_captions_tokenized) - 1
                
                if idx == 0:
                    for enriched_text in enriched_captions[1:11]:
                        tokens = enriched_text.lower().split()
                        all_captions_tokenized.append(tokens)
                else:
                    for _ in range(10):
                        all_captions_tokenized.append(original_cap_tokens)
                
                group_end = len(all_captions_tokenized)
                video_metadata.append({
                    'original_index': idx,
                    'group_range': (group_start, group_end),
                    'original_text': ' '.join(original_cap_tokens),
                    'enriched_count': 10
                })
        else:
            print(f"Warning: {video_id} not in original_captions_dict")
            continue
        
        output_data[video_id] = all_captions_tokenized
        metadata[video_id] = {
            'num_original_captions': num_originals,
            'total_captions': len(all_captions_tokenized),
            'groups': video_metadata
        }
    
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    total_captions = sum(len(caps) for caps in output_data.values())
    print(f"Saved {len(output_data)} videos with {total_captions} total captions to {output_pkl_path}")
    return metadata

def save_reference_json(enriched_data, output_json_path):
    print(f"Saving reference JSON to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    print(f"Saved reference data to {output_json_path}")

# --- THÊM HÀM LƯU CHUẨN HOÁ MỚI CHO OUTPUT JSON ---
def save_standard_json(enriched_data, datatype, original_data, output_json_path):
    """
    Format output:
    {
      "video_id_1": {
        "cap_1": {
          "original": "origin_cap_1",
          "augment": ["aug_1", "aug_2", ..., "aug_n"]
        }
      }
    }
    """
    print(f"Saving standardized JSON to: {output_json_path}")
    unified_data = {}
    
    for key, captions in enriched_data.items():
        if not captions:
            continue
            
        original_cap = captions[0]
        aug_caps = captions[1:]
        
        if datatype == "msrvtt" and original_data is not None:
            # Đối với MSRVTT, key là 'ret0', 'ret1'..., cần lấy lại video_id
            video_id = original_data[key]['video_id']
            cap_id = key # Lưu ID caption (ví dụ: ret0, ret1) thay vì cap_1 để giữ unique
        else:
            # Đối với MSVD và DiDeMo, key chính là video_id, 
            # mỗi video đang được lấy caption đầu tiên
            video_id = key
            cap_id = "cap_1"
            
        if video_id not in unified_data:
            unified_data[video_id] = {}
            
        unified_data[video_id][cap_id] = {
            "original": original_cap,
            "augment": aug_caps
        }
        
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(unified_data)} videos in standardized format to {output_json_path}")
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate enriched queries using GPT",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Bổ sung lựa chọn didemo
    parser.add_argument("--datatype", type=str, required=True,
                       choices=["msrvtt", "msvd", "didemo"],
                       help="Dataset type: msrvtt, msvd or didemo")
    
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to input data file (CSV, test_list.txt, or JSON for didemo)")
    parser.add_argument("--raw_captions", type=str, default=None,
                       help="Path to raw-captions.pkl (MSVD only)")
    
    parser.add_argument("--output_csv", type=str, default=None,
                       help="Output CSV path (MSRVTT only)")
    parser.add_argument("--output_pkl", type=str, default=None,
                       help="Output pickle path (MSVD only)")
    parser.add_argument("--output_reference", type=str, default=None,
                       help="Output reference JSON path (for debugging)")
    
    # --- THÊM THAM SỐ OUTPUT CHUẨN HOÁ CHO CẢ 3 DATASET ---
    parser.add_argument("--output_json", type=str, default=None,
                       help="Output standardized JSON path (Applicable for all datasets)")
    # ------------------------------------------------------

    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY, help="OpenAI API key")
    parser.add_argument("--n_variations", type=int, default=10, help="Number of variations per caption (default: 10)")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="OpenAI model to use (default: gpt-5-mini)")
    parser.add_argument("--sleep_time", type=float, default=1.0, help="Sleep time between API calls")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.datatype == "msrvtt" and args.output_csv is None:
        parser.error("--output_csv is required for MSRVTT")
    elif args.datatype == "msvd":
        if args.raw_captions is None:
            parser.error("--raw_captions is required for MSVD")
        if args.output_pkl is None:
            parser.error("--output_pkl is required for MSVD")
    elif args.datatype == "didemo":
        if args.output_json is None and args.output_reference is None:
            print("Warning: Both --output_json and --output_reference are empty. You should specify at least one.")
    
    print("="*70)
    print("ENRICHED QUERY GENERATION")
    print("="*70)
    print(f"Dataset: {args.datatype.upper()}")
    print(f"Model: {args.model}")
    print(f"Variations per caption: {args.n_variations}")
    print("="*70)
    
    input_captions = {}
    original_data = None
    original_captions_dict = {}

    if args.datatype == "msrvtt":
        print("\n[MSRVTT] Loading data...")
        original_data = load_msrvtt_csv(args.data_path)
        for key, data in original_data.items():
            input_captions[key] = data['sentence']
        print(f"Extracted {len(input_captions)} captions to enrich")
        
    elif args.datatype == "msvd":
        print("\n[MSVD] Loading data...")
        test_data = load_msvd_data(args.data_path, args.raw_captions)
        for video_id, video_data in test_data.items():
            input_captions[video_id] = video_data['first_caption_text']
            original_captions_dict[video_id] = video_data['original_captions']
        print(f"Extracted {len(input_captions)} videos")
        
    # --- LOGIC ĐỌC DỮ LIỆU ĐẦU VÀO CỦA DIDEMO ---
    elif args.datatype == "didemo":
        print("\n[DiDeMo] Loading data...")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            input_captions = json.load(f)
        print(f"Loaded {len(input_captions)} queries from DiDeMo JSON")
    # --------------------------------------------
    
    if args.max_samples:
        print(f"\nLimiting to first {args.max_samples} samples for testing")
        input_captions = dict(list(input_captions.items())[:args.max_samples])
    
    print(f"\nTotal captions to enrich: {len(input_captions)}")
    
    # Tạo đường dẫn lưu Reference mặc định nếu người dùng không truyền vào nhưng logic cần
    if not args.output_reference:
        output_dir = os.path.dirname(args.output_json) if args.output_json else os.path.dirname(args.data_path)
        args.output_reference = os.path.join(output_dir, f"{args.datatype}_reference_data.json")

    print("\n" + "="*70)
    print("GENERATING ENRICHED QUERIES...")
    print("="*70)
    
    enriched_data = generate_enriched_queries(
        input_captions=input_captions,
        output_json_path=args.output_reference, 
        api_key=args.api_key,
        n_variations=args.n_variations,
        model=args.model,
        sleep_time=args.sleep_time
    )
    
    print("\nQuery generation completed!")
    print(f"Generated {len(enriched_data)} enriched video captions")
    
    print("\n" + "="*70)
    print("SAVING OUTPUT FILES...")
    print("="*70)
    
    # 1. Lưu các Output Đặc trưng Dataset / Reference 
    if args.datatype == "msrvtt":
        save_msrvtt_enriched_csv(enriched_data, original_data, args.output_csv, args.n_variations)
        
        video_id_enriched_data = {}
        for key, captions in enriched_data.items():
            video_id = original_data[key]['video_id']
            if video_id not in video_id_enriched_data:
                video_id_enriched_data[video_id] = []
            video_id_enriched_data[video_id].extend(captions)
        save_reference_json(video_id_enriched_data, args.output_reference)
        
    elif args.datatype == "msvd":
        save_msvd_enriched_pkl(enriched_data, original_captions_dict, args.output_pkl)
        save_reference_json(enriched_data, args.output_reference)
        
    elif args.datatype == "didemo":
        if args.output_reference:
            save_reference_json(enriched_data, args.output_reference)

    # 2. LƯU ĐỊNH DẠNG JSON CHUẨN HOÁ CHO TẤT CẢ DATASETS
    if args.output_json:
        save_standard_json(enriched_data, args.datatype, original_data, args.output_json)

    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()