import json


def make_augments_same_as_original(input_path, output_path, num_augments=6):
    """
    Read input JSON and rewrite augment sentences to be identical to original.
    
    Args:
        input_path (str): path to input json file
        output_path (str): path to output json file
        num_augments (int): number of augment sentences to generate
    """

    # Load input JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each video
    for video_id, caps in data.items():
        for cap_id, cap_content in caps.items():
            original_text = cap_content.get("original", "")

            # Replace augment list with identical original sentences
            cap_content["augment"] = [original_text] * num_augments

    # Save output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Done! Saved output to: {output_path}")


if __name__ == "__main__":
    input_file = r"0_CLIPKG4Vid_MUSE\datasets\MSRVTT\msrvtt_data\MSRVTT_query_augmentation_k_6_captions.json"
    output_file = r"0_CLIPKG4Vid_MUSE\datasets\MSRVTT\msrvtt_data\MSRVTT_query_augmentation_k_6_captions_same.json"

    make_augments_same_as_original(input_file, output_file, num_augments=6)