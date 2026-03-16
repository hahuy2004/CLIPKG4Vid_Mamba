"""
Enriched Data Loader
Load enriched text data (1 original + N variations) for evaluation

NOTE: CURRENTLY UNUSED - Designed for future batch processing optimization.
Currently, eval_epoch_enriched() in main_task_retrieval.py loads JSON files directly.
This module is kept for backward compatibility and potential future use.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset


class EnrichedTextDataset(Dataset):
    """
    Dataset wrapper that loads enriched text data.
    Each sample contains 11 text variations (1 original + 10 enriched).
    """
    
    def __init__(self, base_dataset, enriched_json_path, tokenizer, max_words=32):
        """
        Args:
            base_dataset: Original dataset (e.g., MSRVTTDataset)
            enriched_json_path: Path to JSON file with enriched queries
            tokenizer: Text tokenizer
            max_words: Maximum number of words per text
        """
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_words = max_words
        
        # Load enriched data
        with open(enriched_json_path, 'r', encoding='utf-8') as f:
            self.enriched_data = json.load(f)
        
        print(f"Loaded enriched data from {enriched_json_path}")
        print(f"Enriched data contains {len(self.enriched_data)} videos")
        
    def __len__(self):
        return len(self.base_dataset)
    
    def _tokenize_text(self, text):
        """Tokenize a single text string."""
        words = self.tokenizer.tokenize(text)
        words = words[:self.max_words]
        
        # Convert to token IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        
        # Pad to max_words
        while len(input_ids) < self.max_words:
            input_ids.append(0)
        
        input_ids = np.array(input_ids)
        input_mask = (input_ids > 0).astype(int)
        
        return input_ids, input_mask
    
    def __getitem__(self, idx):
        """
        Returns enriched text data for a video.
        
        Returns:
            video_id: Video identifier
            video: Video features
            video_mask: Video attention mask
            enriched_input_ids: (n_variations, max_words) - tokenized text IDs
            enriched_input_mask: (n_variations, max_words) - attention masks
        """
        # Get base sample
        sample = self.base_dataset[idx]
        
        # Extract video ID (depends on dataset structure)
        if hasattr(self.base_dataset, 'data'):
            video_id = self.base_dataset.data.iloc[idx]['video_id']
        elif hasattr(self.base_dataset, 'video_dict'):
            video_id = list(self.base_dataset.video_dict.keys())[idx]
        else:
            # Fallback: use index as ID
            video_id = f"video_{idx}"
        
        # Get enriched text variations
        if video_id in self.enriched_data:
            text_variations = self.enriched_data[video_id]
        else:
            # Fallback: use original text only
            if len(sample) >= 4:
                original_text = sample[0]  # Assuming first element is text
            else:
                original_text = "a video"
            text_variations = [original_text] * 11
            print(f"Warning: No enriched data for {video_id}, using original text only")
        
        # Tokenize all variations
        enriched_input_ids = []
        enriched_input_mask = []
        
        for text in text_variations:
            input_ids, input_mask = self._tokenize_text(text)
            enriched_input_ids.append(input_ids)
            enriched_input_mask.append(input_mask)
        
        enriched_input_ids = np.stack(enriched_input_ids)  # (n_variations, max_words)
        enriched_input_mask = np.stack(enriched_input_mask)
        
        # Get video features from base sample
        # Assuming sample format: (input_ids, input_mask, segment_ids, video, video_mask)
        if len(sample) >= 5:
            video = sample[3]
            video_mask = sample[4]
        else:
            # Create dummy video features
            video = np.zeros((12, 512))
            video_mask = np.ones(12)
        
        return (
            video_id,
            torch.from_numpy(video) if isinstance(video, np.ndarray) else video,
            torch.from_numpy(video_mask) if isinstance(video_mask, np.ndarray) else video_mask,
            torch.from_numpy(enriched_input_ids),
            torch.from_numpy(enriched_input_mask)
        )


def create_enriched_dataloader(base_dataloader, enriched_json_path, tokenizer, max_words=32):
    """
    Create enriched dataloader from base dataloader.
    
    Args:
        base_dataloader: Original dataloader
        enriched_json_path: Path to enriched queries JSON
        tokenizer: Text tokenizer
        max_words: Maximum words per text
        
    Returns:
        EnrichedTextDataset wrapped with original dataloader settings
    """
    base_dataset = base_dataloader.dataset
    
    enriched_dataset = EnrichedTextDataset(
        base_dataset=base_dataset,
        enriched_json_path=enriched_json_path,
        tokenizer=tokenizer,
        max_words=max_words
    )
    
    # Create new dataloader with same settings
    from torch.utils.data import DataLoader
    
    enriched_dataloader = DataLoader(
        enriched_dataset,
        batch_size=base_dataloader.batch_size,
        shuffle=False,
        num_workers=getattr(base_dataloader, 'num_workers', 0),
        pin_memory=getattr(base_dataloader, 'pin_memory', False),
        drop_last=False
    )
    
    return enriched_dataloader


def create_enriched_dataloader_simple(args, tokenizer, enriched_json_path):
    """
    Simplified function to create enriched dataloader for evaluation.
    This is a helper that can be called directly in main_task_retrieval.py
    
    Args:
        args: Arguments object with dataset info
        tokenizer: Text tokenizer
        enriched_json_path: Path to enriched queries JSON
        
    Returns:
        Enriched dataloader
    """
    from dataloaders.data_dataloaders import DATALOADER_DICT
    
    # Get base test dataloader
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        base_dataloader, _ = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)
    elif DATALOADER_DICT[args.datatype]["val"] is not None:
        base_dataloader, _ = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        raise ValueError("No test or val dataloader available")
    
    return create_enriched_dataloader(
        base_dataloader=base_dataloader,
        enriched_json_path=enriched_json_path,
        tokenizer=tokenizer,
        max_words=args.max_words
    )


if __name__ == "__main__":
    # Test enriched dataloader
    print("Testing Enriched DataLoader...")
    
    # Create dummy enriched data
    dummy_enriched = {
        "video_0": ["original caption"] + [f"variation {i}" for i in range(1, 11)],
        "video_1": ["another caption"] + [f"variant {i}" for i in range(1, 11)]
    }
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dummy_enriched, f)
        temp_path = f.name
    
    print(f"Created dummy enriched data at {temp_path}")
    print("Test completed! Use with actual dataset for full testing.")
