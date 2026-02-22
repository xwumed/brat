
import argparse
import os
import glob
import torch
import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    ScaleIntensityd,
    Spacingd,
)
from model import brat

# Define the expected modalities in order for Multimodal model
# Order: T1c, T1, T2, FLAIR
MODALITIES = ['t1c', 't1w', 't2w', 't2f']
SHAPE = (32, 256, 256)

def get_transforms():
    """
    MONAI preprocessing pipeline matching the training configuration.
    """
    return Compose([
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"], channel_dim="no_channel"),
        Spacingd(keys=["img"], pixdim=[1, 1, 1], mode="bilinear"),
        Orientationd(keys=["img"], axcodes="SAR"),
        Resized(keys=["img"], spatial_size=SHAPE),
        NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=True),
        ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
    ])

def process_patient(patient_dir, model, device, transforms):
    patient_id = os.path.basename(patient_dir)
    print(f"Processing {patient_id}...")

    # Identify files for each modality
    files = {}
    for f in os.listdir(patient_dir):
        if not f.endswith('.nii.gz') and not f.endswith('.nii'):
            continue
        lower_f = f.lower()
        if 't1c' in lower_f:
            files['t1c'] = f
        elif 't1w' in lower_f or 't1.' in lower_f: # simple check for t1
            files['t1w'] = f
        elif 't2w' in lower_f or 't2.' in lower_f:
            files['t2w'] = f
        elif 't2f' in lower_f or 'flair' in lower_f:
            files['t2f'] = f
        
    stacked_list = []
    
    for mod in MODALITIES:
        if mod in files:
            img_path = os.path.join(patient_dir, files[mod])
            print(f"  Found {mod}: {files[mod]}")
            data = {"img": img_path}
            try:
                out = transforms(data)
                tensor = out["img"] # shape (1, 32, 256, 256)
                stacked_list.append(tensor)
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                stacked_list.append(torch.zeros(1, *SHAPE))
        else:
            print(f"  Missing {mod}, zero-filling.")
            stacked_list.append(torch.zeros(1, *SHAPE))
            
    # Stack along channel dimension: 4 x (1, D, H, W) -> (4, D, H, W)
    # Actually we want (4, D, H, W) where 4 is channel. 
    # Current list items are (1, D, H, W).
    # cat along dim 0 gives (4, D, H, W).
    input_tensor = torch.cat(stacked_list, dim=0).unsqueeze(0) # (1, 4, D, H, W)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        # Get query tokens (features)
        features = model.get_query_tokens(input_tensor)
        
    return features.cpu()

def main():
    parser = argparse.ArgumentParser(description="Run BRAT inference on a folder of NIfTI files.")
    parser.add_argument("--weights", default="brat_4m_densenet169.bin", help="Path to model weights")
    parser.add_argument("--data_dir", required=True, help="Directory containing patient subdirectories")
    parser.add_argument("--output_dir", default=None, help="Output directory for features (default: data_dir + '_brat')")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.data_dir.rstrip('/') + '_brat'
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    print(f"Loading model from {args.weights}...")
    state_dict = torch.load(args.weights, map_location="cpu")
    
    # Initialize Multimodal model (DenseNet169, 4 channels)
    model = brat.from_state_dict(
        state_dict,
        vision_model_name="densenet169",
        in_channels=4,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Model loaded on {device}.")
    
    transforms = get_transforms()
    
    patient_dirs = sorted(glob.glob(os.path.join(args.data_dir, "*")))
    count = 0
    for p_dir in patient_dirs:
        if not os.path.isdir(p_dir):
            continue
        
        try:
            feats = process_patient(p_dir, model, device, transforms)
            save_path = os.path.join(args.output_dir, os.path.basename(p_dir) + ".pt")
            torch.save(feats, save_path)
            print(f"Saved features to {save_path} with shape {tuple(feats.shape)}")
            count += 1
        except Exception as e:
            print(f"Failed to process {p_dir}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Finished processing {count} patients.")

if __name__ == "__main__":
    main()
