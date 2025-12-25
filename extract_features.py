#!/usr/bin/env python3
import argparse

import torch

from model import brat, VisionEncoderOnly

SHAPE = (32, 256, 256)

def parse_args():
    parser = argparse.ArgumentParser(description="Run brat and extract query tokens.")
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to checkpoint state_dict.",
    )
    parser.add_argument(
        "--vision-model-name",
        required=True,
        choices=[
            "densenet121",
            "densenet169",
            "resnet50",
            "vit",
        ],
        help="Vision backbone name used to build the model.",
    )
    parser.add_argument(
        "--in-channels",
        required=True,
        type=int,
        help="Number of input channels for the vision model.",
    )
    parser.add_argument(
        "--mode",
        choices=["vision", "full"],
        default="full",
        help="vision: load only vision encoder weights; full: load full BRAT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    state_dict = torch.load(args.weights, map_location="cpu")

    if args.mode == "vision":
        model = VisionEncoderOnly.from_state_dict(
            state_dict,
            vision_model_name=args.vision_model_name,
            in_channels=args.in_channels,
        )
    else:
        model = brat.from_state_dict(
            state_dict,
            vision_model_name=args.vision_model_name,
            in_channels=args.in_channels,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    pixel_values = torch.randn(
        1, args.in_channels, SHAPE[0], SHAPE[1], SHAPE[2], device=device
    )

    with torch.no_grad():
        if args.mode == "vision":
            features = model.get_visual_features(pixel_values)
            print(f"visual features shape: {tuple(features.shape)}")
            print(f"dtype: {features.dtype}, device: {features.device}")
        else:
            query_tokens = model.get_query_tokens(pixel_values)
            print(f"query_tokens shape: {tuple(query_tokens.shape)}")
            print(f"dtype: {query_tokens.dtype}, device: {query_tokens.device}")


if __name__ == "__main__":
    main()
