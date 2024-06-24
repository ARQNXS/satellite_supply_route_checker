# Sentinel Data Fetcher and Image Analysis Tool

This Python script fetches satellite images from the Sentinel Hub API, compares them using a pre-trained TensorFlow model, and analyzes the resulting difference image. It includes classes for fetching Sentinel data and analyzing images.

## Requirements

To run this script, you will need the following Python packages:

- `numpy`
- `pillow`
- `datetime`
- `matplotlib`
- `sentinelhub`
- `scikit-image`
- `tensorflow`
- `tensorflow_hub`

You can install these packages using pip:

```bash
pip install numpy pillow datetime matplotlib sentinelhub scikit-image tensorflow tensorflow_hub
