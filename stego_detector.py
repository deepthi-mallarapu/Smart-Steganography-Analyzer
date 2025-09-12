from PIL import Image
import numpy as np


def image_entropy(pixels):
    """Compute grayscale image entropy"""
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def analyze_image(image_path):
    """
    Analyze image for LSB steganography using scaled Chi-Square + entropy.
    Returns chi_score, entropy, suspicion_level, hist_data.
    """
    # Open image and convert to grayscale
    img = Image.open(image_path).convert('L')
    pixels = np.array(img).flatten()

    # Histogram for charting
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
    hist_data = hist.tolist()

    # Skip nearly uniform images
    if np.std(pixels) < 1:
        return {
            "chi_score": 0,
            "entropy": 0,
            "suspicion_level": "Low",
            "hist_data": hist_data
        }

    # Chi-Square LSB detection
    chi_score = 0.0
    for i in range(0, 256, 2):
        o_even = hist[i]
        o_odd = hist[i+1]
        n = o_even + o_odd
        if n == 0:
            continue
        e = n / 2
        chi_score += ((o_even - e)**2) / e + ((o_odd - e)**2) / e

    # Normalize Chi-Square by total pixels and non-zero bins
    nonzero_bins = np.count_nonzero(hist)
    chi_score_scaled = chi_score / len(pixels) / max(nonzero_bins, 1)

    # Entropy
    entropy = image_entropy(pixels)

    # Dynamic suspicion thresholds
    if chi_score_scaled < 0.1 and entropy < 7.3:
        suspicion_level = "Low"
    elif chi_score_scaled < 0.5 or entropy < 7.8:
        suspicion_level = "Medium"
    else:
        suspicion_level = "High"

    return {
        "chi_score": round(float(chi_score_scaled), 3),
        "entropy": round(float(entropy), 3),
        "suspicion_level": suspicion_level,
        "hist_data": hist_data
    }


# -------------------------
# Test the function
# -------------------------
if __name__ == "__main__":
    for file in ["clean_image.png", "slightly_modified.png", "heavily_stego.png"]:
        result = analyze_image(file)
        print(f"{file}: {result}")
