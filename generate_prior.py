import os
import glob
import sys
import numpy as np
import argparse

try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed.")
    print("Please install the required packages by running:")
    print("  pip install opencv-python opencv-contrib-python")
    sys.exit(1)

# Check for required OpenCV contrib module (which provides guidedFilter)
try:
    _ = cv2.ximgproc
except AttributeError:
    print("Error: `opencv-contrib-python` is required for guidedFilter.")
    print("Please install it by running:")
    print("  pip install opencv-contrib-python")
    sys.exit(1)

def histogram_spread(channel):
    hist, _ = np.histogram(channel, bins=256, range=(0, 1))
    return np.std(hist)

def LACC(input_img: np.ndarray, is_vid=False, is_run=False):
    """
    Locally Adaptive Color Correction.
    """
    ## zip [(img_mean, img)], it (b, g, r)
    small, medium, large = sorted(list(zip(cv2.mean(input_img), cv2.split(input_img), ['b', 'g', 'r'])))
    ## sorted by mean (small to large)
    small, medium, large = list(small), list(medium), list(large)
    
    ## exchange wrong channel
    if is_vid and not is_run:
        if histogram_spread(medium[1]) < histogram_spread(large[1]) and (large[0] - medium[0]) < 0.07 and small[2] == 'r':
            large, medium = medium, large
        is_run = True
        
    elif not is_vid:
        if histogram_spread(medium[1]) < histogram_spread(large[1]) and (large[0] - medium[0]) < 0.07 and small[2] == 'r':
            large, medium = medium, large

    ## Max attenuation
    max_attenuation = 1 - (small[1]**1.2)
    max_attenuation = np.expand_dims(max_attenuation, axis=2)

    ## Detail image
    blurred_image = cv2.GaussianBlur(input_img, (7, 7), 0)
    detail_image = input_img - blurred_image
    
    ## corrected large channel
    max_large = cv2.minMaxLoc(large[1])[1]
    min_large = cv2.minMaxLoc(large[1])[0]
    if max_large - min_large > 0:
        large[1] = (large[1] - min_large) * (1/(max_large - min_large))
    large[0] = cv2.mean(large[1])[0]
    
    ## Iter corrected 
    loss = float('inf')
    max_iter = 50 
    iter_count = 0
    while loss > 1e-2 and iter_count < max_iter:
        medium[1] = medium[1] + (large[0] - cv2.mean(medium[1])[0]) * large[1]
        small[1] = small[1] + (large[0] - cv2.mean(small[1])[0]) * large[1]
        loss = abs(large[0] - cv2.mean(medium[1])[0]) + abs(large[0] - cv2.mean(small[1])[0])
        iter_count += 1

    ## b, g, r combine
    b_ch = g_ch = r_ch = None
    for _, ch, color in [large, medium, small]:
        if color == 'b':
            b_ch = ch
        elif color == 'g':
            g_ch = ch
        else:
            r_ch = ch
    img_corrected = cv2.merge([b_ch, g_ch, r_ch])
    
    ## LACC Result
    LACC_img = detail_image + (max_attenuation * img_corrected) + ((1 - max_attenuation) * input_img)
    LACC_img = np.clip(LACC_img, 0.0, 1.0) 

    return LACC_img, is_run

def process_block(block, lc_variance, block_mean, block_variance, beta):
    if block_variance == 0:
        alpha = float("inf")
    else:
        alpha = ((lc_variance) / block_variance)
        
    if alpha < beta:
        block = block_mean + (alpha * (block - block_mean))
    else:
        block = block_mean + (beta * (block - block_mean))

    return block

def LACE(input_img: np.ndarray, beta: float):
    """
    Locally Adaptive Contrast Enhancement.
    """
    ## Process input image
    input_img = input_img.astype(np.uint8) 

    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(input_img)

    ## Set parament
    block_size = 25
    beta_var = beta # Enhancement value
    radius = 10
    eps = 0.01
    
    ## Assuming l_channel
    lc_variance = np.var(l_channel)
    integral_sum, integral_sqsum = cv2.integral2(l_channel)
    height, width = l_channel.shape

    l_channel_processed = np.zeros_like(l_channel, dtype=np.float64)
    weight_sum = np.zeros_like(l_channel, dtype=np.float64)

    ## Process each block
    for i in range(0, height, 20):
        for j in range(0, width, 20):
            ## Define block boundaries
            start_i = i
            end_i = min(i + block_size, height)
            start_j = j
            end_j = min(j + block_size, width)
            
            ## Extract block
            block = l_channel[start_i:end_i, start_j:end_j]

            ## Cal block var, mean
            block_sum = integral_sum[end_i, end_j] - integral_sum[start_i, end_j] - integral_sum[end_i, start_j] + integral_sum[start_i, start_j]
            block_mean = block_sum / ((end_i - start_i) * (end_j - start_j))
            block_sum_sq = integral_sqsum[end_i, end_j] - integral_sqsum[start_i, end_j] - integral_sqsum[end_i, start_j] + integral_sqsum[start_i, start_j]
            block_variance = block_sum_sq / ((end_i - start_i) * (end_j - start_j)) - np.square(block_mean)

            ## Process block
            block_processed = process_block(block, lc_variance, block_mean, block_variance, beta_var)
            
            ## Put block back into image
            l_channel_processed[start_i:end_i, start_j:end_j] += block_processed
            weight_sum[start_i:end_i, start_j:end_j] += 1.0

    l_channel_processed /= weight_sum
    l_channel_processed = np.clip(l_channel_processed, 0, 255).astype('uint8')
    
    ## guided filter
    l_channel_processed = cv2.ximgproc.guidedFilter(l_channel, l_channel_processed, radius, eps)

    ## ab channel balance
    a_mean = np.mean(a_channel)
    b_mean = np.mean(b_channel)
    a_channel = a_channel.astype(np.float64)
    b_channel = b_channel.astype(np.float64)
    
    if a_mean > b_mean:
        b_channel = (b_channel + b_channel * ((a_mean - b_mean) / (a_mean + b_mean)))
    else:
        a_channel = (a_channel + a_channel * ((b_mean - a_mean)/(a_mean + b_mean)))
        
    b_channel = np.clip(b_channel, 0, 255).astype(np.uint8)
    a_channel = np.clip(a_channel, 0, 255).astype(np.uint8)

    ## Combine channel
    Result = cv2.merge([l_channel_processed, a_channel, b_channel])
    Result = cv2.cvtColor(Result, cv2.COLOR_LAB2BGR)
    return Result

def generate_prior_image(img_path, output_path, beta=1.5):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return False
        
    print(f"Processing {os.path.basename(img_path)}...")
    
    # MLLE processing
    img_lacc, _ = LACC(img / 255.0)
    img_prior = LACE(img_lacc * 255.0, beta)
    
    cv2.imwrite(output_path, img_prior)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Prior Image using Python implementation of MLLE")
    parser.add_argument('base_dir', type=str, help="Base directory containing the standard subfolders (e.g., EUVP, UIEB)")
    parser.add_argument('-b', '--beta', type=float, default=1.5, help="Beta value for enhancement (default: 1.5)")
    
    args = parser.parse_args()
    
    folders_to_process = {
        'testA': 'testPrior',
        'trainA': 'trainPrior',
        'valA': 'valPrior'
    }
    
    for in_folder, out_folder in folders_to_process.items():
        input_dir = os.path.join(args.base_dir, in_folder)
        output_dir = os.path.join(args.base_dir, out_folder)
        
        if not os.path.exists(input_dir):
            print(f"Skipping {input_dir}: Directory does not exist.")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        image_extensions = ('*.png', '*.jpg', '*.jpeg')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
            
        # Remove duplicates
        image_files = list(set(image_files))
            
        if not image_files:
            print(f"No images found in {input_dir}")
        else:
            print(f"Found {len(image_files)} images in {input_dir}, starting conversion to {output_dir}...")
            for img_path in image_files:
                out_name = os.path.basename(img_path)
                out_path = os.path.join(output_dir, out_name)
                generate_prior_image(img_path, out_path, beta=args.beta)
            print(f"Prior images successfully saved to {output_dir}")
