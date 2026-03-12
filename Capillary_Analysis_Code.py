import cv2
import numpy as np
import sys
import math

# ==========================================
# SECTION 1: IMPORTS & SETUP
# ==========================================

try:
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    from skimage import color 
except ImportError as e:
    print(f"\n[CRITICAL ERROR] Missing Library: {e}")
    print("Please run: pip install opencv-python numpy scipy scikit-image matplotlib\n")
    sys.exit(1)

import matplotlib
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    HEADLESS = False
except Exception as e:
    print(f"Warning: UI Backend not available ({e}). Running in Headless Mode (saving files).")
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HEADLESS = True

# ==========================================
# SECTION 2: CONFIGURATION
# ==========================================

DEFAULT_CALIBRATION = 1.48
PERIVASCULAR_MARGIN_UM = 11

# [UPDATED FILTERS] Lowered to catch fainter capillaries in softer lighting
MIN_HEIGHT_UM = 40       
MAX_WIDTH_UM = 400
MIN_ASPECT_RATIO = 0.5
MIN_SKELETON_RATIO = 0.8 
MAX_TILT_ANGLE_DEG = 45

MIN_REDNESS_DIFFERENCE = 1.5

HEMO_MIN_SOLIDITY = 0.85
HEMO_MAX_ASPECT_RATIO = 2.0
HEMO_MIN_DIAMETER_UM = 35
HEMO_MAX_INTENSITY = 100

BASE_MIN_AREA_PX = 50
BASE_MIN_HOLE_PX = 10
MIN_VARIANCE_THRESHOLD = 30

MAX_TORTUOSITY_LIMIT = 10.0

# ==========================================
# SECTION 3: ALGORITHMS
# ==========================================

def algo_get_redness_heatmap(img_color):
    if img_color is None: return None
    lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    a_blurred = cv2.GaussianBlur(a, (3,3), 0)
    return a_blurred

def algo_validate_candidate_color(redness_map, binary_mask_single):
    dilated = cv2.dilate(binary_mask_single, np.ones((9,9), np.uint8), iterations=2)
    bg_ring = cv2.subtract(dilated, binary_mask_single)
    if cv2.countNonZero(bg_ring) == 0: return 0.0

    mean_inside = cv2.mean(redness_map, mask=binary_mask_single)[0]
    mean_bg = cv2.mean(redness_map, mask=bg_ring)[0]
    return mean_inside - mean_bg

def algo_generate_noise_mask(gray_img, block_size=15):
    mu = cv2.blur(gray_img, (block_size, block_size))
    mu2 = cv2.blur(gray_img * gray_img, (block_size, block_size))
    sigma = cv2.sqrt(mu2 - mu * mu)
    sigma = cv2.normalize(sigma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(sigma, MIN_VARIANCE_THRESHOLD, 255, cv2.THRESH_BINARY)
    return mask

def algo_enhance_contrast(gray_img):
    blurred = cv2.medianBlur(gray_img, 3)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    return enhanced

def algo_bridge_breaker(binary_mask):
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    eroded = cv2.erode(binary_mask, h_kernel, iterations=1)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    cleaned = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, v_kernel, iterations=1)
    restore_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    final = cv2.dilate(cleaned, restore_kernel, iterations=1)
    return final

def algo_glue_segments(binary_mask):
    glue_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    glued = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, glue_kernel)
    return glued

def algo_cut_crossing_mask(single_object_mask, min_hole_px):
    single_object_mask = single_object_mask.astype(np.uint8)
    cnts_out = cv2.findContours(single_object_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts_out[0] if len(cnts_out) == 2 else cnts_out[1]
    hierarchy = cnts_out[1] if len(cnts_out) == 2 else cnts_out[2]
    cut_mask = single_object_mask.copy()
    was_cut = False
    if hierarchy is not None:
        for i, h_data in enumerate(hierarchy[0]):
            if h_data[3] != -1:
                area = cv2.contourArea(contours[i])
                if area > min_hole_px:
                    top_y_idx = np.argmin(contours[i][:, 0, 1])
                    crossing_start_pt = contours[i][top_y_idx][0]
                    start_pt = (crossing_start_pt[0], crossing_start_pt[1])
                    end_pt = (crossing_start_pt[0], crossing_start_pt[1] - 20)
                    cv2.line(cut_mask, start_pt, end_pt, 0, thickness=2)
                    was_cut = True
    return cut_mask, was_cut

def algo_hildebrand_thickness(binary_mask):
    return distance_transform_edt(binary_mask) * 2

def algo_sort_skeleton_points(points_list):
    if len(points_list) < 2: return points_list
    points = np.array(points_list)
    centroid = np.mean(points, axis=0)
    dists_from_center = np.linalg.norm(points - centroid, axis=1)
    start_idx = np.argmax(dists_from_center)
    ordered = [tuple(points[start_idx])]
    mask = np.ones(len(points), dtype=bool)
    mask[start_idx] = False
    current_pt = points[start_idx]
    
    for _ in range(len(points) - 1):
        remaining_indices = np.where(mask)[0]
        if len(remaining_indices) == 0: break
        remaining_pts = points[remaining_indices]
        dists = np.linalg.norm(remaining_pts - current_pt, axis=1)
        nearest_local_idx = np.argmin(dists)
        actual_idx = remaining_indices[nearest_local_idx]
        current_pt = points[actual_idx]
        ordered.append(tuple(current_pt))
        mask[actual_idx] = False
    return ordered

def algo_triangular_index(points):
    if len(points) < 3: return 1.0
    
    pts_arr = np.array(points)
    p1 = pts_arr[0]
    p2 = pts_arr[-1]
    
    denom = np.linalg.norm(p2 - p1)
    if denom == 0: return 1.0
    
    max_dist = 0
    for p0 in pts_arr:
        num = np.abs((p2[0] - p1[0]) * (p1[1] - p0[1]) - (p1[0] - p0[0]) * (p2[1] - p1[1]))
        dist = num / denom
        if dist > max_dist:
            max_dist = dist
            
    side1 = np.sqrt(max_dist**2 + (denom/2)**2)
    tri_index = (side1 * 2) / denom
    return tri_index

def algo_isolate_cap_and_measure(mask_single, thick_map_global, skel_pts):
    if not skel_pts or len(skel_pts) < 5:
        return 0, 0, (0,0), (0,0)

    apex_idx = np.argmin([p[1] for p in skel_pts])
    apex_pt = skel_pts[apex_idx]

    cnts_out = cv2.findContours(mask_single.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts_out[0] if len(cnts_out) == 2 else cnts_out[1]
    if not contours:
         return 0, 0, apex_pt, apex_pt
         
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # [BUG FIX] Deeper cap isolation ensures the ellipse fits the limbs better
    cap_height = int(w * 1.2) 
    if cap_height < 15: cap_height = 15

    cap_mask = np.zeros_like(mask_single)
    cap_top = y
    cap_bottom = min(y + cap_height, y + h)
    
    cv2.drawContours(cap_mask, contours, 0, 255, -1)
    cap_mask[cap_bottom:, :] = 0

    cap_cnts, _ = cv2.findContours(cap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cap_cnts or len(cap_cnts[0]) < 5:
        d = thick_map_global[apex_pt[1], apex_pt[0]] if apex_pt[1] < thick_map_global.shape[0] else 0
        return d, d, apex_pt, apex_pt

    (xc, yc), (MA, ma), angle = cv2.fitEllipse(cap_cnts[0])
    
    a = np.deg2rad(angle); b = np.deg2rad(angle + 90)
    r_minor = min(MA, ma) / 2
    dx = r_minor * math.cos(b); dy = r_minor * math.sin(b)
    
    p1 = (int(xc + dx), int(yc + dy))
    p2 = (int(xc - dx), int(yc - dy))
    
    if p1[0] < p2[0]: 
        p_art, p_ven = p1, p2
    else: 
        p_art, p_ven = p2, p1
        
    img_h, img_w = thick_map_global.shape
    
    # [BUG FIX] Local search logic! If the ellipse point lands outside the mask, 
    # scan a 4-pixel radius to find the actual thickest part of the blood vessel.
    def get_local_max_thickness(pt, radius=4):
        x_pt, y_pt = pt
        x_min = max(0, x_pt - radius)
        x_max = min(img_w, x_pt + radius + 1)
        y_min = max(0, y_pt - radius)
        y_max = min(img_h, y_pt + radius + 1)
        region = thick_map_global[y_min:y_max, x_min:x_max]
        if region.size == 0: return 0.0
        return np.max(region)

    d_a = get_local_max_thickness(p_art)
    d_v = get_local_max_thickness(p_ven)
    
    # Only fallback to apex if the local search entirely fails (very rare now)
    if d_a == 0:
        d_a = thick_map_global[apex_pt[1], apex_pt[0]] if apex_pt[1] < img_h else 0
    if d_v == 0:
        d_v = thick_map_global[apex_pt[1], apex_pt[0]] if apex_pt[1] < img_h else 0
        
    return d_a, d_v, p_art, p_ven

def algo_color_contrast_ciede2000(lab_img_float, binary_mask, mpp):
    margin_px = int(PERIVASCULAR_MARGIN_UM / mpp)
    if margin_px < 1: margin_px = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px*2, margin_px*2))
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    skin_mask = cv2.subtract(dilated, binary_mask)
    
    if np.sum(skin_mask) == 0: return 0.0
    
    cap_mean = cv2.mean(lab_img_float, mask=binary_mask)[:3]
    skin_mean = cv2.mean(lab_img_float, mask=skin_mask)[:3]
    
    dE = color.deltaE_ciede2000(np.array(cap_mean), np.array(skin_mean))
    return float(dE)

# ==========================================
# SECTION 4: VISUALIZATION
# ==========================================

def visualize_all_results(original_img, capillaries, hemorrhages, filename="analysis_full_map.png"):
    print("STATUS: Generating Full Map Visualization...")
    vis_img = original_img.copy()
    overlay = vis_img.copy()
    for cap in capillaries:
        apex, p_art, p_ven = cap['pts']
        bbox = cap['bbox']
        cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
        
        cv2.circle(vis_img, apex, 3, (0, 255, 255), -1)
        cv2.circle(vis_img, p_art, 2, (255, 0, 0), -1) 
        cv2.circle(vis_img, p_ven, 2, (0, 0, 255), -1) 
        
        skel_pts = cap['skel_coords']
        if len(skel_pts) > 1:
            pts_arr = np.array(skel_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts_arr], False, (0, 255, 0), 1)

        cv2.putText(vis_img, f"C{cap['id']}", (bbox[0], bbox[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    for hemo in hemorrhages:
        bbox = hemo['bbox']
        cnt = hemo['contour']
        cv2.drawContours(vis_img, [cnt], -1, (0, 0, 255), 2)
        cv2.putText(vis_img, "HEMO", (bbox[0], bbox[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    alpha = 0.8
    cv2.addWeighted(vis_img, alpha, overlay, 1 - alpha, 0, vis_img)
    if HEADLESS:
        cv2.imwrite(filename, vis_img)
    else:
        plt.figure(figsize=(14, 9))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Analysis: {len(capillaries)} Capillaries, {len(hemorrhages)} Hemorrhages")
        plt.axis('off')
        plt.show(block=False)

def show_debug_panel(green, redness_map, mask_structural, binary_final, capillaries, hemorrhages):
    print("STATUS: Opening Debug Panel...")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Debug: V17 (Fixed Geometry & Sensitivity)", fontsize=16)

    plt.subplot(2, 3, 1); plt.imshow(green, cmap='gray'); plt.title("1. Green Channel")
    plt.axis('off')

    plt.subplot(2, 3, 2); plt.imshow(redness_map, cmap='jet'); plt.title("2. Redness Heatmap")
    plt.axis('off')

    plt.subplot(2, 3, 3); plt.imshow(mask_structural, cmap='gray'); plt.title("3. Structure (Noisy)")
    plt.axis('off')
    
    plt.subplot(2, 3, 4); plt.imshow(binary_final, cmap='gray'); plt.title("4. Validated Structure")
    plt.axis('off')

    debug_canvas = np.zeros_like(green)
    debug_canvas = cv2.cvtColor(debug_canvas, cv2.COLOR_GRAY2BGR)
    for c in capillaries:
        bbox = c['bbox']
        cv2.rectangle(debug_canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    for h in hemorrhages:
        bbox = h['bbox']
        cv2.rectangle(debug_canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    plt.subplot(2, 3, 5); plt.imshow(debug_canvas); plt.title(f"5. Result: {len(capillaries)} Caps, {len(hemorrhages)} Hemos")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.5, "Updates:\n1. Relaxed Redness/Variance Filters.\n2. Added Local Geometry Search\n   to prevent identical diameters.", fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=False)

# ==========================================
# SECTION 5: MAIN PROCESSING
# ==========================================

def process_image(filepath, is_test_mode=False, show_debug_plots=False):
    print("\n" + "="*110)
    if is_test_mode:
        print("STATUS: Generating Test Image...")
        cal_val = DEFAULT_CALIBRATION
        img_color = np.zeros((400, 500, 3), dtype=np.uint8)
        img_color[:] = (200, 200, 200)
        cv2.ellipse(img_color, (150, 200), (30, 100), 0, 180, 360, (50, 50, 80), 15)
        cv2.circle(img_color, (350, 200), 40, (30, 30, 80), -1)
        cv2.circle(img_color, (400, 50), 10, (100, 100, 100), -1)
        green_channel = img_color[:, :, 1]
    else:
        try:
            user_cal = input(f"Enter Microns/Pixel [Default {DEFAULT_CALIBRATION}]: ").strip()
            cal_val = float(user_cal) if user_cal else DEFAULT_CALIBRATION
        except ValueError: cal_val = DEFAULT_CALIBRATION
        filepath_clean = filepath.strip().strip("'\"").replace("\\ ", " ")
        img_color = cv2.imread(filepath_clean)
        if img_color is None:
            print(f"[ ERROR: CANNOT READ FILE ]\nCheck file path: {filepath_clean}\n")
            return
        green_channel = img_color[:, :, 1]
    
    scale_factor = (DEFAULT_CALIBRATION / cal_val)
    dyn_min_area = BASE_MIN_AREA_PX * (scale_factor ** 2)
    dyn_min_hole = BASE_MIN_HOLE_PX * (scale_factor ** 2)
    dyn_min_height = MIN_HEIGHT_UM / cal_val
    dyn_hemo_min_area = (HEMO_MIN_DIAMETER_UM / 2)**2 * math.pi / (cal_val**2)

    print("STATUS: Enhancing structure (Contrast)...")
    variance_mask = algo_generate_noise_mask(green_channel)
    gray_enhanced = algo_enhance_contrast(green_channel)
    filter_size = 25
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))
    blackhat = cv2.morphologyEx(gray_enhanced, cv2.MORPH_BLACKHAT, kernel)
    blur_k = (3, 3)
    blackhat_blurred = cv2.GaussianBlur(blackhat, blur_k, 0)
    otsu_val, mask_otsu = cv2.threshold(blackhat_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_adapt = cv2.adaptiveThreshold(blackhat_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, -5) 
    clean_kernel_adapt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_adapt_clean = cv2.morphologyEx(mask_adapt, cv2.MORPH_OPEN, clean_kernel_adapt, iterations=1)
    binary_combined = cv2.bitwise_or(mask_otsu, mask_adapt_clean)
    binary_structural = cv2.bitwise_and(binary_combined, variance_mask)
    binary_glued = algo_glue_segments(binary_structural)
    binary_final = algo_bridge_breaker(binary_glued)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binary_final = cv2.morphologyEx(binary_final, cv2.MORPH_OPEN, clean_kernel, iterations=1)

    print("STATUS: Generating Color Maps...")
    redness_map = algo_get_redness_heatmap(img_color)
    
    img_float = img_color.astype(np.float32) / 255.0
    lab_img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)

    h, w = binary_final.shape
    thickness_map_global = algo_hildebrand_thickness(binary_final)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(binary_final)
    capillaries = []
    hemorrhages = []
    print(f"STATUS: Validating {num-1} candidates via Dual-Factor logic...")

    for i in range(1, num):
        h_obj = stats[i, cv2.CC_STAT_HEIGHT]
        w_obj = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < dyn_min_area: continue
        
        mask_raw = (labels == i).astype(np.uint8) * 255
        cnts_temp, _ = cv2.findContours(mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_temp: continue
        contour = cnts_temp[0]
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0
        aspect_ratio = h_obj / float(w_obj)
        redness_diff = algo_validate_candidate_color(redness_map, mask_raw)
        
        # 1. HEMORRHAGE CHECK
        if (solidity > HEMO_MIN_SOLIDITY and
            aspect_ratio < HEMO_MAX_ASPECT_RATIO and
            area > dyn_hemo_min_area):
            if redness_diff > 1.0:
                dE = algo_color_contrast_ciede2000(lab_img_float, mask_raw, cal_val)
                hemorrhages.append({
                    'id': i,
                    'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                             stats[i, cv2.CC_STAT_LEFT]+w_obj, stats[i, cv2.CC_STAT_TOP]+h_obj),
                    'contour': contour,
                    'area_um2': area * (cal_val**2),
                    'delta_E': dE
                })
            continue

        # 2. CAPILLARY CHECK
        if h_obj < dyn_min_height: continue
        if aspect_ratio < MIN_ASPECT_RATIO: continue

        if len(contour) >= 5:
            (xc, yc), (MA, ma), angle = cv2.fitEllipse(contour)
            tilt = min(angle, 180 - angle)
            if tilt > MAX_TILT_ANGLE_DEG: continue

        if redness_diff < MIN_REDNESS_DIFFERENCE:
            continue

        mask_cut, was_cut = algo_cut_crossing_mask(mask_raw, dyn_min_hole)
        
        skel = skeletonize(mask_cut // 255).astype(np.uint8)
        
        y_coords, x_coords = np.where(skel > 0)
        if len(x_coords) < 10: continue
        
        raw_pts = list(zip(x_coords, y_coords))
        
        skel_pts = algo_sort_skeleton_points(raw_pts)
        skel_len_px = len(skel_pts)
        skel_ratio = skel_len_px / float(h_obj)
        
        if skel_ratio < MIN_SKELETON_RATIO: continue

        apex_idx = np.argmin([p[1] for p in skel_pts])
        apex_pt = skel_pts[apex_idx]

        d_a, d_v, pt_a, pt_v = algo_isolate_cap_and_measure(mask_cut, thickness_map_global, skel_pts)
        d_apex = thickness_map_global[apex_pt[1], apex_pt[0]] if apex_pt[1] < h else 0
        
        if pt_a[1] < apex_pt[1] or pt_v[1] < apex_pt[1]: continue

        tort = algo_triangular_index(skel_pts)

        pts_arr = np.array(skel_pts)
        raw_len_px = np.sum(np.linalg.norm(pts_arr[1:] - pts_arr[:-1], axis=1))
        
        dE = algo_color_contrast_ciede2000(lab_img_float, mask_raw, cal_val)
        
        capillaries.append({
            'id': i, 'apex': apex_pt, 'score': h_obj,
            'vals': (d_apex, d_a, d_v, tort, dE, raw_len_px),
            'pts': (apex_pt, pt_a, pt_v),
            'status': "Cut" if was_cut else "OK",
            'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                     stats[i, cv2.CC_STAT_LEFT]+w_obj, stats[i, cv2.CC_STAT_TOP]+h_obj),
            'skel_coords': skel_pts
        })

    # --- 5. RESULTS ---
    capillaries.sort(key=lambda x: x['apex'][0])
    inter_apex_gaps = []
    for k in range(len(capillaries) - 1):
        gap_um = abs(capillaries[k+1]['apex'][0] - capillaries[k]['apex'][0]) * cal_val
        inter_apex_gaps.append(gap_um)
        
    avg_gap = np.mean(inter_apex_gaps) if inter_apex_gaps else 0.0
    print("\n" + "="*110)
    print(f" RESULTS: {len(capillaries)} Capillaries | {len(hemorrhages)} Hemorrhages")
    print(f" DENSITY: Avg Gap {avg_gap:.2f} µm")
    print("-" * 110)
    print(f"{'TYPE':<8} {'ID':<4} {'APEX(μm)':<10} {'ART(μm)':<10} {'VEN(μm)':<10} {'TORT':<10} {'ΔE':<10}")
    print("-" * 110)

    if not capillaries and not hemorrhages:
        print("No features found.")
        
    for cap in capillaries:
        da, dart, dven, tort, de, raw_len = cap['vals']
        
        if tort > MAX_TORTUOSITY_LIMIT:
            print(f"CAPILLARY #{cap['id']:<3} Morphologically Abnormal - non-linear Path")
        else:
            print(f"CAPILLARY #{cap['id']:<3} {da*cal_val:<10.2f} {dart*cal_val:<10.2f} "
                  f"{dven*cal_val:<10.2f} {tort:<10.4f} {de:<10.2f}")
                  
    print("-" * 110)
    for hemo in hemorrhages:
        print(f"HEMORRHAGE #{hemo['id']:<3} Area: {hemo['area_um2']:<10.2f} µm² {'':<22} Contrast: {hemo['delta_E']:.2f}")

    print("="*110 + "\n")

    if show_debug_plots:
        show_debug_panel(green_channel, redness_map, binary_structural, binary_final, capillaries, hemorrhages)

    if capillaries or hemorrhages:
        visualize_all_results(img_color, capillaries, hemorrhages)
        
    if (show_debug_plots or capillaries or hemorrhages) and not HEADLESS:
        plt.show()

if __name__ == "__main__":
    filepath = None
    is_test = False
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        is_test = False
        debug_choice = input("Enable Step-by-Step Debug Plots? (y/n): ").strip().lower()
    else:
        print("NAILFOLD CAPILLARY ANALYZER (V17: Fixed Geometry & Sensitivity)")
        choice = input("Run Test (1) or File (2): ").strip()
        if choice == '1':
            is_test = True
            debug_choice = input("Enable Step-by-Step Debug Plots? (y/n): ").strip().lower()
        else:
            is_test = False
            filepath = input("File Path: ").strip()
            debug_choice = input("Enable Step-by-Step Debug Plots? (y/n): ").strip().lower()

    show_debug = True if debug_choice == 'y' else False
    if is_test:
        process_image(None, True, show_debug_plots=show_debug)
    elif filepath:
        process_image(filepath, False, show_debug_plots=show_debug)
    else:
        print("No file provided. Exiting.")
