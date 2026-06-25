#!/usr/bin/env python3

"""
=============================================================
Streak Detection: Canny Edges + Hough Lines
=============================================================
Goal: Detect satellite trails, Starlink chains, and airplane
      streaks in FITS images using classical OpenCV methods.

Pipeline:
  FITS → stretch → inpaint and preprocess → Canny → HoughLinesP → filter → annotate

"""

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch
import h5py


REQUIRED_CV2_ATTRIBUTES = (
    "Canny",
    "GaussianBlur",
    "HoughLinesP",
    "INPAINT_TELEA",
    "NORM_MINMAX",
    "addWeighted",
    "circle",
    "createCLAHE",
    "inpaint",
    "medianBlur",
    "normalize",
    "subtract",
)

DEFAULT_HOUGH_CONFIG = {
    "rho": 1,
    "theta": np.pi / 360,
    "threshold": 10,
    "minLineLength": 150,
    "maxLineGap": 12,
}


class OpenCVUnavailable(RuntimeError):
    """Raised when the installed ``cv2`` module cannot run detection."""


def _require_opencv():
    """Raise a clear exception if OpenCV is missing or incomplete."""

    if cv2 is None:
        raise OpenCVUnavailable("OpenCV/cv2 is not installed.")

    missing = [
        attr for attr in REQUIRED_CV2_ATTRIBUTES if not hasattr(cv2, attr)
    ]
    if missing:
        raise OpenCVUnavailable(
            "OpenCV/cv2 is missing required attributes: "
            + ", ".join(sorted(missing))
        )


def load_and_stretch(filepath, hdu_index=1):
    """Load a FITS file and return both raw float32 and stretched uint8."""

    with fits.open(filepath) as hdul:
        data = hdul[hdu_index].data.astype(np.float32)

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    stretched = AsinhStretch(a=0.1)(norm)
    uint8 = (stretched * 255).astype(np.uint8)

    return data, uint8

def build_star_mask_from_dr(dr_path, img_shape, radius):
    """
    Build a star mask from fistar extractions using circular masks.
    Radius per star = s * psf_scale, where s is the Gaussian sigma.
    """

    mask = np.zeros(img_shape, dtype=np.uint8)

    with h5py.File(dr_path, 'r') as f:

        base_catalog = 'ProjectedSources/Version000'
        xcat = f[f'{base_catalog}/x'][:]
        ycat = f[f'{base_catalog}/y'][:]

    for x, y in zip(xcat, ycat):
        cv2.circle(mask, (int(round(x)), int(round(y))), radius, 255, -1)

    return mask.astype(bool)


def inpaint_stars(uint8_img, star_mask, method):
    """
    Replace star pixels with local background or use of
    cv2 inpaint method to suppress false edges from stars
    before Canny/Hough.

    method:
      'median'  — fast, replaces with global/local background median
      'inpaint' — slower, uses OpenCV inpainting for smoother results
                  (better near bright/large stars)
    """
    if method == 'median':
        # Background = median of non-star pixels
        background_level = np.median(uint8_img[~star_mask])

        cleaned = uint8_img.copy()
        cleaned[star_mask] = background_level

        # Add a touch of noise so the patched regions don't look
        # unnaturally flat to Canny (flat regions are fine, but
        # matching the noise floor avoids a sharp mask-edge artifact)
        noise = np.random.normal(0, 2, size=np.sum(star_mask)).astype(np.int16)
        patched = cleaned[star_mask].astype(np.int16) + noise
        cleaned[star_mask] = np.clip(patched, 0, 255).astype(np.uint8)

        return cleaned

    elif method == 'inpaint':
        # OpenCV inpainting — fills masked regions using surrounding texture
        # Slower but handles large saturated stars more naturally
        mask_uint8 = (star_mask.astype(np.uint8)) * 255
        cleaned = cv2.inpaint(uint8_img, mask_uint8,
                              inpaintRadius=6,
                              flags=cv2.INPAINT_TELEA)
        return cleaned

# ─────────────────────────────────────────────────────────────
# Preprocessing pipeline for streak detection
# ─────────────────────────────────────────────────────────────

def preprocess_for_streaks(uint8_img):
    """
    Prepare the image so that streaks are maximally visible
    and stars/noise are suppressed before edge detection.

    Returns dict of intermediate stages
    """
    stages = {}
    stages['input'] = uint8_img.copy()

    # ── Step 1: Median filter ─────────────────────────────────
    # Removes cosmic rays and hot pixels without blurring streak edges.
    # Use 3×3 — enough to kill isolated spikes.
    denoised = cv2.medianBlur(uint8_img, 3)
    stages['denoised'] = denoised

    # ── Step 2: Background subtraction ───────────────────────
    # Large Gaussian (kernel 61×61) captures the slowly varying
    # sky background and vignetting. Subtracting it flattens the field
    # so thresholding works uniformly everywhere in the image.
    background = cv2.GaussianBlur(denoised, (61, 61), sigmaX=25)
    bg_sub = cv2.subtract(denoised, background)
    bg_sub = cv2.normalize(bg_sub, None, 0, 255, cv2.NORM_MINMAX)
    stages['bg_subtracted'] = bg_sub

    # ── Step 3: CLAHE — local contrast enhancement ────────────
    # Boosts faint streaks in regions where the background is dark.
    # clipLimit=2 prevents over-amplifying noise.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bg_sub)
    stages['clahe'] = enhanced

    # ── Step 4: Unsharp mask ──────────────────────────────────
    # Sharpens edges — makes streak boundaries crisper for Canny.
    # Formula: sharpened = original + weight * (original - blurred)
    blur_for_sharp = cv2.GaussianBlur(enhanced, (9, 9), sigmaX=3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blur_for_sharp, -0.5, 0)
    stages['sharpened'] = sharpened

    return stages


# ─────────────────────────────────────────────────────────────
# Canny Edge Detector
# ─────────────────────────────────────────────────────────────

def apply_canny(img, low_thresh, high_thresh):
    """
    Canny works in 4 internal steps:
      1. Gaussian blur (built-in) — smooths noise
      2. Sobel gradient — finds intensity changes in X and Y
      3. Non-maximum suppression — thins edges to 1px wide
      4. Hysteresis thresholding — two thresholds:
            - pixels ABOVE high_thresh   → definitely an edge
            - pixels BETWEEN low & high  → edge only if connected to a strong edge
            - pixels BELOW low_thresh    → discarded

    For astronomy streaks:
      - Too low threshold  → stars produce false edges everywhere
      - Too high threshold → faint streaks get missed
      - Ratio 1:3-4 (low:high) is a good starting point
    """
    edges = cv2.Canny(img, low_thresh, high_thresh, apertureSize=3, L2gradient=True)
    return edges

# ─────────────────────────────────────────────────────────────
# Hough Line Transform (the core detector)
# ─────────────────────────────────────────────────────────────

def detect_streaks_hough(edge_img, config=None):
    """
    HoughLinesP — Probabilistic Hough Line Transform.

    How it works:
      Every edge pixel "votes" for all lines that pass through it
      in a (rho, theta) parameter space called the accumulator.
      Where many votes accumulate → a line exists.

      'Probabilistic' = only samples a random subset of edge pixels
      → faster, returns line SEGMENTS (x1,y1,x2,y2) not infinite lines.

    Key parameters:
      rho        — distance resolution in pixels (1 = precise)
      theta      — angle resolution in radians (np.pi/180 = 1°)
      threshold  — minimum votes for a line to be accepted
                   higher → fewer, more certain detections
      minLineLength — shortest segment to report (pixels)
                      set to ~10% of image diagonal for real streaks
      maxLineGap    — max gap in pixels between collinear segments
                      to merge them into one line
                      larger → connects broken streaks

    TUNING GUIDE:
      Too many false positives → raise threshold or minLineLength
      Missing faint streaks   → lower threshold, raise maxLineGap
    """

    lines = cv2.HoughLinesP(
        edge_img,
        rho           = config['rho'],
        theta         = config['theta'],
        threshold     = config['threshold'],
        minLineLength = config['minLineLength'],
        maxLineGap    = config['maxLineGap'],
    )

    return lines

# ─────────────────────────────────────────────────────────────
# Merge collinear Hough segments into cleaner detections
# ─────────────────────────────────────────────────────────────

def merge_collinear_segments(lines, angle_tol=3.0, dist_tol=30.0):
    """
    Merge Hough segments that belong to the same physical streak.

    Two segments are merged if:
      1. Their angles agree within angle_tol degrees
      2. The perpendicular distance between their midpoints
         is within dist_tol pixels (same track across the image)

    Returns a reduced list of lines where each entry represents
    one merged streak, keeping the outermost endpoints.
    """
    if lines is None:
        return None

    # Convert to list of [x1,y1,x2,y2] for easier handling
    segs = [line[0].tolist() for line in lines]

    def seg_angle(s):
        return np.degrees(np.arctan2(s[3]-s[1], s[2]-s[0])) % 180

    def perp_distance(s1, s2):
        """Distance from s2's midpoint to the infinite line through s1."""
        x1,y1,x2,y2 = s1
        mx,my = (s2[0]+s2[2])/2, (s2[1]+s2[3])/2
        # Line direction vector
        dx,dy = x2-x1, y2-y1
        length = np.sqrt(dx*dx + dy*dy) + 1e-9
        # Perpendicular distance formula
        return abs(dy*mx - dx*my + x2*y1 - y2*x1) / length

    def merge_two(s1, s2):
        """Return the segment spanning the outermost endpoints of s1 and s2."""
        pts = [(s1[0],s1[1]), (s1[2],s1[3]),
               (s2[0],s2[1]), (s2[2],s2[3])]
        # Project all points onto the direction of s1
        dx = s1[2]-s1[0]; dy = s1[3]-s1[1]
        length = np.sqrt(dx*dx+dy*dy)+1e-9
        projs = [(dx*p[0]+dy*p[1])/length for p in pts]
        i_min = np.argmin(projs); i_max = np.argmax(projs)
        return [pts[i_min][0], pts[i_min][1],
                pts[i_max][0], pts[i_max][1]]

    merged = True
    while merged:
        merged = False
        used = set()
        new_segs = []
        for i in range(len(segs)):
            if i in used:
                continue
            current = segs[i]
            for j in range(i+1, len(segs)):
                if j in used:
                    continue
                a1 = seg_angle(current)
                a2 = seg_angle(segs[j])
                angle_diff = abs(a1-a2)
                angle_diff = min(angle_diff, 180-angle_diff)
                if angle_diff < angle_tol and perp_distance(current, segs[j]) < dist_tol:
                    current = merge_two(current, segs[j])
                    used.add(j)
                    merged = True
            used.add(i)
            new_segs.append(current)
        segs = new_segs

    # Repack into numpy format HoughLinesP returns
    return np.array([[[int(s[0]),int(s[1]),int(s[2]),int(s[3])]] for s in segs])

# ─────────────────────────────────────────────────────────────
# Filter and classify detected lines
# ─────────────────────────────────────────────────────────────

def compute_line_properties(x1, y1, x2, y2):
    """Compute length, angle, and midpoint of a line segment."""
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    angle  = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180   # 0–180°
    mid_x  = (x1 + x2) / 2
    mid_y  = (y1 + y2) / 2
    return length, angle, mid_x, mid_y


def group_parallel_lines(line_props, angle_tol=8.0, dist_tol=50.0):
    """
    Identify groups of parallel lines — the signature of Starlink chains.

    Two lines are 'parallel' if:
      - Their angles differ by < angle_tol degrees
      - Their midpoints are within dist_tol pixels

    Returns list of groups; groups with 2+ lines → Starlink candidate.
    """
    groups = []
    used = set()

    for i, p1 in enumerate(line_props):
        if i in used:
            continue
        group = [i]
        for j, p2 in enumerate(line_props):
            if j <= i or j in used:
                continue
            angle_diff = abs(p1['angle'] - p2['angle'])
            angle_diff = min(angle_diff, 180 - angle_diff)   # handle wrap-around
            mid_dist   = np.sqrt((p1['mid_x'] - p2['mid_x'])**2 +
                                 (p1['mid_y'] - p2['mid_y'])**2)
            if angle_diff < angle_tol and mid_dist < dist_tol:
                group.append(j)
                used.add(j)
        used.add(i)
        groups.append(group)

    return groups


def measure_streak_profile(img, detection, n_samples=10):
    """
    Sample the brightness profile perpendicular to the streak
    at n_samples points along its length.
    Returns the mean peak width (in pixels) of the cross-section.
    A wide smooth profile = real trail (Gaussian PSF convolved with streak)
    A narrow sharp profile = detector artifact
    """
    x1,y1,x2,y2 = detection['x1'],detection['y1'],detection['x2'],detection['y2']
    
    # Unit vector along the streak
    dx, dy = x2-x1, y2-y1
    length = np.sqrt(dx*dx + dy*dy)
    ux, uy = dx/length, dy/length
    
    # Perpendicular unit vector
    px, py = -uy, ux
    
    widths = []
    for i in range(n_samples):
        # Sample point along the streak
        t = (i + 1) / (n_samples + 1)
        cx = x1 + t*dx
        cy = y1 + t*dy
        
        # Sample 20px either side perpendicularly
        profile = []
        for offset in range(-10, 11):
            sx = int(cx + offset*px)
            sy = int(cy + offset*py)
            if 0 <= sy < img.shape[0] and 0 <= sx < img.shape[1]:
                profile.append(float(img[sy, sx]))
        
        if len(profile) < 5:
            continue
            
        profile = np.array(profile)
        peak = profile.max()
        background = profile.min()
        if peak - background < 5:
            continue
            
        # FWHM: how many pixels are above half the peak height
        half_max = background + (peak - background) * 0.5
        fwhm = np.sum(profile > half_max)
        widths.append(fwhm)
    
    return np.mean(widths) if widths else 0.0


def classify_detections(img, lines, min_length):
    """
    Given raw HoughLinesP output, classify each line as:
      - 'satellite'  : single long streak
      - 'starlink'   : member of a parallel group (2+ parallel lines)
      - 'airplane'   : shorter streak, moderate brightness
      - 'noise'      : too short, reject

    Returns list of detection dicts with classification + properties.
    """
    if lines is None:
        return []

    img_diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    detections = []

    # Compute properties for all accepted lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length, angle, mid_x, mid_y = compute_line_properties(x1, y1, x2, y2)

        if length < min_length:
            continue   # too short → noise

        d = {
            'x1': int(x1), 'y1': int(y1),
            'x2': int(x2), 'y2': int(y2),
            'length': float(length),
            'angle':  float(angle),
            'mid_x':  float(mid_x),
            'mid_y':  float(mid_y),
            'label':  'satellite',
        }

        profile_width = measure_streak_profile(img, d)
        d['profile_width'] = profile_width

        # Real trails: PSF-broadened, typically 4–15px wide
        # Artifacts: 1–2px sharp step
        if profile_width < 3.0:
            continue   # reject as detector artifact

        detections.append(d)

    # Check for Starlink groups
    if len(detections) >= 2:
        groups = group_parallel_lines(detections, angle_tol=10, dist_tol=80)
        for group in groups:
            if len(group) >= 2:
                for idx in group:
                    detections[idx]['label'] = 'starlink'

    # Reclassify shorter single streaks as airplane candidates
    for d in detections:
        if d['label'] == 'satellite' and d['length'] < img_diagonal * 0.25:
            d['label'] = 'airplane'

    return detections


def count_raw_streak_lines(fits_path, dr_path, hdu_index=1, config=None):
    """Return the number of merged raw Hough streak candidates.

    Args:
        fits_path: Path to the FITS image to inspect.
        dr_path: Path to the matching data reduction HDF5 file.
        hdu_index: FITS HDU index containing image data.
        config: Optional Hough transform configuration overrides.

    Raises:
        OpenCVUnavailable: If the installed ``cv2`` module is missing the
            functions needed by this detector.
    """

    _require_opencv()

    raw_data, uint8_img = load_and_stretch(fits_path, hdu_index=hdu_index)

    star_mask = build_star_mask_from_dr(
        dr_path, img_shape=raw_data.shape, radius=7
    )
    uint8_img = inpaint_stars(uint8_img, star_mask, method="inpaint")

    stages = preprocess_for_streaks(uint8_img)
    edge_img = apply_canny(
        stages["sharpened"], low_thresh=100, high_thresh=450
    )

    hough_config = DEFAULT_HOUGH_CONFIG.copy()
    if config is not None:
        hough_config.update(config)

    raw_lines = detect_streaks_hough(edge_img, hough_config)
    raw_lines = merge_collinear_segments(
        raw_lines, angle_tol=3.0, dist_tol=30.0
    )

    return len(raw_lines) if raw_lines is not None else 0
