from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import os
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from typing import List, Dict
import json
# Grounding DINO imports
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything imports
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI()

# Configuration
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
GROUNDED_CHECKPOINT = "groundingdino_swinb_cogcoor.pth"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


# Initialize models at startup
@app.on_event("startup")
async def startup_event():
    global grounding_dino_model, sam_predictor

    print(f"Using device: {DEVICE}")
    print("Loading models...")

    # Load Grounding DINO model
    grounding_dino_model = load_model(CONFIG_PATH, GROUNDED_CHECKPOINT, DEVICE)

    # Load SAM model
    sam_model = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam_model = sam_model.to(DEVICE)
    sam_predictor = SamPredictor(sam_model)

    print("Models loaded successfully!")


@app.post("/analyze")
async def analyze_image(
        image: UploadFile = File(...),
        text_prompt: str = Form(...),
        box_threshold: float = Form(0.3),
        text_threshold: float = Form(0.25)
):
    try:
        # Save uploaded image to output directory
        image_path = os.path.join(OUTPUT_DIR, "input_image.jpg")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        print(f"Processing image with prompt: {text_prompt}")

        # Load and process image
        image_pil, image_tensor = load_image(image_path)

        # Get Grounding DINO output
        boxes_filt, pred_phrases = get_grounding_output(
            grounding_dino_model,
            image_tensor,
            text_prompt,
            box_threshold,
            text_threshold,
            device=DEVICE
        )

        # Process image for SAM
        image_array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image_array)

        # Transform boxes
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_array.shape[:2]).to(DEVICE)

        # Generate masks
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Create visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(image_array)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')

        # Save output image
        output_path = os.path.join(OUTPUT_DIR, "grounded_sam_output.jpg")
        plt.savefig(
            output_path,
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0
        )
        plt.close()

        print(f"Analysis complete. Result saved to: {output_path}")

        if os.path.exists(output_path):
            return FileResponse(
                output_path,
                media_type="image/jpeg",
                filename="grounded_sam_output.jpg"
            )
        else:
            return {"error": "Failed to generate output image"}

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return {"error": str(e)}


def mask_to_polygon(mask: np.ndarray, tolerance: float = 0.5) -> List[List[int]]:
    """
    마스크를 단일 다각형 좌표로 변환합니다.

    Args:
        mask: 2D numpy array (binary mask)
        tolerance: 다각형 단순화 파라미터 (낮을수록 더 정확한 윤곽선)

    Returns:
        List of [x,y] coordinates representing the polygon
    """
    try:
        # Debug: Print mask statistics
        print(f"Mask shape: {mask.shape}")
        print(f"Mask value range: [{mask.min()}, {mask.max()}]")

        # Ensure mask is binary and convert to uint8
        mask_binary = (mask > 0.1).astype(np.uint8) * 255

        # Debug: Print number of non-zero pixels
        print(f"Number of non-zero pixels: {np.count_nonzero(mask_binary)}")

        # Find contours with CHAIN_APPROX_NONE for maximum detail
        contours, hierarchy = cv2.findContours(
            mask_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        # Debug: Print number of contours found
        print(f"Number of contours found: {len(contours)}")

        if not contours:
            print("No contours found in mask")
            return []

        # Get the largest contour by area
        main_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(main_contour)
        print(f"Largest contour area: {contour_area}")

        # Only proceed if the contour area is significant
        if contour_area < 10:  # Minimum area threshold
            print("Contour area too small")
            return []

        # Use a smaller epsilon for more precise approximation
        epsilon = tolerance * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)

        # Debug: Print number of points in approximated polygon
        print(f"Number of points in approximated polygon: {len(approx)}")

        # Convert the contour points to a list of [x,y] coordinates
        polygon = [[int(point[0][0]), int(point[0][1])] for point in approx]

        # Ensure we have enough points for a meaningful polygon
        if len(polygon) < 3:
            print("Not enough points for a polygon")
            return []

        return polygon

    except Exception as e:
        print(f"Error in mask_to_polygon: {str(e)}")
        return []


def find_intersection(params1, params2):
    """Find intersection point of two lines given their parameters."""
    det = params1[0] * params2[1] - params2[0] * params1[1]
    if abs(det) < 0.5:  # lines are approximately parallel
        return (-1, -1)

    x = (params2[1] * -params1[2] - params1[1] * -params2[2]) / det
    y = (params1[0] * -params2[2] - params2[0] * -params1[2]) / det
    return (int(x), int(y))


def calc_line_params(p1, p2):
    """Calculate line equation parameters (ax + by + c = 0)."""
    if p2[1] - p1[1] == 0:
        a, b = 0.0, -1.0
    elif p2[0] - p1[0] == 0:
        a, b = -1.0, 0.0
    else:
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = -1.0
    c = (-a * p1[0]) - b * p1[1]
    return a, b, c


def filter_lines(lines, rho_threshold=10, theta_threshold=np.pi / 36):
    """Filter similar lines based on rho and theta values."""
    if len(lines) == 0:
        return []

    filtered_lines = []
    lines = lines.squeeze()

    for line in lines:
        rho1, theta1 = line[0], line[1]
        exists = False

        for filtered_line in filtered_lines:
            rho2, theta2 = filtered_line[0], filtered_line[1]
            if (abs(rho1 - rho2) < rho_threshold and
                    abs(theta1 - theta2) < theta_threshold):
                exists = True
                break

        if not exists:
            filtered_lines.append(line)
            if len(filtered_lines) == 4:  # We only need 4 lines
                break

    return filtered_lines


def find_board_corners(mask):
    """Find board corners using Hough Transform."""
    # Apply edge detection
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    # Apply Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)
    if lines is None or len(lines) < 4:
        print(f"not enough HoughLines {len(lines)}")
        return None, None, None

    # Filter lines to get the strongest 4 lines
    filtered_lines = filter_lines(lines)
    if len(filtered_lines) < 4:
        print(f"not enough filtered_lines {len(filtered_lines)}")
        return None, None, None

    # Convert lines from polar to cartesian coordinates
    cartesian_lines = []
    for rho, theta in filtered_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cartesian_lines.append([x1, y1, x2, y2])

    # Find corners from line intersections
    corners = []
    line_params = [calc_line_params([line[0], line[1]], [line[2], line[3]])
                   for line in cartesian_lines]

    for i in range(len(line_params)):
        for j in range(i + 1, len(line_params)):
            corner = find_intersection(line_params[i], line_params[j])
            if (corner[0] >= 0 and corner[1] >= 0 and
                    corner[0] < mask.shape[1] and corner[1] < mask.shape[0]):
                corners.append(corner)

    # Sort corners to get a consistent order
    if len(corners) == 4:
        corners = np.array(corners)
        center = np.mean(corners, axis=0)

        # Calculate angles between points and center
        angles = np.arctan2(corners[:, 1] - center[1],
                            corners[:, 0] - center[0])

        # Sort corners by angle
        sorted_indices = np.argsort(angles)
        corners = corners[sorted_indices]

        return corners, cartesian_lines, filtered_lines

    return None, None, None


def find_board_corners_with_fallback(mask):
    """
    Find board corners using Hough Transform with goodFeaturesToTrack as fallback.
    Falls back if detected area is less than 20% of image area.

    Args:
        mask: Binary mask image (numpy array)

    Returns:
        tuple: (corners, cartesian_lines, filtered_lines)
        - corners: numpy array of corner coordinates or None if detection fails
        - cartesian_lines: list of line coordinates or None
        - filtered_lines: list of filtered Hough lines or None
    """
    # First try with Hough Transform
    corners, cartesian_lines, filtered_lines = find_board_corners(mask)

    # Check if corners were found
    if corners is not None and len(corners) == 4:
        # Calculate the area of the detected quadrilateral
        height, width = mask.shape
        total_image_area = height * width

        # Calculate area using Shoelace formula
        corners_wrapped = np.vstack((corners, corners[0]))  # Add first point at end
        area = 0
        for i in range(len(corners)):
            area += corners_wrapped[i][0] * corners_wrapped[i + 1][1]
            area -= corners_wrapped[i + 1][0] * corners_wrapped[i][1]
        area = abs(area) / 2

        # Check if area is at least 20% of image area
        if area >= 0.2 * total_image_area:
            return corners, cartesian_lines, filtered_lines
        else:
            print(f"Detected area ({area:.2f}) is less than 20% of image area ({total_image_area:.2f}), falling back to goodFeaturesToTrack")
    else:
        print("Hough transform corner detection failed, trying goodFeaturesToTrack...")

    try:
        # Convert to proper format for goodFeaturesToTrack
        mask_uint8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask

        # Get image dimensions
        height, width = mask_uint8.shape

        # Calculate dynamic parameters based on image size
        min_dimension = min(height, width)

        # minDistance: 이미지 크기의 약 2%~5% 정도가 적절
        min_distance = max(100, int(min_dimension * 0.1))  # 최소 20픽셀

        # blockSize: 이미지 크기에 따라 3~15 사이의 값으로 조정
        block_size = max(11, int(min_dimension * 0.01))
        if block_size % 2 == 0:  # blockSize는 홀수여야 함
            block_size += 1
        block_size = min(15, block_size)  # 최대값 제한

        # Add padding to help with corner detection near edges
        pad_size = block_size
        padded_mask = np.pad(mask_uint8, ((pad_size, pad_size), (pad_size, pad_size)),
                             mode='constant', constant_values=0)

        # Apply Gaussian blur to reduce noise
        blur_size = max(3, block_size // 2)
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(padded_mask, (blur_size, blur_size), 0)

        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            blurred,
            maxCorners=8,  # Detect more corners than needed
            qualityLevel=0.01,
            minDistance=min_distance,
            blockSize=block_size
        )

        if corners is None or len(corners) < 4:
            print("Failed to detect enough corners with goodFeaturesToTrack")
            print(f"Used parameters - minDistance: {min_distance}, blockSize: {block_size}")
            return None, None, None

        # Remove padding offset from coordinates
        corners = corners.squeeze() - pad_size

        # If we have more than 4 corners, select the best 4
        if len(corners) > 4:
            # Calculate distances from center
            center = np.mean(corners, axis=0)
            distances = np.linalg.norm(corners - center, axis=1)

            # Get indices of the 4 corners furthest from center
            corner_indices = np.argsort(distances)[-4:]
            corners = corners[corner_indices]

        # Sort corners in clockwise order
        center = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1] - center[1],
                            corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        corners = corners[sorted_indices]

        return corners, None, None  # Return None for lines since we didn't use Hough

    except Exception as e:
        print(f"Error in goodFeaturesToTrack fallback: {str(e)}")
        return None, None, None

# Update the visualization code to handle the fallback case
def visualize_corners_on_image(ax, image, corners, cartesian_lines=None):
    """
    Visualize detected corners and lines on the image

    Args:
        ax: matplotlib axis
        image: original image
        corners: detected corners
        cartesian_lines: detected lines (optional)
    """
    ax.imshow(image)

    if corners is not None:
        # Draw corners
        for i, corner in enumerate(corners):
            ax.plot(corner[0], corner[1], 'ro', markersize=8)
            ax.annotate(f'C{i + 1}', (corner[0], corner[1]),
                        xytext=(10, 10), textcoords='offset points')

        # Draw lines between corners
        corners_wrapped = np.vstack((corners, corners[0]))
        for i in range(len(corners)):
            ax.plot([corners_wrapped[i][0], corners_wrapped[i + 1][0]],
                    [corners_wrapped[i][1], corners_wrapped[i + 1][1]],
                    color='red', linewidth=2)

    # Draw Hough lines if available
    if cartesian_lines is not None:
        for line in cartesian_lines:
            x1, y1, x2, y2 = line
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.5)

    ax.axis('off')

@app.post("/analyze/visualize")
async def analyze_image_with_visualization(
        image: UploadFile = File(...),
        text_prompt: str = Form(...),
        box_threshold: float = Form(0.3),
        text_threshold: float = Form(0.25)
) -> FileResponse:
    try:
        # 입력 이미지 저장
        image_path = os.path.join(OUTPUT_DIR, "input_image.jpg")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 이미지 로드 및 처리
        image_pil, image_tensor = load_image(image_path)

        # Grounding DINO 출력
        boxes_filt, pred_phrases = get_grounding_output(
            grounding_dino_model,
            image_tensor,
            text_prompt,
            box_threshold,
            text_threshold,
            device=DEVICE
        )

        # SAM 처리
        image_array = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image_array)

        # 박스 변환
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_array.shape[:2]).to(DEVICE)

        # SAM을 사용하여 마스크 생성
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # 전체 시각화를 위한 subplot 설정
        num_masks = len(masks)
        num_stages = 5
        fig = plt.figure(figsize=(20, 4 * num_masks))
        colors = plt.cm.rainbow(np.linspace(0, 1, num_masks))

        for mask_idx, (mask, box, phrase, color) in enumerate(zip(masks, boxes_filt, pred_phrases, colors)):
            mask_np = mask.cpu().numpy().squeeze()

            # 1. 원본 마스크
            ax1 = plt.subplot(num_masks, num_stages, mask_idx * num_stages + 1)
            ax1.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            ax1.set_title(f'Original Mask {mask_idx + 1}')
            ax1.axis('off')

            # 2. 바이너리 마스크
            mask_binary = (mask_np > 0.1).astype(np.uint8) * 255

            # 패딩 및 모폴로지 연산 적용
            min_side = min(mask_binary.shape[0], mask_binary.shape[1])
            kernel_size = max(3, int(min_side * 0.05))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            padding_size = kernel_size * 2

            padded_mask = np.pad(mask_binary, ((padding_size, padding_size), (padding_size, padding_size)),
                                 mode='constant', constant_values=0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            padded_mask = cv2.morphologyEx(padded_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            mask_binary = padded_mask[padding_size:-padding_size, padding_size:-padding_size]

            ax2 = plt.subplot(num_masks, num_stages, mask_idx * num_stages + 2)
            ax2.imshow(mask_binary, cmap='gray', vmin=0, vmax=255)
            ax2.set_title(f'Binary Mask {mask_idx + 1}')
            ax2.axis('off')

            # 3. 엣지
            edges = cv2.Canny(mask_binary, 50, 150, apertureSize=3)
            ax3 = plt.subplot(num_masks, num_stages, mask_idx * num_stages + 3)
            ax3.imshow(edges, cmap='gray')
            ax3.set_title(f'Edges {mask_idx + 1}')
            ax3.axis('off')

            # 4. 코너 검출 결과 (with fallback)
            corners, cartesian_lines, _ = find_board_corners_with_fallback(mask_binary)
            ax4 = plt.subplot(num_masks, num_stages, mask_idx * num_stages + 4)

            # Create a blank image for lines
            line_img = np.zeros_like(mask_binary)
            if cartesian_lines is not None:
                for line in cartesian_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)

            ax4.imshow(line_img, cmap='gray')
            if corners is not None:
                for corner in corners:
                    ax4.plot(corner[0], corner[1], 'ro', markersize=8)

            detection_method = "Hough Transform" if cartesian_lines is not None else "GoodFeaturesToTrack"
            ax4.set_title(f'Corners ({detection_method}) {mask_idx + 1}')
            ax4.axis('off')

            # 5. 최종 결과
            ax5 = plt.subplot(num_masks, num_stages, mask_idx * num_stages + 5)
            visualize_corners_on_image(ax5, image_array, corners, cartesian_lines)

            confidence = float(phrase.split('(')[-1].strip(')'))
            ax5.set_title(f'Final Result {mask_idx + 1}\nConfidence: {confidence:.3f}')

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, "visualization_output.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
        plt.close()

        return FileResponse(
            output_path,
            media_type="image/png",
            filename="visualization_output.png"
        )

    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )


@app.post("/analyze/masks")
async def analyze_image_masks(
        image: UploadFile = File(...),
        text_prompt: str = Form(...),
        box_threshold: float = Form(0.3),
        text_threshold: float = Form(0.25)
) -> Dict:
    try:
        # ... (previous code remains the same until mask generation)

        # Process each mask
        results = []
        for idx, (mask, box, phrase) in enumerate(zip(masks, boxes_filt, pred_phrases)):
            print(f"\nProcessing mask {idx + 1}")
            mask_np = mask.cpu().numpy().squeeze()

            # Convert to binary mask
            mask_binary = (mask_np > 0.1).astype(np.uint8) * 255

            # Find corners using Hough Transform with goodFeaturesToTrack fallback
            corners, _, _ = find_board_corners_with_fallback(mask_binary)

            if corners is not None:
                # Extract confidence score from phrase
                confidence = float(phrase.split('(')[-1].strip(')'))
                label = phrase.split('(')[0].strip()

                results.append({
                    "id": idx,
                    "label": label,
                    "confidence": confidence,
                    "bbox": box.tolist(),
                    "corners": corners.tolist(),
                    "detection_method": "hough_transform" if _ is not None else "good_features"
                })

        return JSONResponse(content={
            "success": True,
            "objects": results,
            "image_size": {"width": W, "height": H}
        })

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)