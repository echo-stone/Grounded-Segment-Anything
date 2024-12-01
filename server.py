from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
import os
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import tempfile

# Grounding DINO imports
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything imports
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI()

# Configuration
# CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# GROUNDED_CHECKPOINT = "groundingdino_swint_ogc.pth"
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
GROUNDED_CHECKPOINT = "groundingdino_swinb_cogcoor.pth"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    # Load Grounding DINO model
    grounding_dino_model = load_model(CONFIG_PATH, GROUNDED_CHECKPOINT, DEVICE)

    # Load SAM model
    sam_model = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam_model = sam_model.to(DEVICE)
    sam_predictor = SamPredictor(sam_model)


@app.post("/analyze/")
async def analyze_image(
        image: UploadFile = File(...),
        text_prompt: str = Form(...),
        box_threshold: float = Form(0.3),
        text_threshold: float = Form(0.25)
):
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded image
        image_path = Path(temp_dir) / "input_image.jpg"
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Load and process image
        image_pil, image_tensor = load_image(str(image_path))

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
        image_array = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
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
        output_path = Path(temp_dir) / "grounded_sam_output.jpg"
        plt.savefig(
            output_path,
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0
        )
        plt.close()

        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename="grounded_sam_output.jpg"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)