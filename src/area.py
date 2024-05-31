import argparse
import io
import subprocess

import numpy as np
import pymupdf
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import label2rgb, rgb2gray
from skimage.feature import canny
from skimage.measure import label, regionprops

BBox = tuple[float, float, float, float]

PANEL_PAGE_AREA_THRESHOLD_PERCENT = 2


def check_bbox_overlap(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def merge_bbox(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def get_panel_bbox(page: Image.Image, width: int, height: int) -> list[BBox]:
    grayscale = rgb2gray(page)
    edges = canny(grayscale)
    # Image.fromarray(edges).save("canny.png")
    segmentation = ndi.binary_fill_holes(edges)
    # Image.fromarray(segmentation).save("segmentation.png")

    labels = label(segmentation)
    # Image.fromarray(np.uint8(label2rgb(labels, bg_label=0) * 255)).save("labels.png")

    regions = regionprops(labels)
    panels = []

    for region in regions:
        for i, panel in enumerate(panels):
            if check_bbox_overlap(region.bbox, panel):
                panels[i] = merge_bbox(panel, region.bbox)
                break
        else:
            panels.append(region.bbox)

    for i, bbox in reversed(list(enumerate(panels))):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        print(f"{area}, {100 * area / (width * height):.2f}%")
        if area < (PANEL_PAGE_AREA_THRESHOLD_PERCENT / 100) * width * height:
            del panels[i]

    return panels


def main():
    parser = argparse.ArgumentParser(description="Calculate text percent of a comic")
    parser.add_argument("input_file", help="Path to input PDF file")
    parser.add_argument("output_dir", help="Path to output directory")
    args = parser.parse_args()

    doc = pymupdf.open(args.input_file)

    for page_index in range(len(doc)):
        page = doc[page_index]
        text_blocks = page.get_text("blocks")  # pyright: ignore
        pix = page.get_pixmap()  # pyright: ignore

        img_buffer = io.BytesIO()
        pix.pil_save(img_buffer, format="PNG")
        img_buffer.seek(0)
        img = Image.open(img_buffer)

        panels = get_panel_bbox(img, pix.width, pix.height)

        panel_img = np.zeros((pix.height, pix.width))
        for i, bbox in enumerate(panels, start=1):
            panel_img[bbox[0] : bbox[2], bbox[1] : bbox[3]] = i

        path = args.output_dir + f"/panels{page_index}.png"
        Image.fromarray((label2rgb(panel_img, bg_label=0) * 255).astype(np.uint8)).save(
            path
        )

    subprocess.run(f"open {args.input_file} {args.output_dir}/*", shell=True)
