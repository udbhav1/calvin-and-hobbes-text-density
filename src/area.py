import argparse
import io
import subprocess
import time

import easyocr
import numpy as np
import pymupdf
from PIL import Image, ImageDraw
from pymupdf import Matrix, Page
from scipy import ndimage as ndi
from skimage.color import label2rgb, rgb2gray
from skimage.feature import canny
from skimage.measure import label, regionprops

BBox = tuple[float, float, float, float]

PANEL_PAGE_AREA_THRESHOLD_PERCENT = 2

reader = easyocr.Reader(["en"])


def calculate_bbox_area(bbox: BBox) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def do_bboxes_overlap(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def merge_bboxes(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def get_panel_bboxes(page: Image.Image, width: int, height: int) -> list[BBox]:
    """Extracts panel bounding boxes with canny edge detection"""

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
            if do_bboxes_overlap(region.bbox, panel):
                panels[i] = merge_bboxes(panel, region.bbox)
                break
        else:
            panels.append(region.bbox)

    for i, bbox in reversed(list(enumerate(panels))):
        area = calculate_bbox_area(bbox)
        # print(f"{area}, {100 * area / (width * height):.2f}%")
        if area < (PANEL_PAGE_AREA_THRESHOLD_PERCENT / 100) * width * height:
            del panels[i]

    return panels


def expand_panel_bboxes(panels: list[BBox]) -> list[BBox]:
    """Expands panel bounding boxes to fill gaps in layout"""

    layout_top = min(bbox[0] for bbox in panels)
    layout_left = min(bbox[1] for bbox in panels)
    layout_bottom = max(bbox[2] for bbox in panels)
    layout_right = max(bbox[3] for bbox in panels)

    expanded = []

    # make wider
    for i, bbox in enumerate(panels):
        top, left, bottom, right = bbox
        tb, lb, bb, rb = top, None, bottom, None

        for j, other_bbox in enumerate(panels):
            if i != j:
                if other_bbox[3] < left:
                    if lb is None or other_bbox[3] > lb:
                        lb = other_bbox[3]
                if other_bbox[1] > right:
                    if rb is None or other_bbox[1] < rb:
                        rb = other_bbox[1]

        if lb is None:
            lb = layout_left
        if rb is None:
            rb = layout_right

        expanded.append((tb, lb, bb, rb))

    final = []

    # make taller
    for i, bbox in enumerate(expanded):
        top, left, bottom, right = bbox
        adj_left_bb = None
        adj_right_bb = None

        for j, other_bbox in enumerate(expanded):
            if i != j:
                # make sure we're on the same horizontal strip
                if abs(other_bbox[0] - top) < 10 or abs(other_bbox[2] - bottom) < 10:
                    # 5 = tolerance because we might slightly overlap
                    if other_bbox[3] < (left + 5):
                        if adj_left_bb is None or other_bbox[3] > adj_left_bb[3]:
                            adj_left_bb = other_bbox
                    if other_bbox[1] > (right - 5):
                        if adj_right_bb is None or other_bbox[1] < adj_right_bb[3]:
                            adj_right_bb = other_bbox

        # expand to top and bottom of adjacent panels
        if adj_left_bb is not None and adj_right_bb is not None:
            top = min(top, adj_left_bb[0], adj_right_bb[0])
            bottom = max(bottom, adj_left_bb[2], adj_right_bb[2])
        elif adj_left_bb is not None:
            top = min(top, adj_left_bb[0])
            bottom = max(bottom, adj_left_bb[2])
        elif adj_right_bb is not None:
            top = min(top, adj_right_bb[0])
            bottom = max(bottom, adj_right_bb[2])

        final.append((top, left, bottom, right))

    return final


def find_text_bboxes(page: Page, crop: BBox = (0.0, 0.0, 0.0, 0.0)) -> list[BBox]:
    # dont want to import typing but want crop: BBox in signature so sentinel it is
    if crop != (0.0, 0.0, 0.0, 0.0):
        top, left, bottom, right = crop
        page.set_cropbox(pymupdf.Rect(left, top, right, bottom))  # pyright: ignore

    bboxes = []

    scaling = 2
    pix = page.get_pixmap(matrix=Matrix(scaling, 0, 0, scaling, 0, 0))  # pyright: ignore
    img = Image.open(io.BytesIO(pix.tobytes()))

    img = img.convert("L")

    img_np = np.array(img)
    ocr = reader.readtext(img_np)

    crop_x, crop_y = crop[1], crop[0]
    for detection in ocr:
        corners, text, confidence = detection
        x_coords = [point[0] for point in corners]
        y_coords = [point[1] for point in corners]

        top = (min(y_coords) / scaling) + crop_y
        left = (min(x_coords) / scaling) + crop_x
        bottom = (max(y_coords) / scaling) + crop_y
        right = (max(x_coords) / scaling) + crop_x

        bbox = (top, left, bottom, right)
        bboxes.append(bbox)

    return bboxes


def render_panels(panels: list[BBox], width: int, height: int, path: str) -> None:
    panel_img = np.zeros((height, width))
    for i, bbox in enumerate(panels, start=1):
        panel_img[bbox[0] : bbox[2], bbox[1] : bbox[3]] = i

    Image.fromarray((label2rgb(panel_img, bg_label=0) * 255).astype(np.uint8)).save(
        path
    )


def render_annotated_page(
    panels: list[BBox], text: list[BBox], page: Image.Image, path: str
) -> None:
    draw = ImageDraw.Draw(page)

    for bbox in text:
        top, left, bottom, right = bbox
        draw.rectangle([left, top, right, bottom], outline="green")

    for bbox in panels:
        top, left, bottom, right = bbox
        draw.rectangle([left, top, right, bottom], outline="red")

    page.save(path)


def main():
    parser = argparse.ArgumentParser(description="Calculate text percent of a comic")
    parser.add_argument("input_file", help="Path to input PDF file")
    parser.add_argument("output_dir", help="Path to output directory")
    args = parser.parse_args()

    doc = pymupdf.open(args.input_file)

    for page_index in range(len(doc)):
        start = time.time()

        page = doc.load_page(page_index)
        pix = page.get_pixmap()  # pyright: ignore

        img_buffer = io.BytesIO()
        pix.pil_save(img_buffer, format="PNG")
        img_buffer.seek(0)
        img = Image.open(img_buffer)

        panels = get_panel_bboxes(img, pix.width, pix.height)
        panels = expand_panel_bboxes(panels)

        panels = sorted(panels, key=lambda bbox: (bbox[0], bbox[1]))

        text = []

        for bbox in panels:
            panel_area = calculate_bbox_area(bbox)
            text_bboxes = find_text_bboxes(page, crop=bbox)
            if text_bboxes:
                text += text_bboxes

        end = time.time()
        print(f"Processed page {page_index} in {end - start:.2f}s")

        render_panels(
            panels,
            pix.width,
            pix.height,
            args.output_dir + f"/panels{page_index}.png",
        )

        render_annotated_page(
            panels, text, img, args.output_dir + f"/page{page_index}.png"
        )

    subprocess.run(f"open {args.output_dir}/*", shell=True)
