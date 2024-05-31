import argparse
import io
import subprocess

import numpy as np
import pymupdf
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.color import label2rgb, rgb2gray
from skimage.feature import canny
from skimage.measure import label, regionprops

BBox = tuple[float, float, float, float]

PANEL_PAGE_AREA_THRESHOLD_PERCENT = 2


def bbox_area(bbox: BBox) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def do_bboxes_overlap(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def merge_bboxes(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def get_panel_bboxes(page: Image.Image, width: int, height: int) -> list[BBox]:
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
        area = bbox_area(bbox)
        # print(f"{area}, {100 * area / (width * height):.2f}%")
        if area < (PANEL_PAGE_AREA_THRESHOLD_PERCENT / 100) * width * height:
            del panels[i]

    return panels


def expand_panel_bboxes(panels: list[BBox]) -> list[BBox]:
    """Expands panel bounding boxes to fill gaps in layout"""

    layout_left = min(bbox[0] for bbox in panels)
    layout_top = min(bbox[1] for bbox in panels)
    layout_right = max(bbox[2] for bbox in panels)
    layout_bottom = max(bbox[3] for bbox in panels)

    expanded = []

    # make wider
    for i, bbox in enumerate(panels):
        left, top, right, bottom = bbox
        lb, tb, rb, bb = left, None, right, None

        for j, other_bbox in enumerate(panels):
            if i != j:
                if other_bbox[3] < top:
                    if tb is None or other_bbox[3] > tb:
                        tb = other_bbox[3]
                        adj_left_bb = other_bbox
                if other_bbox[1] > bottom:
                    if bb is None or other_bbox[1] < bb:
                        bb = other_bbox[1]
                        adj_right_bb = other_bbox

        if tb is None:
            tb = layout_top
        if bb is None:
            bb = layout_bottom

        expanded.append((lb, tb, rb, bb))

    final = []

    # make taller
    for i, bbox in enumerate(expanded):
        left, top, right, bottom = bbox
        adj_left_bb = None
        adj_right_bb = None

        for j, other_bbox in enumerate(expanded):
            if i != j:
                # make sure we're on the same horizontal strip
                if abs(other_bbox[0] - left) < 10 or abs(other_bbox[2] - right) < 10:
                    # 5 = tolerance because we might slightly overlap
                    if other_bbox[3] < (top + 5):
                        if adj_left_bb is None or other_bbox[3] > adj_left_bb[3]:
                            adj_left_bb = other_bbox
                    if other_bbox[1] > (bottom - 5):
                        if adj_right_bb is None or other_bbox[1] < adj_right_bb[3]:
                            adj_right_bb = other_bbox

        # expand to top and bottom of adjacent panels
        if adj_left_bb is not None and adj_right_bb is not None:
            left = min(left, adj_left_bb[0], adj_right_bb[0])
            right = max(right, adj_left_bb[2], adj_right_bb[2])
        elif adj_left_bb is not None:
            left = min(left, adj_left_bb[0])
            right = max(right, adj_left_bb[2])
        elif adj_right_bb is not None:
            left = min(left, adj_right_bb[0])
            right = max(right, adj_right_bb[2])

        final.append((left, top, right, bottom))

    return final


def render_panels(panels: list[BBox], width: int, height: int, path: str) -> None:
    panel_img = np.zeros((height, width))
    for i, bbox in enumerate(panels, start=1):
        panel_img[bbox[0] : bbox[2], bbox[1] : bbox[3]] = i

    Image.fromarray((label2rgb(panel_img, bg_label=0) * 255).astype(np.uint8)).save(
        path
    )


def render_panels_on_page(panels: list[BBox], page: Image.Image, path: str) -> None:
    draw = ImageDraw.Draw(page)
    for bbox in panels:
        x0, y0, x1, y1 = bbox[1], bbox[0], bbox[3], bbox[2]
        draw.rectangle([x0, y0, x1, y1], outline="red")

    page.save(path)


def main():
    parser = argparse.ArgumentParser(description="Calculate text percent of a comic")
    parser.add_argument("input_file", help="Path to input PDF file")
    parser.add_argument("output_dir", help="Path to output directory")
    args = parser.parse_args()

    doc = pymupdf.open(args.input_file)

    for page_index in range(len(doc)):
        print(f"Processing page {page_index}")
        page = doc[page_index]
        text_blocks = page.get_text("blocks")  # pyright: ignore
        pix = page.get_pixmap()  # pyright: ignore

        img_buffer = io.BytesIO()
        pix.pil_save(img_buffer, format="PNG")
        img_buffer.seek(0)
        img = Image.open(img_buffer)

        panels = get_panel_bboxes(img, pix.width, pix.height)
        panels = expand_panel_bboxes(panels)

        render_panels(
            panels,
            pix.width,
            pix.height,
            args.output_dir + f"/panels{page_index}.png",
        )

        render_panels_on_page(panels, img, args.output_dir + f"/page{page_index}.png")

    subprocess.run(f"open {args.output_dir}/*", shell=True)
