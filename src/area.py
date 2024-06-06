import argparse
import io
import subprocess
import time

import easyocr
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image, ImageDraw
from pymupdf import Matrix, Page
from scipy import ndimage as ndi
from skimage.color import label2rgb, rgb2gray
from skimage.feature import canny
from skimage.measure import label, regionprops

BBox = tuple[float, float, float, float]
Strip = list[BBox]

PANEL_PAGE_AREA_THRESHOLD_PERCENT = 2

reader = easyocr.Reader(["en"])


def calculate_bbox_area(bbox: BBox) -> float:
    """
    Calculate the area of a bounding box.

    Parameters
    ----------
    bbox : BBox
        A bounding box represented as a tuple (top, left, bottom, right).

    Returns
    -------
    float
        The area of the bounding box.
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def do_bboxes_overlap(a: BBox, b: BBox) -> bool:
    """
    Check if two bounding boxes overlap.

    Parameters
    ----------
    a : BBox
        The first bounding box represented as a tuple (top, left, bottom, right).
    b : BBox
        The second bounding box represented as a tuple (top, left, bottom, right).

    Returns
    -------
    bool
        True if the bounding boxes overlap, False otherwise.
    """
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def merge_bboxes(a: BBox, b: BBox) -> BBox:
    """
    Merge two bounding boxes.

    Parameters
    ----------
    a : BBox
        The first bounding box represented as a tuple (top, left, bottom, right).
    b : BBox
        The second bounding box represented as a tuple (top, left, bottom, right).

    Returns
    -------
    BBox
        A bounding box that covers the area of both input bounding boxes.
    """
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def calculate_strip_bbox(strip: Strip) -> BBox:
    """
    Calculate the bounding box of a strip.

    Parameters
    ----------
    strip : Strip
        A list of bounding boxes representing a strip, where each bounding box
        is a tuple (top, left, bottom, right).

    Returns
    -------
    BBox
        A single bounding box that covers all bounding boxes in the strip.
    """
    top = min(bbox[0] for bbox in strip)
    left = min(bbox[1] for bbox in strip)
    bottom = max(bbox[2] for bbox in strip)
    right = max(bbox[3] for bbox in strip)

    return (top, left, bottom, right)


def calculate_covered_area(bboxes: list[BBox]) -> int:
    """
    Calculate the total covered area of a list of bounding boxes that may overlap.

    Parameters
    ----------
    bboxes : list[BBox]
        List of bounding boxes.

    Returns
    -------
    int
        Total covered area.
    """

    if not bboxes:
        return 0

    # Find the overall bounding box to limit the grid size
    min_top = int(min(bbox[0] for bbox in bboxes))
    min_left = int(min(bbox[1] for bbox in bboxes))
    max_bottom = int(max(bbox[2] for bbox in bboxes))
    max_right = int(max(bbox[3] for bbox in bboxes))

    grid_height = max_bottom - min_top
    grid_width = max_right - min_left
    grid = np.zeros((grid_height, grid_width), dtype=bool)

    # Mark the covered areas on the grid
    for bbox in bboxes:
        top, left, bottom, right = map(int, bbox)
        grid[top - min_top : bottom - min_top, left - min_left : right - min_left] = (
            True
        )

    return np.sum(grid)


def get_panel_bboxes(page: Image.Image, width: int, height: int) -> list[BBox]:
    """
    Extract panel bounding boxes with Canny edge detection.

    Parameters
    ----------
    page : Image
        The PIL image from which to extract panel bounding boxes.
    width : int
        The width of the image.
    height : int
        The height of the image.

    Returns
    -------
    list[BBox]
        A list of bounding boxes for each detected panel, where each bounding
        box is represented as a tuple (top, left, bottom, right).
    """

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
        if area < (PANEL_PAGE_AREA_THRESHOLD_PERCENT / 100) * width * height:
            del panels[i]

    return panels


def find_adjacent_panels(
    panels: list[BBox], index: int
) -> tuple[BBox | None, BBox | None]:
    """
    Find left and right neighbors given a panel index.

    Parameters
    ----------
    panels : list[BBox]
        List of panel bounding boxes.
    index : int
        Index of the panel in the panels list.

    Returns
    -------
    tuple[BBox | None, BBox | None]
        Tuple of left and right adjacent panel bounding boxes, or None if not found.
    """

    top, left, bottom, right = panels[index]
    adj_left_bb = None
    adj_right_bb = None

    for i, other_bbox in enumerate(panels):
        if i != index:
            # make sure we're on the same horizontal strip
            if (
                abs(other_bbox[0] - top) < 10
                or abs(other_bbox[2] - bottom) < 10
                or (other_bbox[0] < top and other_bbox[2] > bottom)
            ):
                # 5 = tolerance because we might slightly overlap
                if other_bbox[3] < (left + 5):
                    if adj_left_bb is None or other_bbox[3] > adj_left_bb[3]:
                        adj_left_bb = other_bbox
                if other_bbox[1] > (right - 5):
                    if adj_right_bb is None or other_bbox[1] < adj_right_bb[3]:
                        adj_right_bb = other_bbox

    return adj_left_bb, adj_right_bb


def heighten_panel_bboxes(panels: list[BBox]) -> list[BBox]:
    """
    Vertically expand panel bounding boxes to match adjacent panels.

    Parameters
    ----------
    panels : list[BBox]
        List of panel bounding boxes.

    Returns
    -------
    list[BBox]
        List of expanded panel bounding boxes in the same order.
    """

    expanded = []

    for i, bbox in enumerate(panels):
        top, left, bottom, right = bbox

        adj_left_bb, adj_right_bb = find_adjacent_panels(panels, i)

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

        expanded.append((top, left, bottom, right))

    return expanded


def widen_panel_bboxes(panels: list[BBox], expand_edges: bool = False) -> list[BBox]:
    """
    Horizontally expand panel bounding boxes to fill gaps in layout.

    Parameters
    ----------
    panels : list[BBox]
        List of panel bounding boxes.
    expand_edges: bool, optional
        Whether to allow edge panels to expand toward other panels. Default is False,
        meaning edge panels expand toward layout edges and only middle panels expand
        toward other panels.

    Returns
    -------
    list[BBox]
        List of expanded panel bounding boxes in the same order.
    """

    panels = panels.copy()

    layout_left = min(bbox[1] for bbox in panels)
    layout_right = max(bbox[3] for bbox in panels)

    expanded = []

    for i, bbox in enumerate(panels):
        top, left, bottom, right = bbox

        adj_left_bb, adj_right_bb = find_adjacent_panels(panels, i)

        gap_to_left = 0
        gap_to_right = 0

        # if we're expanding toward an edge, fill all available space
        if adj_left_bb is None:
            gap_to_left = left - layout_left
        if adj_right_bb is None:
            gap_to_right = layout_right - right

        # if we're expanding toward a panel, leave some padding
        if adj_left_bb and adj_right_bb:
            gap_to_left = left - adj_left_bb[3] - 5
            gap_to_right = adj_right_bb[1] - right - 5

        if expand_edges:
            if adj_left_bb and not adj_right_bb:
                gap_to_left = left - adj_left_bb[3] - 5
            if adj_right_bb and not adj_left_bb:
                gap_to_right = adj_right_bb[1] - right - 5

        # only expand if the gap is significant
        if gap_to_left > 10:
            left -= gap_to_left
        if gap_to_right > 10:
            right += gap_to_right

        new_bbox = (top, left, bottom, right)
        # update list so we don't double expand adjacent panels
        panels[i] = new_bbox
        expanded.append(new_bbox)

    return expanded


def segment_panels_by_strip(panels: list[BBox]) -> list[Strip]:
    """
    Segment a list of panel bounding boxes into strips.

    Parameters
    ----------
    panels : list[BBox]
        List of panel bounding boxes.

    Returns
    -------
    list[Strip]
        A list of strips, where each strip is a list of panel bounding boxes. Strips
        will be sorted top to bottom and panels will be sorted in reading order.
    """

    if not panels:
        return []

    panels = panels.copy()
    panels = sorted(panels, key=lambda bbox: (bbox[0], bbox[1]))
    strips = []

    # panels on the same horizontal level are always in the same strip
    cur = []
    last_top = panels[0][0]
    for bbox in panels:
        top = bbox[0]
        if abs(top - last_top) < 20:
            cur.append(bbox)
        else:
            strips.append(cur)
            cur = [bbox]
        last_top = top

    strips.append(cur)

    if len(strips) < 3:
        return strips

    # small vertical gaps indicate a multi-line strip
    strip_bboxes = [calculate_strip_bbox(strip) for strip in strips]
    final_strips = []
    cur = []

    for i, strip in enumerate(strips):
        if i == 0:
            cur += strip
            continue

        top, _, bottom, _ = strip_bboxes[i]
        prev_bottom = strip_bboxes[i - 1][2]
        # this gap is usually less than 10 or over 30
        if top - prev_bottom < 15:
            cur += strip
        else:
            final_strips.append(cur)
            cur = strip

    final_strips.append(cur)

    return final_strips


def find_text_bboxes(page: Page, crop: BBox = (0.0, 0.0, 0.0, 0.0)) -> list[BBox]:
    """
    Find text bounding boxes with OCR.

    Parameters
    ----------
    page : Page
        PyMuPDF page to extract text from.
    crop : BBox, optional
        Bounding box to crop the page to before OCR. Default is (0.0, 0.0, 0.0, 0.0),
        meaning the uncropped page is used.

    Returns
    -------
    list[BBox]
        List of text bounding boxes.
    """

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
    """
    Render the panel bounding boxes in a new image and save it to the specified path.

    Parameters
    ----------
    panels : list[BBox]
        List of panel bounding boxes.
    width : int
        Width of the image.
    height : int
        Height of the image.
    path : str
        Path to save the rendered image.

    Returns
    -------
    None
    """

    panel_img = np.zeros((height, width))
    for i, bbox in enumerate(panels, start=1):
        panel_img[bbox[0] : bbox[2], bbox[1] : bbox[3]] = i

    Image.fromarray((label2rgb(panel_img, bg_label=0) * 255).astype(np.uint8)).save(
        path
    )


def render_strips(page: Image.Image, strips: list[Strip], path: str) -> None:
    """
    Render the strip bounding boxes on the given page image and save it to the specified
    path.

    Parameters
    ----------
    page : Image
        The PIL image on which to render the bounding boxes.
    strips : list[Strip]
        List of lists of bounding boxes.
    path : str
        Path to save the rendered annotated image.

    Returns
    -------
    None
    """

    draw = ImageDraw.Draw(page)

    for strip in strips:
        bbox = calculate_strip_bbox(strip)
        top, left, bottom, right = bbox
        draw.rectangle([left, top, right, bottom], outline="green")

    page.save(path)


def render_annotated_page(
    page: Image.Image, panels: list[BBox], text: list[BBox], path: str
) -> None:
    """
    Render the panel and text bounding boxes on the given page image and save it to the
    specified path.

    Parameters
    ----------
    page : Image
        The PIL image on which to render the bounding boxes.
    panels : list[BBox]
        List of panel bounding boxes.
    text : list[BBox]
        List of text bounding boxes.
    path : str
        Path to save the rendered annotated image.

    Returns
    -------
    None
    """

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

    data = []

    for page_index in range(len(doc)):
        start = time.time()

        page = doc.load_page(page_index)
        pix = page.get_pixmap()  # pyright: ignore

        img_buffer = io.BytesIO()
        pix.pil_save(img_buffer, format="PNG")
        img_buffer.seek(0)
        img = Image.open(img_buffer)

        panels = get_panel_bboxes(img, pix.width, pix.height)
        panels = sorted(panels, key=lambda bbox: (bbox[0], bbox[1]))
        panels = heighten_panel_bboxes(panels)
        panels = widen_panel_bboxes(panels)
        # for the <edge panel>, <negative panel>, <edge panel> case
        panels = widen_panel_bboxes(panels, expand_edges=True)

        strips = segment_panels_by_strip(panels)

        all_text = []

        for strip_index, strip in enumerate(strips):
            strip_data = []
            for panel_index, bbox in enumerate(strip):
                panel_area = calculate_bbox_area(bbox)
                text_bboxes = find_text_bboxes(page, crop=bbox)
                text_area = calculate_covered_area(text_bboxes)

                data.append(
                    {
                        "page": page_index,
                        "strip": strip_index,
                        "panel": panel_index,
                        "panel_area": panel_area,
                        "text_area": text_area,
                    }
                )

                all_text += text_bboxes

        end = time.time()
        print(f"Processed page {page_index} in {end - start:.2f}s")

        render_panels(
            panels,
            pix.width,
            pix.height,
            args.output_dir + f"/page_{page_index}_panels.png",
        )
        render_strips(img, strips, args.output_dir + f"/page_{page_index}_strips.png")
        render_annotated_page(
            img, panels, all_text, args.output_dir + f"/page_{page_index}_annotated.png"
        )

    df = pd.DataFrame(data)

    df.to_csv(f"{args.output_dir}/data.csv", index=False)

    subprocess.run(f"open {args.output_dir}/*.png", shell=True)
