import argparse

from pypdf import PdfReader, PdfWriter


def main():
    parser = argparse.ArgumentParser(description="Extract pages from PDF")
    parser.add_argument("input_file", help="Path to input PDF file")
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument(
        "pages", type=int, nargs="+", help="Page numbers (at least one)"
    )
    args = parser.parse_args()

    reader = PdfReader(args.input_file)
    writer = PdfWriter()

    path = args.output_dir + "/"

    for page_num in args.pages:
        writer.add_page(reader.pages[page_num])
        path += f"{page_num}_"

    path = path[:-1] + ".pdf"

    with open(path, "wb") as f:
        writer.write(f)
