"""PDF page rendering with a yellow highlight over the cited passage."""
import pymupdf
import streamlit as st


@st.cache_resource
def get_pdf_document(path: str):
    return pymupdf.open(path)


def render_highlighted_page(page_num: int, bboxes: list, path: str) -> bytes:
    doc = get_pdf_document(path)
    page = doc.load_page(page_num)
    pix_page = page  # annotate a fresh copy each render, then revert
    added = []
    for bbox in bboxes:
        rect = pymupdf.Rect(*bbox)
        annot = pix_page.add_highlight_annot(rect)
        annot.set_colors(stroke=(1, 1, 0))
        annot.update()
        added.append(annot)
    pix = pix_page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6))
    png_bytes = pix.tobytes("png")
    for annot in added:
        pix_page.delete_annot(annot)
    return png_bytes
