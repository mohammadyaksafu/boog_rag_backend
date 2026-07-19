"""
PDF structure extraction.

- Uses PyMuPDF (fitz) to walk the document page by page, pulling out text
  blocks in reading order (this is the "PDF structure model" part - it
  respects block/line layout rather than doing a naive whole-page text dump).
- For pages that are mostly images (e.g. scanned pages, charts, diagrams)
  and have little/no extractable text, we fall back to a local VLM
  (image-captioning model, run fully offline/locally via transformers) to
  produce a text description of the page so it still becomes searchable.
- Every chunk keeps track of its page number so answers can always cite
  the exact page(s) they came from.
"""
import io
from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF
from PIL import Image

from app.config import settings

_vlm_processor = None
_vlm_model = None


def _load_vlm():
    """Lazy-load the local VLM only when we actually need it (image-heavy page)."""
    global _vlm_processor, _vlm_model
    if _vlm_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _vlm_processor = BlipProcessor.from_pretrained(settings.vlm_model)
        _vlm_model = BlipForConditionalGeneration.from_pretrained(settings.vlm_model).to(settings.device)
    return _vlm_processor, _vlm_model


def _caption_page_image(pix: "fitz.Pixmap") -> str:
    processor, model = _load_vlm()
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(settings.device)
    out = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(out[0], skip_special_tokens=True)


@dataclass
class PageChunk:
    page: int  # 1-indexed page number, matches what a reader sees
    text: str


def extract_pdf_structure(pdf_path: str, min_text_chars_for_vlm_fallback: int = 40) -> List[PageChunk]:
    doc = fitz.open(pdf_path)
    page_texts: List[PageChunk] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1  # human-facing page number

        # Reading-order text extraction respecting block layout
        blocks = page.get_text("blocks")
        blocks_sorted = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
        text = "\n".join(b[4].strip() for b in blocks_sorted if b[4].strip())

        if len(text) < min_text_chars_for_vlm_fallback:
            # Likely a scanned/image-heavy page -> describe it with the local VLM
            try:
                pix = page.get_pixmap(dpi=150)
                caption = _caption_page_image(pix)
                text = (text + "\n" + f"[Page image description: {caption}]").strip()
            except Exception as e:
                text = text or f"[Page {page_number} could not be parsed: {e}]"

        page_texts.append(PageChunk(page=page_number, text=text))

    doc.close()
    return page_texts


def chunk_pages(pages: List[PageChunk], chunk_size: int = 800, overlap: int = 150) -> List[PageChunk]:
    """
    Split long pages into overlapping chunks while always keeping the
    originating page number attached, so retrieval can cite it precisely.
    Short pages become a single chunk.
    """
    chunks: List[PageChunk] = []
    for pc in pages:
        text = pc.text
        if len(text) <= chunk_size:
            chunks.append(PageChunk(page=pc.page, text=text))
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(PageChunk(page=pc.page, text=text[start:end]))
            if end == len(text):
                break
            start = end - overlap
    return chunks
