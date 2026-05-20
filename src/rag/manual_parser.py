"""Manual document parser: extracts text chunks and image references from manual files."""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from loguru import logger


@dataclass
class ManualChunk:
    """A chunk of text from a manual with optional image references."""
    chunk_id: str
    manual_name: str
    text: str
    image_ids: list[str] = field(default_factory=list)
    chunk_index: int = 0


@dataclass
class ParsedManual:
    """Parsed result of a single manual file."""
    manual_name: str
    total_chars: int
    total_images: int
    chunks: list[ManualChunk] = field(default_factory=list)


class ManualParser:
    """Parse manual text files into chunks with image ID mapping."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_file(self, file_path: Path) -> ParsedManual:
        """Parse a single manual file into chunks."""
        content = file_path.read_text(encoding="utf-8")
        manual_name = file_path.stem

        # Separate text content from image ID array at the end
        text_content, image_ids = self._split_content_and_images(content)

        # Split text at <PIC> boundaries, then chunk each section
        sections = self._split_at_pics(text_content)
        chunks = self._create_chunks(sections, image_ids, manual_name)

        return ParsedManual(
            manual_name=manual_name,
            total_chars=len(text_content),
            total_images=len(image_ids),
            chunks=chunks,
        )

    def parse_all(self, manual_dir: Path) -> list[ParsedManual]:
        """Parse all manual files in a directory."""
        results = []
        for fp in sorted(manual_dir.glob("*.txt")):
            if fp.stem == "汇总英文手册":
                # Handle the large English manual separately
                results.append(self._parse_english_manual(fp))
            else:
                parsed = self.parse_file(fp)
                results.append(parsed)
                logger.info(
                    f"Parsed {parsed.manual_name}: "
                    f"{parsed.total_chars} chars, "
                    f"{len(parsed.chunks)} chunks, "
                    f"{parsed.total_images} images"
                )
        return results

    def _split_content_and_images(self, content: str) -> tuple[str, list[str]]:
        """Split manual content into text and image ID array.

        Manual format: text content with <PIC> placeholders, followed by
        a JSON array of image IDs at the end like: ["img1", "img2", ...]
        """
        # Find the last JSON array in the content
        last_bracket = content.rfind("[")
        if last_bracket < 0:
            return content, []

        arr_str = content[last_bracket:]
        try:
            # Extract array content
            match = re.match(r'\[([^\]]+)\]', arr_str)
            if match:
                # Parse quoted strings
                image_ids = re.findall(r'"([^"]+)"', match.group(1))
                text = content[:last_bracket].strip()
                return text, image_ids
        except Exception:
            pass

        return content, []

    def _split_at_pics(self, text: str) -> list[dict]:
        """Split text at <PIC> markers into sections.

        Each section contains text and optionally references to an image.
        """
        parts = re.split(r"(<PIC>)", text)
        sections = []
        current_text = ""
        pic_count = 0

        for part in parts:
            if part == "<PIC>":
                if current_text.strip():
                    sections.append({
                        "text": current_text.strip(),
                        "has_pic": False,
                        "pic_index": None,
                    })
                    current_text = ""
                # Mark the next section as having an image
                sections.append({
                    "text": "",
                    "has_pic": True,
                    "pic_index": pic_count,
                })
                pic_count += 1
            else:
                current_text += part

        if current_text.strip():
            sections.append({
                "text": current_text.strip(),
                "has_pic": False,
                "pic_index": None,
            })

        return sections

    def _create_chunks(
        self,
        sections: list[dict],
        image_ids: list[str],
        manual_name: str,
    ) -> list[ManualChunk]:
        """Create text chunks from sections, preserving image references."""
        chunks = []
        current_chunk_text = ""
        current_images = set()
        chunk_index = 0

        for section in sections:
            if section["has_pic"]:
                # Associate image with current chunk
                pic_idx = section["pic_index"]
                if pic_idx is not None and pic_idx < len(image_ids):
                    current_images.add(image_ids[pic_idx])
                # Add PIC marker to text
                img_id = image_ids[pic_idx] if pic_idx is not None and pic_idx < len(image_ids) else "PIC"
                current_chunk_text += f" [图: {img_id}] "
                continue

            text = section["text"]
            if not text:
                continue

            # If adding this section would exceed chunk size, start a new chunk
            if len(current_chunk_text) + len(text) > self.chunk_size and current_chunk_text:
                chunk = self._make_chunk(
                    manual_name, current_chunk_text, current_images, chunk_index
                )
                chunks.append(chunk)

                # Overlap: keep last portion of previous chunk
                if self.chunk_overlap > 0 and len(current_chunk_text) > self.chunk_overlap:
                    overlap_text = current_chunk_text[-self.chunk_overlap:]
                else:
                    overlap_text = ""

                current_chunk_text = overlap_text + text
                current_images = set()
                chunk_index += 1
            else:
                current_chunk_text += text

        # Don't forget the last chunk
        if current_chunk_text.strip():
            chunk = self._make_chunk(
                manual_name, current_chunk_text, current_images, chunk_index
            )
            chunks.append(chunk)

        return chunks

    def _make_chunk(
        self,
        manual_name: str,
        text: str,
        images: set[str],
        index: int,
    ) -> ManualChunk:
        return ManualChunk(
            chunk_id=f"{manual_name}_{index}",
            manual_name=manual_name,
            text=text.strip(),
            image_ids=list(images),
            chunk_index=index,
        )

    def _parse_english_manual(self, file_path: Path) -> ParsedManual:
        """Parse the large compiled English manual with special handling."""
        content = file_path.read_text(encoding="utf-8")
        manual_name = file_path.stem

        # The English manual has different structure - split by sections (# headers)
        sections = re.split(r"(?=\n# )", content)
        chunks = []
        chunk_index = 0

        for section in sections:
            if not section.strip():
                continue

            # Extract image references from this section (only actual image IDs)
            section_image_ids = re.findall(r'\b([A-Za-z]+_\d+)\b', section)
            seen = set()
            unique_image_ids = []
            for iid in section_image_ids:
                if iid not in seen:
                    seen.add(iid)
                    unique_image_ids.append(iid)

            # Split long sections
            if len(section) > self.chunk_size * 2:
                sub_chunks = self._split_long_section(section)
                for sub in sub_chunks:
                    # Each sub-chunk gets images mentioned within it
                    sub_image_ids = re.findall(r'\b([A-Za-z]+_\d+)\b', sub)
                    sub_seen = set()
                    sub_unique = []
                    for iid in sub_image_ids:
                        if iid not in sub_seen:
                            sub_seen.add(iid)
                            sub_unique.append(iid)
                    chunks.append(ManualChunk(
                        chunk_id=f"{manual_name}_{chunk_index}",
                        manual_name=manual_name,
                        text=sub.strip(),
                        image_ids=sub_unique[:10],  # Cap at 10 images per chunk
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1
            else:
                chunks.append(ManualChunk(
                    chunk_id=f"{manual_name}_{chunk_index}",
                    manual_name=manual_name,
                    text=section.strip(),
                    image_ids=unique_image_ids[:10],
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

        logger.info(
            f"Parsed {manual_name}: "
            f"{len(content)} chars, "
            f"{len(chunks)} chunks"
        )

        return ParsedManual(
            manual_name=manual_name,
            total_chars=len(content),
            total_images=len(set(re.findall(r'\b([A-Za-z]+_\d+)\b', content))),
            chunks=chunks,
        )

    def _split_long_section(self, text: str) -> list[str]:
        """Split a long section into smaller chunks at sentence boundaries."""
        sentences = re.split(r"(?<=[。.!?！？\n])", text)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) > self.chunk_size and current:
                chunks.append(current)
                current = sent
            else:
                current += sent

        if current.strip():
            chunks.append(current)

        return chunks
