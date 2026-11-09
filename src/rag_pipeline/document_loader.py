import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import json
import pdfplumber
from bs4 import BeautifulSoup
import frontmatter


@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]


@dataclass
class Document:
    doc_id: str
    source_path: str
    raw_content: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    clean_text: str = ""  # normalized plaintext for chunking (no frontmatter/HTML)


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: Path) -> Document:
        pass

    def _generate_doc_id(self, file_path: Path) -> str:
        return hashlib.md5(str(file_path.absolute()).encode()).hexdigest()[:12]

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        stat = file_path.stat()
        return {
            "source_file": str(file_path),
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "file_size_bytes": stat.st_size,
            "last_modified": stat.st_mtime,
        }


class TextLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        chunks = self._split_by_sections(raw_content)
        doc_id = self._generate_doc_id(file_path)
        metadata = self._extract_metadata(file_path)

        return Document(
            doc_id=doc_id,
            source_path=str(file_path),
            raw_content=raw_content,
            chunks=chunks,
            metadata=metadata,
            clean_text=raw_content,
        )

    def _split_by_sections(self, content: str) -> List[DocumentChunk]:
        chunks = []
        sections = re.split(r'\n\s*\n', content)
        for i, section in enumerate(sections):
            section = section.strip()
            if section:
                chunks.append(DocumentChunk(
                    content=section,
                    metadata={
                        "chunk_index": i,
                        "section_heading": self._extract_heading(section),
                        "char_count": len(section),
                    }
                ))
        return chunks

    def _extract_heading(self, text: str) -> Optional[str]:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                return line.lstrip('#').strip()
        return None


class MarkdownLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        post = frontmatter.loads(raw_content)
        content = post.content
        frontmatter_metadata = post.metadata

        chunks = self._split_by_headings(content)
        doc_id = self._generate_doc_id(file_path)
        metadata = self._extract_metadata(file_path)
        metadata.update(frontmatter_metadata)

        return Document(
            doc_id=doc_id,
            source_path=str(file_path),
            raw_content=raw_content,
            chunks=chunks,
            metadata=metadata,
            clean_text=content,  # excludes YAML frontmatter
        )

    def _split_by_headings(self, content: str) -> List[DocumentChunk]:
        chunks = []
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')

        current_chunk = []
        current_heading = None
        heading_level = 0
        chunk_index = 0

        for line in lines:
            match = re.match(heading_pattern, line)
            if match:
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk).strip()
                    if chunk_content:
                        chunks.append(DocumentChunk(
                            content=chunk_content,
                            metadata={
                                "chunk_index": chunk_index,
                                "section_heading": current_heading,
                                "heading_level": heading_level,
                                "char_count": len(chunk_content),
                            }
                        ))
                        chunk_index += 1
                current_heading = match.group(2).strip()
                heading_level = len(match.group(1))
                current_chunk = [line]
            else:
                current_chunk.append(line)

        if current_chunk:
            chunk_content = '\n'.join(current_chunk).strip()
            if chunk_content:
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    metadata={
                        "chunk_index": chunk_index,
                        "section_heading": current_heading,
                        "heading_level": heading_level,
                        "char_count": len(chunk_content),
                    }
                ))

        return chunks


class HTMLLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        soup = BeautifulSoup(raw_content, 'html.parser')
        text_content = self._extract_text(soup)
        chunks = self._split_by_sections(text_content, soup)
        doc_id = self._generate_doc_id(file_path)
        metadata = self._extract_metadata(file_path)
        metadata["title"] = soup.title.string if soup.title else None

        return Document(
            doc_id=doc_id,
            source_path=str(file_path),
            raw_content=raw_content,
            chunks=chunks,
            metadata=metadata,
            clean_text=text_content,
        )

    def _extract_text(self, soup: BeautifulSoup) -> str:
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return re.sub(r'\n{3,}', '\n\n', text)

    def _split_by_sections(self, text: str, soup: BeautifulSoup) -> List[DocumentChunk]:
        chunks = []
        sections = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'section', 'article'])
        current_section = None
        current_content = []
        chunk_index = 0

        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section', 'article']):
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current_content:
                    chunk_text = '\n'.join(current_content).strip()
                    if chunk_text:
                        chunks.append(DocumentChunk(
                            content=chunk_text,
                            metadata={
                                "chunk_index": chunk_index,
                                "section_heading": current_section,
                                "heading_level": int(element.name[1]) if element.name[1].isdigit() else 0,
                                "char_count": len(chunk_text),
                            }
                        ))
                        chunk_index += 1
                current_section = element.get_text(strip=True)
                current_content = [current_section]
            else:
                text = element.get_text(strip=True)
                if text:
                    current_content.append(text)

        if current_content:
            chunk_text = '\n'.join(current_content).strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "chunk_index": chunk_index,
                        "section_heading": current_section,
                        "heading_level": 0,
                        "char_count": len(chunk_text),
                    }
                ))

        return chunks


class PDFLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        raw_content = ""
        page_texts = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    raw_content += f"\n--- Page {page_num} ---\n{text}"
                    page_texts.append((page_num, text))

        chunks = self._split_by_pages(page_texts)
        doc_id = self._generate_doc_id(file_path)
        metadata = self._extract_metadata(file_path)
        metadata["page_count"] = len(page_texts)

        clean_text = "\n".join(t for _, t in page_texts)

        return Document(
            doc_id=doc_id,
            source_path=str(file_path),
            raw_content=raw_content,
            chunks=chunks,
            metadata=metadata,
            clean_text=clean_text,
        )

    def _split_by_pages(self, page_texts: List[tuple]) -> List[DocumentChunk]:
        chunks = []
        for page_num, text in page_texts:
            sections = re.split(r'\n\s*\n', text)
            for section in sections:
                section = section.strip()
                if section:
                    chunks.append(DocumentChunk(
                        content=section,
                        metadata={
                            "chunk_index": len(chunks),
                            "page_number": page_num,
                            "section_heading": self._extract_heading(section),
                            "char_count": len(section),
                        }
                    ))
        return chunks

    def _extract_heading(self, text: str) -> Optional[str]:
        lines = text.split('\n')
        for line in lines[:3]:
            line = line.strip()
            if line and (line.isupper() or re.match(r'^\d+\.\s+\w+', line)):
                return line
        return None


class DocumentLoader:
    EXTENSION_MAP = {
        '.txt': TextLoader,
        '.md': MarkdownLoader,
        '.markdown': MarkdownLoader,
        '.html': HTMLLoader,
        '.htm': HTMLLoader,
        '.pdf': PDFLoader,
    }

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_file(self, file_path: str) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在： {file_path}")

        ext = path.suffix.lower()
        if ext not in self.EXTENSION_MAP:
            raise ValueError(f"不支持的文件格式： {ext}")

        loader = self.EXTENSION_MAP[ext]()
        return loader.load(path)

    def load_directory(self, directory: str) -> List[Document]:
        dir_path = Path(directory)
        documents = []
        for ext in self.EXTENSION_MAP:
            for file_path in dir_path.rglob(f"*{ext}"):
                try:
                    doc = self.load_file(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"加载 {file_path} 失败： {e}")
        return documents

    def save_raw(self, document: Document) -> str:
        raw_path = self.raw_dir / f"{document.doc_id}{Path(document.source_path).suffix}"
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(document.raw_content)
        return str(raw_path)

    def save_processed(self, document: Document) -> str:
        processed_path = self.processed_dir / f"{document.doc_id}.json"
        data = {
            "doc_id": document.doc_id,
            "source_path": document.source_path,
            "metadata": document.metadata,
            "clean_text": document.clean_text,
            "chunks": [
                {"content": chunk.content, "metadata": chunk.metadata}
                for chunk in document.chunks
            ]
        }
        with open(processed_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(processed_path)

    def load_processed(self, doc_id: str) -> Optional[Document]:
        processed_path = self.processed_dir / f"{doc_id}.json"
        if not processed_path.exists():
            return None

        with open(processed_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = [
            DocumentChunk(content=c["content"], metadata=c["metadata"])
            for c in data["chunks"]
        ]
        return Document(
            doc_id=data["doc_id"],
            source_path=data["source_path"],
            raw_content="",
            chunks=chunks,
            metadata=data["metadata"],
            clean_text=data.get("clean_text", ""),
        )

    def process_and_save(self, file_path: str) -> Document:
        document = self.load_file(file_path)
        self.save_raw(document)
        self.save_processed(document)
        return document

    def process_directory(self, directory: str) -> List[Document]:
        documents = self.load_directory(directory)
        for doc in documents:
            self.save_raw(doc)
            self.save_processed(doc)
        return documents


if __name__ == "__main__":
    loader = DocumentLoader()
    print("文档加载器已初始化，可以开始处理文件。")
    print(f"原始文档目录： {loader.raw_dir}")
    print(f"处理结果目录： {loader.processed_dir}")
    print(f"支持的格式： {list(DocumentLoader.EXTENSION_MAP.keys())}")