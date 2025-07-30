import os
import uuid
import logging
from typing import List, Generator
from pathlib import Path
import gc

import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        base_dir = Path(__file__).resolve().parent
        self.client = chromadb.PersistentClient(path=str(base_dir / "chroma_db"))

    def extract_text_from_pdf_streaming(self, pdf_path: str, max_pages_per_batch: int = 10) -> Generator[str, None, None]:
        """Extracts text from PDF in batches to reduce memory usage."""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                logger.info(f"📊 Total pages in {os.path.basename(pdf_path)}: {total_pages}")
                
                for i in range(0, total_pages, max_pages_per_batch):
                    batch_text = ""
                    end_page = min(i + max_pages_per_batch, total_pages)
                    
                    for page_num in range(i, end_page):
                        try:
                            page_text = reader.pages[page_num].extract_text() or ""
                            batch_text += page_text + "\n"
                        except Exception as e:
                            logger.warning(f"⚠️ Error extracting page {page_num}: {e}")
                            continue
                    
                    if batch_text.strip():
                        yield batch_text.strip()
                    
                    # Force garbage collection after each batch
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")

    def chunk_text_generator(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> Generator[str, None, None]:
        """Splits text into overlapping chunks using a generator to save memory."""
        if not text or len(text) == 0:
            return
        
        # Clean up the text first
        text = ' '.join(text.split())  # Remove extra whitespace
        text_length = len(text)
        
        if text_length <= chunk_size:
            # If text is smaller than chunk size, return it as one chunk
            if text.strip() and len(text.strip()) > 50:  # Only meaningful chunks
                yield text.strip()
            return
        
        start = 0
        chunk_count = 0
        max_chunks = 100  # Limit chunks per document to prevent explosion
        
        while start < text_length and chunk_count < max_chunks:
            end = min(text_length, start + chunk_size)
            
            # Try to break at sentence boundaries
            if end < text_length:
                # Look for sentence endings within the last 200 chars
                sentence_break = text.rfind('.', end - 200, end)
                if sentence_break > start + chunk_size // 2:  # Don't make chunks too small
                    end = sentence_break + 1
            
            chunk = text[start:end].strip()
            
            if chunk and len(chunk) > 50:  # Only yield meaningful chunks
                yield chunk
                chunk_count += 1
            
            # Move start position with overlap
            if end >= text_length:
                break
            start = max(start + chunk_size - overlap, end - overlap)
            
            # Prevent infinite loops
            if start >= end:
                break

    def process_single_pdf_streaming(self, file_path: Path, collection, level: str):
        """Process a single PDF file with streaming to minimize memory usage."""
        logger.info(f"📄 Processing: {file_path.name}")
        
        total_chunks = 0
        batch_chunks = []
        batch_metadatas = []
        batch_ids = []
        batch_size = 32  # Increased batch size for efficiency
        
        # Process PDF in page batches
        for batch_num, text_batch in enumerate(self.extract_text_from_pdf_streaming(str(file_path), max_pages_per_batch=20)):
            logger.info(f"🔄 Processing text batch {batch_num + 1} from {file_path.name} ({len(text_batch)} chars)")
            
            # Generate chunks from this text batch
            chunk_count_from_batch = 0
            for chunk_idx, chunk in enumerate(self.chunk_text_generator(text_batch)):
                batch_chunks.append(chunk)
                batch_metadatas.append({
                    "source": file_path.name,
                    "chunk_index": total_chunks,
                    "cmmc_level": level,
                    "batch_num": batch_num,
                    "char_count": len(chunk)
                })
                batch_ids.append(f"{file_path.stem}_{total_chunks}_{uuid.uuid4().hex[:8]}")
                total_chunks += 1
                chunk_count_from_batch += 1
                
                # Process batch when it reaches the desired size
                if len(batch_chunks) >= batch_size:
                    self._add_batch_to_collection(collection, batch_chunks, batch_metadatas, batch_ids)
                    
                    # Clear batch arrays and force garbage collection
                    batch_chunks.clear()
                    batch_metadatas.clear()
                    batch_ids.clear()
                    gc.collect()
            
            logger.info(f"📊 Generated {chunk_count_from_batch} chunks from batch {batch_num + 1}")
            
            # Force garbage collection after each text batch
            del text_batch
            gc.collect()
        
        # Process any remaining chunks
        if batch_chunks:
            self._add_batch_to_collection(collection, batch_chunks, batch_metadatas, batch_ids)
        
        logger.info(f"✅ Completed {file_path.name}: {total_chunks} total chunks processed")

    def _add_batch_to_collection(self, collection, batch_chunks: List[str], batch_metadatas: List[dict], batch_ids: List[str]):
        """Add a batch of chunks to the collection with error handling."""
        if not batch_chunks:
            return
            
        try:
            logger.info(f"🔄 Generating embeddings for {len(batch_chunks)} chunks...")
            # Generate embeddings with smaller batch size
            embeddings = self.model.encode(
                batch_chunks, 
                show_progress_bar=True,  # Show progress for user feedback
                batch_size=16
            ).tolist()
            
            logger.info(f"💾 Adding {len(batch_chunks)} chunks to collection...")
            collection.add(
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=embeddings
            )
            
            logger.info(f"✅ Successfully added batch of {len(batch_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Error adding batch to collection: {str(e)}")
            # Try with even smaller batches if there's still a memory error
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.info("🔄 Retrying with smaller batches...")
                self._add_batch_fallback(collection, batch_chunks, batch_metadatas, batch_ids)
            else:
                logger.error(f"❌ Skipping batch due to error: {str(e)}")

    def _add_batch_fallback(self, collection, batch_chunks: List[str], batch_metadatas: List[dict], batch_ids: List[str]):
        """Fallback method to add chunks one by one if batch processing fails."""
        for chunk, metadata, chunk_id in zip(batch_chunks, batch_metadatas, batch_ids):
            try:
                embedding = self.model.encode([chunk], show_progress_bar=False).tolist()
                collection.add(
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[chunk_id],
                    embeddings=embedding
                )
            except Exception as e:
                logger.error(f"❌ Failed to add individual chunk {chunk_id}: {str(e)}")

    def process_documents(self, folder: str):
        """Reads, processes, and stores PDF chunks into ChromaDB with memory optimization."""
        collection_name = "cmmc_documents"

        try:
            self.client.delete_collection(collection_name)
            logger.info(f"🗑️ Deleted old collection: {collection_name}")
        except:
            pass

        collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "CMMC Level 1-3 documents"}
        )

        pdf_files = ["CMMC_Level1.pdf", "CMMC_Level2.pdf", "CMMC_Level3.pdf"]
        folder_path = Path(folder).resolve()

        for pdf_file in pdf_files:
            file_path = folder_path / pdf_file
            if not file_path.is_file():
                logger.warning(f"⚠️ Missing file: {pdf_file}")
                continue

            level = pdf_file.replace("CMMC_Level", "").replace(".pdf", "")
            
            try:
                self.process_single_pdf_streaming(file_path, collection, level)
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file}: {str(e)}")
                continue
            
            # Force garbage collection between files
            gc.collect()

        final_count = collection.count()
        logger.info(f"📊 Final collection contains {final_count} documents")
        
        if final_count > 0:
            logger.info("🎉 Document processing completed successfully!")
        else:
            logger.warning("⚠️ No documents were added to the collection.")

def main():
    folder = os.path.join(os.path.dirname(__file__), "documents")
    if not os.path.exists(folder):
        os.makedirs(folder)
        logger.info(f"📁 Created folder: {folder}")
        logger.info("Please add the following files:")
        logger.info("- CMMC_Level1.pdf")
        logger.info("- CMMC_Level2.pdf")
        logger.info("- CMMC_Level3.pdf")
        return

    processor = DocumentProcessor()
    processor.process_documents(folder)

if __name__ == "__main__":
    main()