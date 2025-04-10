import os
import glob
import time
from tqdm import tqdm
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer

# Configuration
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "document_search"
FILES_DIR = r"C:\Users\neeln\OneDrive\Documents\College\UFlorida\Other Stuff\Job Stuff\AI Internship\MilvusP\text"
BATCH_SIZE = 1000
TEXT_MAX_LENGTH = 65000

# Initialize embedding model
embedding_model = SentenceTransformer('all-mpnet-base-v2')

def create_collection():
    """Create Milvus collection with schema for hybrid search and text match capability"""
    client = MilvusClient(uri=MILVUS_URI)

    # Drop collection if exists
    if COLLECTION_NAME in client.list_collections():
        client.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: {COLLECTION_NAME}")

    # Create schema
    schema = client.create_schema()

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=256)

    # Add text field with text match capability enabled
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=TEXT_MAX_LENGTH,
        enable_analyzer=True,
        analyzer_params={"type": "english"},
        enable_match=True
    )

    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR,
                     dim=768)  # Using 768-dim embeddings from all-mpnet-base-v2
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    index_params = client.prepare_index_params()

    # Dense vector index
    index_params.add_index(
        field_name="dense",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )

    # Sparse vector index for BM25
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25"
    )

    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )

    print(f"Created collection: {COLLECTION_NAME}")
    return client

def split_text_into_chunks(text, max_length=55000):
    """Split text into chunks with some overlap for better semantic search"""
    overlap = 1000  # 1000 char overlap to maintain context
    chunks = []

    if len(text) <= max_length:
        return [text]

    for i in range(0, len(text), max_length - overlap):
        chunk = text[i:i + max_length]
        if len(chunk) > 100:
            chunks.append(chunk)

    return chunks

def get_file_content(file_path):
    """Read file content with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            return content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def insert_files(client, files_dir):
    """Process and insert files in batches, with chunking for large files"""
    file_paths = glob.glob(os.path.join(files_dir, "**"), recursive=True)
    file_paths = [f for f in file_paths if os.path.isfile(f)]

    print(f"Found {len(file_paths)} files to process")

    total_inserted = 0
    batch_num = 1
    chunk_counter = 0

    # Process files in batches
    for i in range(0, len(file_paths), BATCH_SIZE):
        batch_files = file_paths[i:i + BATCH_SIZE]
        all_chunks = []

        print(f"Processing batch {batch_num} ({len(batch_files)} files)...")

        for file_path in tqdm(batch_files):
            content = get_file_content(file_path)
            if not content:
                continue

            # Split large files into chunks
            chunks = split_text_into_chunks(content)
            if len(chunks) > 1:
                print(f"Split {file_path} into {len(chunks)} chunks")

            for idx, chunk in enumerate(chunks):
                filename = os.path.basename(file_path)
                if len(chunks) > 1:
                    chunk_filename = f"{filename}_chunk_{idx + 1}of{len(chunks)}"
                else:
                    chunk_filename = filename

                all_chunks.append({
                    "filename": chunk_filename,
                    "text": chunk
                })
                chunk_counter += 1

        if all_chunks:
            # Process chunks in sub-batches to avoid memory issues
            sub_batch_size = 200

            for j in range(0, len(all_chunks), sub_batch_size):
                sub_batch = all_chunks[j:j + sub_batch_size]

                # Generate embeddings for sub-batch
                texts = [item["text"] for item in sub_batch]
                print(f"Generating embeddings for {len(texts)} chunks...")
                embeddings = embedding_model.encode(texts, show_progress_bar=True)

                # Add embeddings to batch data
                for k, item in enumerate(sub_batch):
                    item["dense"] = embeddings[k].tolist()

                # Insert sub-batch into Milvus
                result = client.insert(COLLECTION_NAME, sub_batch)
                total_inserted += result["insert_count"]
                print(f"Inserted {result['insert_count']} chunks")

        batch_num += 1

    print(f"Total files processed: {len(file_paths)}")
    print(f"Total chunks inserted: {total_inserted} (from {chunk_counter} chunks)")
    return total_inserted


class SearchEngine:
    """Unified search engine for hybrid search and text matching"""

    def __init__(self, uri=MILVUS_URI, collection_name=COLLECTION_NAME):
        """Initialize search engine with Milvus client"""
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name

    def _escape_query(self, query):
        """Escape special characters in query text for TEXT_MATCH"""
        return query.replace("'", "''")

    def hybrid_search(self, query_text, text_match_filter=None, limit=100, dense_weight=0.6, sparse_weight=0.4):
        """
        Perform hybrid search using both dense and sparse vectors with TEXT_MATCH filtering

        Parameters:
        - query_text: The search query
        - text_match_filter: Optional specific text to match exactly
        - limit: Maximum number of results to return
        - dense_weight: Weight for dense vector search (semantic)
        - sparse_weight: Weight for sparse vector search (lexical)

        Returns:
        - List of search results
        """
        # Generate dense embedding for query
        query_embedding = embedding_model.encode([query_text])[0].tolist()

        # For text match searches use higher limit to get more matches
        limit_multiplier = 10 if text_match_filter and not query_text else 3

        ef_value = limit * 3
        print(f"Using ef={ef_value} to retrieve more potential matches")

        # Dense vector search request
        dense_search_params = {
            "data": [query_embedding],
            "anns_field": "dense",
            "param": {
                "metric_type": "COSINE",
                "params": {"ef": ef_value}
            },
            "limit": limit * limit_multiplier
        }

        # Full-text search request
        sparse_search_params = {
            "data": [query_text],
            "anns_field": "sparse",
            "param": {
                "metric_type": "BM25",
                "params": {"drop_ratio_search": 0.2}
            },
            "limit": limit * limit_multiplier
        }

        if text_match_filter:
            escaped_filter = self._escape_query(text_match_filter)

            words = escaped_filter.split()

            if len(words) > 1:
                # Build a combined TEXT_MATCH expression for each word
                word_expressions = []
                for word in words:
                    word_expressions.append(f"TEXT_MATCH(text, '{word}')")

                expr = " && ".join(word_expressions)

                # Add proximity search for better accuracy
                expr = f"({expr}) && TEXT_MATCH(text, '{escaped_filter}')"
            else:
                # Single word query
                expr = f"TEXT_MATCH(text, '{escaped_filter}')"

            print(f"Applying text match filter: {expr}")
            dense_search_params["expr"] = expr
            sparse_search_params["expr"] = expr

        dense_request = AnnSearchRequest(**dense_search_params)
        sparse_request = AnnSearchRequest(**sparse_search_params)

        # Use RRF ranker
        print("Using RRF ranker")
        ranker = RRFRanker()

        # Execute hybrid search
        start_time = time.time()
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            output_fields=["filename", "text"],
            limit=limit * limit_multiplier
        )
        search_time = time.time() - start_time

        print(f"Search completed in {search_time:.4f} seconds")

        processed_results = []
        seen_files = set()

        for hit in results[0]:
            filename = hit['entity']['filename']
            base_filename = filename.split('_chunk_')[0] if '_chunk_' in filename else filename

            # Only add first occurrence of each file
            if base_filename not in seen_files and len(processed_results) < limit:
                seen_files.add(base_filename)
                processed_results.append(hit)

        return processed_results

    def text_match_search(self, term, limit=20):
        """
        Perform a direct text match search using TEXT_MATCH

        Parameters:
        - term: Text term to search for
        - limit: Maximum number of results to return

        Returns:
        - List of matching documents
        """
        escaped_term = self._escape_query(term)

        # multi-word queries
        words = escaped_term.split()

        if len(words) > 1:
            # Build a combined TEXT_MATCH expression for each word
            word_expressions = []
            for word in words:
                word_expressions.append(f"TEXT_MATCH(text, '{word}')")

            expr = " && ".join(word_expressions)

            expr = f"({expr}) && TEXT_MATCH(text, '{escaped_term}')"
        else:
            # Single word query
            expr = f"TEXT_MATCH(text, '{escaped_term}')"

        print(f"Searching with expression: {expr}")

        try:
            # Direct query approach
            results = self.client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["filename", "text"],
                limit=limit
            )
            print(f"Found {len(results)} results using query method")
            return results
        except Exception as e:
            print(f"Query method failed: {e}")

            try:
                # Fall back to search TEXT_MATCH
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[""],
                    anns_field="sparse",
                    param={"metric_type": "BM25"},
                    limit=limit,
                    expr=expr,
                    output_fields=["filename", "text"]
                )
                print(f"Found {len(results[0])} results using search method")
                return results[0]
            except Exception as e:
                print(f"Search method failed too: {e}")
                return []

    def find_all_occurrences(self, results, term):
        """
        Find and extract all occurrences of a term in search results

        Parameters:
        - results: List of search results
        - term: Term to find

        Returns:
        - Dictionary with occurrences by document
        """
        occurrences = {}
        total_count = 0

        term_lower = term.lower()

        for doc in results:
            if isinstance(doc, dict) and "entity" in doc:
                filename = doc["entity"]["filename"]
                text = doc["entity"]["text"]
            else:
                filename = doc["filename"]
                text = doc["text"]

            # Find all occurrences
            term_positions = []
            start_pos = 0
            text_lower = text.lower()

            while True:
                pos = text_lower.find(term_lower, start_pos)
                if pos == -1:
                    break
                term_positions.append(pos)
                start_pos = pos + len(term)

            # Store occurrences
            if term_positions:
                occurrences[filename] = {
                    "count": len(term_positions),
                    "positions": term_positions,
                    "text": text
                }
                total_count += len(term_positions)

        print(f"Found {total_count} total occurrences of '{term}' across {len(occurrences)} documents")
        return occurrences, total_count

    def display_results(self, results, highlight_term=None):
        """Format and display search results with optional term highlighting"""
        if not results:
            print("No results found.")
            return

        for i, hit in enumerate(results):
            print(f"\n--- Result {i + 1} ---")

            # Get document data
            if isinstance(hit, dict) and "entity" in hit:
                # Search result format
                filename = hit["entity"]["filename"]
                text = hit["entity"]["text"]
                score = hit.get("distance", "N/A")
            else:
                # Query result format
                filename = hit["filename"]
                text = hit["text"]
                score = "N/A"

            # Show if this is a chunk of a larger file
            if '_chunk_' in filename:
                print(f"Filename: {filename}")
                base_name = filename.split('_chunk_')[0]
                chunk_info = filename.split('_chunk_')[1]
                print(f"Original file: {base_name} (Chunk {chunk_info})")
            else:
                print(f"Filename: {filename}")

            if score != "N/A":
                print(f"Score: {score:.4f}")

            # Find occurrences of highlight term if provided
            if highlight_term:
                term_positions = []
                start_pos = 0
                text_lower = text.lower()
                highlight_term_lower = highlight_term.lower()

                while True:
                    pos = text_lower.find(highlight_term_lower, start_pos)
                    if pos == -1:
                        break
                    term_positions.append(pos)
                    start_pos = pos + len(highlight_term)

                if term_positions:
                    print(f"Contains {len(term_positions)} occurrences of '{highlight_term}'")

                    # Show context for first occurrence
                    pos = term_positions[0]
                    start = max(0, pos - 150)
                    end = min(len(text), pos + len(highlight_term) + 150)

                    context = text[start:end]
                    # Highlight the term - get the actual case from the text
                    term_text = text[pos:pos + len(highlight_term)]
                    highlighted = context.replace(term_text, f"**{term_text}**", 1)

                    print(f"Preview: ...{highlighted}...")
                else:
                    # Get a preview of the text (first 200 chars)
                    text_preview = text[:200].replace('\n', ' ').strip()
                    print(f"Preview: {text_preview}...")
            else:
                # Get a preview of the text (first 200 chars)
                text_preview = text[:200].replace('\n', ' ').strip()
                print(f"Preview: {text_preview}...")

            # Show total text length
            print(f"Text length: {len(text)} characters")

    def create_term_index(self, terms, output_file=None):
        """
        Create an index of all occurrences for multiple terms and save to a file

        Parameters:
        - terms: List of terms to index
        - output_file: File to save results to

        Returns:
        - Dictionary with results by term
        """
        term_index = {}

        for term in terms:
            print(f"\nIndexing term: {term}")

            # Get matches
            matches = self.text_match_search(term, limit=2000)

            # Find occurrences
            occurrences, total_count = self.find_all_occurrences(matches, term)

            term_index[term] = {
                "matches": matches,
                "occurrences": occurrences,
                "total_count": total_count
            }

            # Display summary
            print(f"Found {len(matches)} documents with {total_count} total occurrences of '{term}'")

        # Save results to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for term, data in term_index.items():
                    f.write(f"\n\n===== RESULTS FOR TERM: {term} =====\n")
                    f.write(f"Total documents: {len(data['matches'])}\n")
                    f.write(f"Total occurrences: {data['total_count']}\n\n")

                    for filename, info in data["occurrences"].items():
                        f.write(f"Document: {filename}\n")
                        f.write(f"Contains {info['count']} occurrences\n")

                        # Show context for each occurrence
                        for i, pos in enumerate(info["positions"]):
                            text = info["text"]
                            start = max(0, pos - 100)
                            end = min(len(text), pos + len(term) + 100)

                            context = text[start:end]
                            term_text = text[pos:pos + len(term)]
                            highlighted = context.replace(term_text, f"[{term_text}]", 1)

                            f.write(f"  Occurrence {i + 1}: ...{highlighted}...\n\n")

                        f.write("------------------------------\n")

            print(f"\nResults saved to {output_file}")

        return term_index


def main():
    """Main function to run search operations"""
    # search engine
    engine = SearchEngine()

    # weights and limit
    dense_weight = 0.6
    sparse_weight = 0.4
    search_limit = 100  # Default search limit

    while True:
        # CLI for searching
        print("\n=== Milvus Search Interface ===")
        print("Available commands:")
        print("  - create: Initialize collection and insert documents")
        print("  - search <query>: Perform a hybrid search")
        print("  - match <exact text>: Filter for exact text match")
        print("  - hybrid <query> match:<exact text>: Search with text match filter")
        print("  - index <term1,term2,...>: Search index of occurrences")
        print("  - weights <dense_weight> <sparse_weight>: Set search weights")
        print("  - limit <number>: Set the maximum number of search results")
        print(
            f"  - Current settings: limit={search_limit}, weights=[dense={dense_weight:.2f}, sparse={sparse_weight:.2f}]")
        print("  - exit: Exit the program")

        cmd = input("\nEnter command: ").strip()

        if cmd.lower() == 'exit':
            break

        if cmd.lower() == 'create':
            # Create collection and insert documents
            client = create_collection()
            insert_files(client, FILES_DIR)
            print("Collection created and documents inserted")

        elif cmd.lower().startswith('search '):
            query = cmd[7:].strip()
            if not query:
                print("Please provide a search query")
                continue

            # Perform hybrid search without text match
            results = engine.hybrid_search(
                query,
                text_match_filter=None,
                limit=search_limit,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )

            print(f"\nShowing top {min(len(results), search_limit)} results:")
            engine.display_results(results)

        elif cmd.lower().startswith('match '):
            match_text = cmd[6:].strip()
            if not match_text:
                print("Please provide text to match")
                continue

            # Use text match search
            results = engine.text_match_search(match_text, limit=search_limit)

            print(f"\nShowing top {min(len(results), search_limit)} results:")
            engine.display_results(results, highlight_term=match_text)

        elif cmd.lower().startswith('hybrid '):
            parts = cmd[7:].strip().split(' match:')
            if len(parts) != 2:
                print("Usage: hybrid <query> match:<exact text>")
                print("Example: hybrid renewable energy match:solar panel")
                continue

            query = parts[0].strip()
            match_text = parts[1].strip()

            # Perform hybrid search with text match
            results = engine.hybrid_search(
                query,
                text_match_filter=match_text,
                limit=search_limit,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )

            print(f"\nShowing top {min(len(results), search_limit)} results:")
            engine.display_results(results, highlight_term=match_text)

        elif cmd.lower().startswith('index '):
            terms = cmd[6:].strip().split(',')
            terms = [t.strip() for t in terms if t.strip()]

            if not terms:
                print("Please provide at least one term to index")
                continue

            output_file = f"term_index_{time.strftime('%Y%m%d_%H%M%S')}.txt"

            # Create term index
            engine.create_term_index(terms, output_file)

        elif cmd.lower().startswith('weights '):
            try:
                parts = cmd[8:].strip().split()
                if len(parts) != 2:
                    print("Usage: weights <dense_weight> <sparse_weight>")
                    continue

                dense_weight = float(parts[0])
                sparse_weight = float(parts[1])

                # Normalize weights
                total = dense_weight + sparse_weight
                dense_weight /= total
                sparse_weight /= total

                print(f"Weights updated: dense={dense_weight:.2f}, sparse={sparse_weight:.2f}")
            except ValueError:
                print("Error: Weights must be numeric values")

        elif cmd.lower().startswith('limit '):
            try:
                new_limit = int(cmd[6:].strip())
                if new_limit <= 0:
                    print("Error: Limit must be a positive integer")
                    continue

                search_limit = new_limit
                print(f"Search limit updated to {search_limit}")
            except ValueError:
                print("Error: Limit must be a positive integer")

        else:
            print("Unknown command. Available commands: create, search, match, hybrid, index, weights, limit, exit")

if __name__ == "__main__":
    main()