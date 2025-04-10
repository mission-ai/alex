import os
import concurrent.futures
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import math
import pickle
import time


class HistoricalSearch:
    def __init__(self, root_dir, meta_dir, num_gpus=3, batch_size=1000):
        self.root_dir = Path(root_dir)
        self.meta_dir = Path(meta_dir)
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.device_pool = [f'cuda:{i}' for i in range(num_gpus)]
        self.index = None
        self.word_counts = defaultdict(int)
        self.total_docs = 0
        self.metadata = {}

    # Load metadata from corresponding text file in meta directory 
    def _load_metadata(self, file_path):
        try:
            # Convert text file path to metadata file path
            relative_path = file_path.relative_to(self.root_dir)
            meta_file = self.meta_dir / relative_path

            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_content = f.read().strip()

                # Parse the metadata text file
                metadata = {}
                current_key = None
                current_value = []

                for line in meta_content.split('\n'):
                    if ':' in line:
                        if current_key:
                            metadata[current_key] = '; '.join(current_value)

                        key, value = line.split(':', 1)
                        current_key = key.strip()
                        current_value = [value.strip()]
                    else:
                        if current_key and line.strip():
                            current_value.append(line.strip())

                if current_key:
                    metadata[current_key] = '; '.join(current_value)

                clean_metadata = {
                    'title': self._get_first_value(metadata.get('TITLE', '')),
                    'author': self._get_first_value(metadata.get('AUTHOR', '')),
                    'date': self._get_publication_date(metadata.get('DATE', '')),
                    'language': metadata.get('LANGUAGE', ''),
                    'publisher': self._get_first_value(metadata.get('PUBLISHER', '')),
                    'pubplace': self._get_first_value(metadata.get('PUBPLACE', '')),
                    'id': self._get_first_value(metadata.get('IDNO', ''))
                }

                return clean_metadata

        except Exception as e:
            print(f"Error loading metadata for {file_path}: {e}")

        return None

    def _get_first_value(self, text):
        if not text:
            return ''
        return text.split(';')[0].strip()

    def _get_publication_date(self, date_text):
        if not date_text:
            return ''

        dates = date_text.split(';')
        for date in dates:
            date = date.strip()
            # Look for dates in brackets like [1586] or plain years like 1693
            if date.startswith('[') and date.endswith(']'):
                return date[1:-1]
            if date.isdigit() and len(date) == 4:
                return date
            if date.isdigit() and 1400 <= int(date) <= 1800:
                return date
        return dates[1].strip()  # fallback to first date

    def _process_batch(self, files):
        batch_index = defaultdict(list)
        batch_counts = defaultdict(int)
        batch_metadata = {}

        for file_path in files:
            try:
                metadata = self._load_metadata(file_path)
                if metadata:
                    batch_metadata[str(file_path)] = metadata

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    words = content.split()

                    file_word_counts = defaultdict(int)
                    for word in words:
                        file_word_counts[word] += 1

                    for word, count in file_word_counts.items():
                        batch_index[word].append((str(file_path), count))
                        batch_counts[word] += count

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return dict(batch_index), dict(batch_counts), batch_metadata

    def _merge_batch_index(self, batch_index, batch_counts, batch_metadata):
        for word, locations in batch_index.items():
            self.index[word].extend(locations)
        for word, count in batch_counts.items():
            self.word_counts[word] += count
        self.metadata.update(batch_metadata)

    def build_index(self):
        self.index = defaultdict(list)
        self.metadata = {}
        files = list(self.root_dir.rglob('*.txt'))
        self.total_docs = len(files)
        print(f"Found {len(files)} files to index")

        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(0, len(files), self.batch_size):
                batch = files[i:i + self.batch_size]
                futures.append(executor.submit(self._process_batch, batch))

            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               desc="Building index"):
                batch_index, batch_counts, batch_metadata = future.result()
                self._merge_batch_index(batch_index, batch_counts, batch_metadata)

        print(f"Indexed {len(files)} files")
        print(f"Total unique words: {len(self.index)}")
        print(f"Metadata loaded for {len(self.metadata)} documents")

    def save_index(self, filepath):
        if not self.index:
            raise ValueError("No index to save. Build index first.")

        save_data = {
            'index': dict(self.index),
            'word_counts': dict(self.word_counts),
            'total_docs': self.total_docs,
            'root_dir': str(self.root_dir),
            'meta_dir': str(self.meta_dir),
            'metadata': self.metadata
        }

        start_time = time.time()
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Index saved in {time.time() - start_time:.2f} seconds")

    @classmethod
    def load_index(cls, filepath):
        print(f"Loading index from {filepath}...")
        start_time = time.time()

        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        instance = cls(
            root_dir=save_data['root_dir'],
            meta_dir=save_data['meta_dir']
        )
        instance.index = defaultdict(list, save_data['index'])
        instance.word_counts = defaultdict(int, save_data['word_counts'])
        instance.total_docs = save_data['total_docs']
        instance.metadata = save_data.get('metadata', {})

        print(f"Index loaded in {time.time() - start_time:.2f} seconds")
        print(f"Loaded index contains {len(instance.index)} unique words")
        print(f"Total documents indexed: {instance.total_docs}")
        print(f"Metadata available for {len(instance.metadata)} documents")

        return instance

    def search(self, query, operation='and', max_results=None, before_year=None, after_year=None):
        """
        Search for multi-word query and their variants across all indexed files with date filtering

        Args:
            query (str): Space-separated search terms
            operation (str): 'and' or 'or' - determines if all words must be present or just some
            max_results (int, optional): Maximum number of results to return
            before_year (int, optional): Only return results before this year
            after_year (int, optional): Only return results after this year

        Returns:
            List of (file_path, score, metadata, matched_variants) tuples and total matches count
        """
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")

        query_words = query.strip().lower().split()
        word_variants = {word: set(generate_variants(word)) for word in query_words}

        for word, variants in word_variants.items():
            print(f"Variants for '{word}': {', '.join(variants)}")

        word_results = {}
        for word, variants in word_variants.items():
            word_results[word] = self._search_single_word_variants(variants)

        if operation.lower() == 'and':
            combined_results = self._combine_results_and(word_results)
        else:
            combined_results = self._combine_results_or(word_results)

        # Apply date filtering
        if before_year is not None or after_year is not None:
            filtered_results = {}
            for file_path, data in combined_results.items():
                doc_date = self.metadata.get(file_path, {}).get('date', '')
                if doc_date:
                    try:
                        year = int(doc_date)
                        if before_year is not None and year > before_year:
                            continue
                        if after_year is not None and year < after_year:
                            continue
                        filtered_results[file_path] = data
                    except ValueError:
                        continue
            combined_results = filtered_results

        formatted_results = self._format_results(combined_results, word_variants)
        sorted_results = sorted(formatted_results, key=lambda x: x[1], reverse=True)

        if max_results:
            sorted_results = sorted_results[:max_results]

        return sorted_results, len(combined_results)

    # Search for all variants of a single word
    def _search_single_word_variants(self, variants):
        results = defaultdict(lambda: {"score": 0.0, "variants": set(), "counts": defaultdict(int)})

        for variant in variants:
            if variant in self.index:
                for file_path, count in self.index[variant]:
                    idf = math.log(self.total_docs / (len(self.index[variant]) + 1))
                    tf = 1 + math.log(count)  # Logarithmic term frequency
                    score = tf * idf

                    results[file_path]["score"] += score
                    results[file_path]["variants"].add(variant)
                    results[file_path]["counts"][variant] = count

        return results

    # Combine results requiring all words to be present
    def _combine_results_and(self, word_results):
        if not word_results:
            return {}

        first_word = list(word_results.keys())[0]
        combined_results = word_results[first_word].copy()

        for word, results in list(word_results.items())[1:]:
            files_to_remove = set()
            for file_path in combined_results:
                if file_path not in results:
                    files_to_remove.add(file_path)
                else:
                    combined_results[file_path]["score"] += results[file_path]["score"]
                    combined_results[file_path]["variants"].update(results[file_path]["variants"])
                    combined_results[file_path]["counts"].update(results[file_path]["counts"])

            for file_path in files_to_remove:
                del combined_results[file_path]

        return combined_results

    # Combine results requiring at least one word to be present
    def _combine_results_or(self, word_results):

        combined_results = defaultdict(lambda: {"score": 0.0, "variants": set(), "counts": defaultdict(int)})

        for results in word_results.values():
            for file_path, data in results.items():
                combined_results[file_path]["score"] += data["score"]
                combined_results[file_path]["variants"].update(data["variants"])
                combined_results[file_path]["counts"].update(data["counts"])

        return combined_results

    # Format results
    def _format_results(self, results, word_variants):
        formatted_results = []
        original_words = word_variants.keys()

        for file_path, data in results.items():
            filename = Path(file_path).name

            # Get metadata for document
            meta = self.metadata.get(file_path, {})
            meta_str = ""
            if meta:
                meta_str = f"\nTitle: {meta.get('title', 'Unknown Title')}"
                if meta.get('author'):
                    meta_str += f"\nAuthor: {meta.get('author')}"
                if meta.get('date'):
                    meta_str += f"\nDate: {meta.get('date')}"
                if meta.get('publisher'):
                    meta_str += f"\nPublisher: {meta.get('publisher')}"
                if meta.get('pubplace'):
                    meta_str += f"\nPlace: {meta.get('pubplace')}"

            variants_info = []
            for original_word in original_words:
                word_variants_found = [
                    f"{variant}({data['counts'][variant]})"
                    for variant in data["variants"]
                    if variant.lower() in word_variants[original_word]
                       and variant.lower() != original_word
                ]

                if word_variants_found:
                    variants_info.append(f"{original_word}: {', '.join(word_variants_found)}")

            variant_str = f"\nVariants: {'; '.join(variants_info)}" if variants_info else ""

            formatted_results.append((
                filename,
                data["score"],
                meta_str,
                variant_str
            ))

        return formatted_results

# Generate historical spelling variants of a word including case variations
def generate_variants(word):
    base_variants = {word.lower()}
    word_lower = word.lower()

    # Special cases for common proper names
    if word_lower == "jesus":
        specific_variants = {
            "jesus", "iesus", "jesvs", "iesvs"
        }
        all_variants = set()
        for variant in specific_variants:
            all_variants.update([
                variant.lower(),
                variant.upper(),
                variant.capitalize()
            ])
        return list(all_variants)

    # General substitution rules
    substitutions = {
        'i': ['y', 'j'],
        'j': ['i'],
        'y': ['i'],
        'u': ['v'],
        'v': ['u'],
        'w': ['vv'],
        'f': ['ff'],
        'ph': ['f'],
    }

    # Apply substitutions with position constraints
    for old, news in substitutions.items():
        if old in word_lower:
            for new in news:
                if old in ['i', 'j'] and word_lower.startswith(old):
                    variant = new + word_lower[1:]
                    base_variants.add(variant)

                elif old == 'y' and word_lower.endswith(old):
                    variant = word_lower[:-1] + new
                    base_variants.add(variant)
                    if new == 'i':
                        base_variants.add(word_lower[:-1] + 'ie')

                elif old in ['u', 'v']:
                    parts = word_lower.split(old)
                    for i in range(len(parts) - 1):
                        if not parts[i].endswith(('q', 'g')):
                            new_parts = parts.copy()
                            new_parts[i] = new_parts[i] + new
                            variant = old.join(new_parts)
                            base_variants.add(variant)

                elif old == 'f' and not word_lower.endswith(old):
                    variant = word_lower.replace(old, new)
                    base_variants.add(variant)

                elif old == 'ph':
                    variant = word_lower.replace(old, new)
                    base_variants.add(variant)

    # Generate case variations
    all_variants = set()
    for variant in base_variants:
        all_variants.update([
            variant.lower(),
            variant.upper(),
            variant.capitalize(),
            variant.title()
        ])

    return list(all_variants)

def display_results(results, start_idx, page_size):
    end_idx = min(start_idx + page_size, len(results))
    for idx in range(start_idx, end_idx):
        filename, score, metadata, variants = results[idx]
        print(f"\n{filename}")
        print(f"Score: {score:.3f}")
        print(metadata)
        print(variants)
        print("-" * 80)
    return end_idx

if __name__ == "__main__":
    data_dir = "C:/Users/neeln/OneDrive/Documents/College/UFlorida/Other Stuff/Job Stuff/AI Internship/MilvusP/text"
    meta_dir = "C:/Users/neeln/OneDrive/Documents/College/UFlorida/Other Stuff/Job Stuff/AI Internship/MilvusP/meta"
    index_path = "saved_indices/historical_search_index.pkl"

    if os.path.exists(index_path):
        searcher = HistoricalSearch.load_index(index_path)
    else:
        searcher = HistoricalSearch(root_dir=data_dir, meta_dir=meta_dir)
        searcher.build_index()
        searcher.save_index(index_path)

    while True:
        query = input("\nSearch (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        before_year = input("Before year (leave empty if no filter): ")
        after_year = input("After year (leave empty if no filter): ")
        page_size = input("Results per page (default 10): ")
        operation_OA = input("and/or (default 'or'): ")

        before_year = int(before_year) if before_year.strip() else None
        after_year = int(after_year) if after_year.strip() else None
        page_size = int(page_size) if page_size.strip() else 10
        operation_OA = operation_OA.strip().lower() if operation_OA.strip().lower() in ['and', 'or'] else 'or'

        results, total_matches = searcher.search(
            query,
            operation=operation_OA,
            before_year=before_year,
            after_year=after_year
        )

        if not results:
            print("No matches found")
            continue

        print(f"\nFound {total_matches} total matches")
        current_idx = 0

        while current_idx < len(results):
            current_idx = display_results(results, current_idx, page_size)

            if current_idx >= len(results):
                print("\nNo more results.")
                break

            action = input("\nEnter 'n' for next page, 'new' for new search, or 'exit' to quit: ")
            if action.lower() == 'exit':
                exit()
            elif action.lower() == 'new':
                break
            elif action.lower() != 'n':
                print("Invalid input. Starting new search.")
                break