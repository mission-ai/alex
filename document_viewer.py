import os
import sys
import tkinter as tk
from tkinter import scrolledtext
from pathlib import Path
import threading


def view_document_by_id(document_id, root_dir, extensions=None, recursive=True):
    """
    Searches for a file with the given document ID

    Args:
        document_id (str): The document ID to search for (e.g., "A02495")
        root_dir (str): The root directory to search in
        extensions (list, optional): List of file extensions to consider
        recursive (bool, optional): Whether to search subdirectories

    Returns:
        tuple: (content, filename) if found, else (None, None)
    """
    if extensions is None:
        extensions = ['.txt']

    # Clean the input document ID
    document_id = document_id.strip().upper()
    if '.' in document_id:
        document_id = document_id.split('.')[0]

    root_path = Path(root_dir)

    if recursive:
        search_method = root_path.rglob
    else:
        search_method = root_path.glob

    # Search for matching files
    matching_files = []
    for ext in extensions:
        matching_files.extend(search_method(f"*{document_id}*{ext}"))

    if not matching_files:
        return None, None

    # Sort files by length of filename
    matching_files.sort(key=lambda x: len(x.name))

    selected_file = matching_files[0]

    try:
        with open(selected_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, selected_file
    except Exception as e:
        print(f"Error opening file: {e}")
        return None, selected_file

def open_document_window(content, filename):
    """Opens a window to display the document content"""
    root = tk.Tk()
    root.title(f"Document Viewer - {filename.name if filename else 'Not Found'}")
    root.geometry("800x600")

    # Create frame for the content
    main_frame = tk.Frame(root, padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Add label with the filename
    if filename:
        file_label = tk.Label(main_frame, text=f"File: {filename}", anchor=tk.W, font=("Arial", 10, "bold"))
        file_label.pack(fill=tk.X, pady=(0, 5))

    text_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=("Courier New", 10))
    text_area.pack(fill=tk.BOTH, expand=True)

    # Insert content with line numbers
    if content:
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            text_area.insert(tk.END, f"{i:4d} | {line}\n")
    else:
        text_area.insert(tk.END, "Document not found or could not be opened.")

    # Make text area read-only
    text_area.config(state=tk.DISABLED)

    close_button = tk.Button(main_frame, text="Close", command=root.destroy)
    close_button.pack(pady=(5, 0))

    root.mainloop()

def main():
    default_root_dir = "C:/Users/neeln/OneDrive/Documents/College/UFlorida/Other Stuff/Job Stuff/AI Internship/MilvusP/text"

    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        root_dir = sys.argv[1]
    else:
        root_dir = default_root_dir

    print(f"Document viewer initialized. Searching in: {root_dir}")
    print("Enter document ID (e.g., 'A02495') or 'exit' to quit.")

    while True:
        # document ID from user
        document_id = input("\nDocument ID: ").strip()

        if document_id.lower() == 'exit':
            print("Exiting document viewer.")
            break

        if not document_id:
            print("Please enter a valid document ID.")
            continue

        # Search
        content, filename = view_document_by_id(
            document_id,
            root_dir,
            extensions=['.txt', '.P4.txt',],
            recursive=True
        )

        if content is None:
            if filename:
                print(f"Found file {filename} but couldn't read it.")
            else:
                print(f"No document found with ID: {document_id}")
            continue

        print(f"Found document: {filename}")

        threading.Thread(target=open_document_window, args=(content, filename), daemon=True).start()
        print("Document opened in new window. You can continue searching.")

if __name__ == "__main__":
    main()