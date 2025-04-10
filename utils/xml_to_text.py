#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Call tree for the xml_to_text module

process_directory()
    process_xml_file()
        replace_tags()
        extract_metadata()
        process_body()
            get_text() 
            clean_text()       
    format_metadata()
"""
__author__     = "Neel Patel"
__license__    = "MIT License"
__version__    = "0.0.1"
__email__      = "danielmaxwell@ufl.edu"
__maintainer__ = "AI Civilization Lab"
__copyright__  = "AI Civilization Lab @ The University of Florida"
__credits__    = ["Neel Patel"]

import os
import xml.etree.ElementTree as ET
import re

def extract_metadata(root):
    """This function extracts metadata from the XML header section.
    
       Args:
           root: The root element of an XML hierarchy. The object returned by ElementTree getroot().
           
       Returns:
           metadata
    """
    metadata = {}
    header = root.find('.//HEADER')  # Find the HEADER section in the XML
    if header is not None:
        for elem in header.iter():   # Iterate through all elements in the HEADER
            tag = elem.tag.lower()
            if tag not in metadata:
                metadata[tag] = elem.text.strip() if elem.text else ''
            else:
                # Append additional text if the tag already exists
                additional_text = elem.text.strip() if elem.text else ''
                if additional_text:
                    metadata[tag] += f"; {additional_text}"
    return metadata

def process_body(root):
    """This function processes the body content and cleans it.
        
       Args:
           root: The root element of an XML hierarchy. The object returned by ElementTree getroot().
           
       Returns:
           sections: The text found in the xml document's body.
    """
    sections = []
    current_section = []
    start_processing = False

    for elem in root.iter():
        if elem.tag == 'DIV1':
            start_processing = True
            continue  # Skip the metadata sections itself

        if start_processing:
            if elem.tag == 'DIV1':
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
            if elem.tag in ['HEAD', 'Q']:
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
                current_section.append(f"\n{clean_text(get_text(elem))}\n")
            elif elem.tag == 'L':
                current_section.append(clean_text(get_text(elem)))
            elif elem.tag == 'P':
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
                sections.append(clean_text(get_text(elem)))

    if current_section:
        sections.append('\n'.join(current_section))

    return '\n\n'.join(sections)

def get_text(elem):
    """This function extracts the text from an element, including text within tags."""
    text = (elem.text or '') + ''.join((child.text or '') + (child.tail or '') for child in elem)
    return text

def clean_text(text):
    """This function cleans and formats text content."""
    if text is None:
        return ''
    text = text.replace('∣', '').replace('▪', '')  # Remove specific unwanted characters
    text = re.sub(r'<[^>]+>', '', text)            # Remove other HTML-like tags
    text = re.sub(r'[ \t]+', ' ', text)            # Replace multiple spaces/tabs with a single space
    text = re.sub(r'\n\s*\n', '\n\n', text)        # Replace multiple newlines with double newlines

    return text.strip()

def replace_tags(xml_content):
    """This function replaces specific tags in the XML content."""
    # Replace <ABBR><HI>l</HI></ABBR> with (pounds)
    xml_content = xml_content.replace('<ABBR><HI>l</HI></ABBR>', '(pounds)')

    # Replace <P><GAP DESC="music" DISP="〈♫〉"/></P> with (music)
    xml_content = xml_content.replace('<GAP DESC="music" DISP="〈♫〉"/>', '〈♫〉')

    # Replace <SEG REND="decorInit">X</SEG> with X for any letter X
    xml_content = re.sub(r'<SEG REND="decorInit">(.*?)</SEG>', r'\1', xml_content)

    # Replace <HI><SEG REND="decorInit">X</SEG>Y</HI> with XY for any letters X and Y
    xml_content = re.sub(r'<HI><SEG REND="decorInit">(.*?)</SEG>(.*?)</HI>', r'\1\2', xml_content)

    return xml_content

def process_xml_file(xml_file_path):
    """This function parses a single XML file and processes its content.
    
       Args:
           xml_file_path: The path to a single xml file.
           
       Returns:
           metadata, clean_content
    """
    with open(xml_file_path, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # Replace specific tags in the XML content
    xml_content = replace_tags(xml_content)

    tree = ET.ElementTree(ET.fromstring(xml_content))  # Parse the modified XML content
    root = tree.getroot()                              # Get the root element of the XML

    # Extract metadata, process the text body, and then clean it.
    metadata = extract_metadata(root)
    all_text_context = process_body(root)
    clean_content = clean_text(all_text_context)

    return metadata, clean_content

def format_metadata(metadata):
    """This function formats metadata for display."""
    return "\n".join([f"{key.upper()}: {value}" for key, value in metadata.items() if value])

def process_directory(directory_path, metadata_output_directory, text_output_directory):
    """This function process all XML files in a directory and saves output to specified directories.
    
       Args: 
            directory_path: The path to the source xml files.
            metadata_output_directory: The directory where metadata files will be written.
            text_output_directory: The directory where the cleaned text files will be written.
             
       Returns:
            None - prints success message if no errors encountered. 
    """
    for filename in os.listdir(directory_path):
        if filename.endswith('.xml'):
            xml_file_path = os.path.join(directory_path, filename)
            metadata, clean_content = process_xml_file(xml_file_path)
            formatted_metadata = format_metadata(metadata)

            # Ensure the output directories exist
            # os.makedirs(metadata_output_directory, exist_ok=True)
            # os.makedirs(text_output_directory, exist_ok=True)

            # Save metadata to the specified directory
            metadata_file_path = os.path.join(metadata_output_directory, f"{os.path.splitext(filename)[0]}.txt")
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_metadata)

            # Save text content to the specified directory
            text_file_path = os.path.join(text_output_directory, f"{os.path.splitext(filename)[0]}.txt")
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(clean_content)

            print(f"Processed and saved: {metadata_file_path} and {text_file_path}")
            
    return        
    
# Sample code for calling the top-level (process_directory) function.

directory_path = '../B4_Ph2/B4'
metadata_output_directory = '../data_split/meta'
text_output_directory = '../data_split/text'

process_directory(directory_path, metadata_output_directory, text_output_directory)

os.getcwd()
help (process_body)
