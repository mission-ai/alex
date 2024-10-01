readMe for XML_to_text

Extract_metadata
The extract_metadata function is designed to extract metadata from the XML header section. It begins by initializing an empty dictionary named metadata to store the extracted information. The function then searches for the HEADER section within the XML tree starting from the root element using the find method. If the HEADER section is found, the function iterates through all elements within this section. For each element, it converts the tag name to lowercase and checks if the tag is already present in the metadata dictionary. If the tag is not present, it adds the tag as a key with its text content as the value, stripping any leading and trailing whitespace. If the tag is already present, it appends the new text content to the existing value, separated by a semicolon. Finally, the function returns the metadata dictionary containing the extracted metadata from the HEADER section.

process_body
The process_body function processes and cleans the body content of an XML element. It starts by initializing empty lists named sections and current_section.  Set a flag start_processing to False. Iterate through all elements in the XML. If the element tag is DIV1, set start_processing to True and continue. If start_processing is True, process the elements based on their tags (HEAD, Q, L, P). Append the cleaned text to the current_section or sections list. Return the processed sections as a single string.

get_text
This function concatenates the element’s text and the text of its children. Then it returns the concatenated text. 

clean_text
The clean_text function is designed to clean and format text content. It starts by removing specific unwanted characters, such as ‘∣’ and ‘▪’, using the replace method. Next, it removes any HTML-like tags from the text by using the re.sub function with a regular expression that matches any content within angle brackets. The function then replaces multiple spaces or tabs with a single space, again using the re.sub function with a regular expression that matches one or more spaces or tabs. It also replaces multiple newlines with double newlines. Finally, the function returns the cleaned text with leading and trailing whitespace removed by using the strip method.

replace_tags
This function uses string replacement and regular expressions to replace specific tags in the text that can be deemed as important. This returns the modified XML content. More tags can be added based on the meeting with the scholars. 

process_xml_file
The process_xml_file function is designed to parse an XML file and process its content. This function opens each XML file and reads its content. It then replaces specific tags in the content and then parses the modified file. The root element is retrieved, and the metadata is extracted from the root element. The body content is processed and cleaned. The metadata and cleaned content are returned separately as two files are needed. 

format_metadata
The format_metadata function is designed to format metadata for display. It takes a dictionary named metadata as input, where each key-value pair represents a piece of metadata. The function uses a list comprehension to iterate over each key-value pair in the dictionary. For each pair, it converts the key to uppercase and formats it together with the value into a string in the format KEY: value. The function includes only those key-value pairs where the value is not empty. It then joins the formatted strings with newline characters to create a single string, with each key-value pair on a new line. Finally, the function returns this formatted string, making the metadata easy to read and display.

process_directory
The process_directory function processes all XML files in a specified directory and saves the output to another specified directory. It begins by iterating through all files in the directory. If the file is an XML file, the information is processed. The metadata and cleaned content are both extracted from the file. The metadata is formatted for display. The metadata and the cleaned content are directed towards their specified directories (preferably “text” and “metadata” when calling the function). The program finally prints out when the files have been processed and saved to their specified directories.
