#!/bin/bash

echo "Starting to generate MD5 checksums for files listed in tests/data.filenames.txt..."

# Prefix each line from data.filenames.txt with 'data/' and calculate md5sum
while IFS= read -r file; do
    # Display the file being processed
    echo "Generating MD5 for: data/$file"
    # Generate MD5 and append to the output file
    md5sum "data/$file" >> data.filenames.md5
done < data.filenames.txt

echo "MD5 checksum generation complete. Checksums stored in tests/data.filenames.md5."
