import rarfile

# Specify the path to your RAR file
rar_path = "archive.rar"
# Specify the directory where files should be extracted
extract_to = "./extracted_files"

# Open the RAR archive and extract all contents
with rarfile.RarFile(rar_path) as rf:
    rf.extractall(path=extract_to)

print("Extraction completed successfully!")