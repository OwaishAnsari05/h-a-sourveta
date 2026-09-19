# PDF Loader 

from langchain_community.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader("data/documents/Tata_annual_report.pdf")
documents = pdf_loader.load()
print("--- TEXT OUTPUT ---")
print(documents[6].page_content[:400])  # Display the first 400 characters of the sixth page
print(len(documents))
print(documents[0].page_content[:400]) # display the first 400 characters of the first page
print(documents[0].metadata) # display the metadata of the first page