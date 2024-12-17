import pdfplumber

def extract_text_and_tables(pdf_path):
    text_data = []
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                text_data.append({"page": page_number, "content": text})
            table = page.extract_table()
            if table:
                tables.append({"page": page_number, "table": table})
    return text_data, tables

if __name__ == "__main__":
    pdf_path = "https://www.hunter.cuny.edu/dolciani/pdf_files/workshop-materials/mmc-presentations/tables-charts-and-graphs-with-examples-from.pdf"
    text_data, tables = extract_text_and_tables(pdf_path)
    print("Text Data:", text_data)
    print("Tables:", tables)
