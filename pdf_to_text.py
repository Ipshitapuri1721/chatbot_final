from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(r"D:\pic\Attachments\Custom Office Templates\Desktop\college_enquiry_bot\data\pdfs\final0.2.pdf")
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


if __name__ == "__main__":
    pdf_path = "data/questions.pdf"   # your PDF path
    output_path = "data/raw_text.txt"

    text = extract_text_from_pdf(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("PDF converted to raw_text.txt successfully!")