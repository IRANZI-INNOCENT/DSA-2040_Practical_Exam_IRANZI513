import sys
import os

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not available. Installing...")
    os.system("pip install PyPDF2")
    import PyPDF2

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- PAGE {page_num + 1} ---\n"
                text += page.extract_text()
                
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

if __name__ == "__main__":
    pdf_path = r"c:\DSA 2040_Practical_Exam_IRANZI513\Instructiionsdoc.pdf"
    
    if os.path.exists(pdf_path):
        text = extract_pdf_text(pdf_path)
        print(text)
        
        # Save to text file for easier reading
        with open(r"c:\DSA 2040_Practical_Exam_IRANZI513\instructions.txt", 'w', encoding='utf-8') as f:
            f.write(text)
        print("\n\nText saved to instructions.txt")
    else:
        print(f"PDF file not found: {pdf_path}")
