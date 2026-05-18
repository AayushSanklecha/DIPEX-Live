import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            res = []
            for node in tree.iter():
                if node.tag.endswith('}t') and node.text:
                    res.append(node.text)
            return "".join(res)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    path = r"C:\Users\sankl\Desktop\dipex_project\DIPEX_Blackbook_Complete.docx"
    text = extract_text_from_docx(path)
    with open("docx_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Done")
