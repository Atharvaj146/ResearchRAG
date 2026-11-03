import fitz

document = fitz.open("test.pdf")

print("Number of pages:", len(document))

for page_number, page in enumerate(document):
    text = page.get_text()

    print("\n----------------")
    print("Page:", page_number + 1)
    print("Characters extracted:", len(text))
    print("Text:", text[:200])

document.close()