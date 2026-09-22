import io
import zipfile

import pytest

from app.detect.document import analyze_document
from app.clean.document import clean_document


def make_docx():
    buf = io.BytesIO()
    z = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
    z.writestr("[Content_Types].xml", """<?xml version="1.0"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
<Override PartName="/docProps/custom.xml" ContentType="application/vnd.openxmlformats-officedocument.custom-properties+xml"/>
</Types>""")
    z.writestr("_rels/.rels", """<?xml version="1.0"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties" Target="docProps/custom.xml"/>
</Relationships>""")
    z.writestr("word/document.xml", """<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<w:body><w:p w:rsidR="004A1234" w:rsidRDefault="004A1234"><w:r><w:t>Hello world</w:t></w:r></w:p>
<w:p w:rsidR="0055BFFF"><w:ins w:author="Jane Doe" w:date="2026-01-01T00:00:00Z"><w:r><w:t>inserted</w:t></w:r></w:ins></w:p>
</w:body></w:document>""")
    z.writestr("word/settings.xml", """<?xml version="1.0"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml">
<w:rsids><w:rsid w:val="004A1234"/><w:rsid w:val="0055BFFF"/></w:rsids>
<w14:paraId w14:val="1A2B3C4D" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"/>
</w:settings>""")
    z.writestr("docProps/core.xml", """<?xml version="1.0"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:creator>Jane</dc:creator><cp:lastModifiedBy>JDoe</cp:lastModifiedBy>
<dc:title>Report</dc:title>
<dcterms:created xsi:type="dcterms:W3CDTF">2026-01-01T00:00:00Z</dcterms:created>
</cp:coreProperties>""")
    z.writestr("docProps/app.xml", """<?xml version="1.0"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
<Application>Microsoft Word</Application><Company>Acme Corp</Company>
<TotalTime>120</TotalTime></Properties>""")
    z.writestr("docProps/custom.xml", """<?xml version="1.0"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties">
<property fmtid="{D5CDD505-2E9C-101B-9997-08002B2CF9AE}" pid="2" name="EmployeeID"><vt:lpwstr>EMP-9</vt:lpwstr></property>
</Properties>""")
    z.close()
    return buf.getvalue()


def ids(rep):
    return {f.id for f in rep["findings"]}


def test_docx_detect_clean():
    data = make_docx()
    rep = analyze_document(data, "doc.docx")
    fids = ids(rep)
    assert "owner-identity" in fids
    assert "timestamps" in fids
    assert "software-tags" in fids
    assert "custom-properties" in fids
    assert "unique-ids" in fids
    assert "tracked-changes" in fids

    out, ext, actions, _ = clean_document(data, "standard", {"filename": "doc.docx"})
    assert ext == ".docx"
    zf = zipfile.ZipFile(io.BytesIO(out))
    assert zf.testzip() is None
    assert "docProps/custom.xml" not in zf.namelist()
    after = analyze_document(out, "doc.docx")
    assert not {"owner-identity", "unique-ids", "custom-properties"} & ids(after)
    # document body text preserved
    assert b"Hello world" in zf.read("word/document.xml")


def make_pdf():
    import pikepdf
    pdf = pikepdf.new()
    pdf.add_blank_page()
    pdf.docinfo["/Author"] = "Jane"
    pdf.docinfo["/Producer"] = "AcmePDF 1.0"
    pdf.docinfo["/CreationDate"] = "D:20260101000000Z"
    pdf.Root.Metadata = pdf.make_stream(
        b"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description><xmpMM:DocumentID xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/">uuid:AAAA-1234</xmpMM:DocumentID></rdf:Description>
</rdf:RDF></x:xmpmeta>""")
    buf = io.BytesIO()
    pdf.save(buf)
    pdf.close()
    return buf.getvalue()


def test_pdf_detect_clean():
    data = make_pdf()
    rep = analyze_document(data, "doc.pdf")
    fids = ids(rep)
    assert "owner-identity" in fids
    assert "pdf-xmp" in fids
    assert "software-tags" in fids
    out, ext, actions, _ = clean_document(data, "standard", {"filename": "doc.pdf"})
    assert ext == ".pdf"
    after = analyze_document(out, "doc.pdf")
    fids2 = ids(after)
    assert not {"owner-identity", "pdf-xmp", "software-tags"} & fids2
