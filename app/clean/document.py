import io
import os
import re
import zipfile

from ..detect.document import ooxml_kind


def _clean_pdf(data: bytes, strength: str, actions):
    import pikepdf
    pdf = pikepdf.open(io.BytesIO(data))
    root = pdf.Root

    n = len(pdf.docinfo)
    for k in list(pdf.docinfo.keys()):
        del pdf.docinfo[k]
    if n:
        actions.append(f"Cleared {n} docinfo keys")

    removed = []
    for name in ("/Metadata", "/PieceInfo"):
        if name in root:
            del root[name]
            removed.append(name)
    try:
        names = root.get("/Names")
        if names is not None and "/EmbeddedFiles" in names:
            del names["/EmbeddedFiles"]
            removed.append("/Names/EmbeddedFiles")
    except Exception:
        pass
    for page in pdf.pages:
        if "/Metadata" in page:
            del page["/Metadata"]
            removed.append("page /Metadata")

    # JavaScript / open actions
    js_removed = []
    for key in ("/OpenAction", "/AA"):
        if key in root:
            obj = root[key]
            if re.search(r"/JavaScript|/JS\b", str(obj)):
                del root[key]
                js_removed.append(key)
    for page in pdf.pages:
        try:
            aa = page.get("/AA")
            if aa is not None and re.search(r"/JavaScript|/JS\b", str(aa)):
                del page["/AA"]
                js_removed.append("page /AA")
        except Exception:
            pass
    if js_removed:
        actions.append(f"Removed JavaScript actions ({', '.join(sorted(set(js_removed)))})")

    if removed:
        actions.append("Deleted " + ", ".join(removed))

    if strength in ("standard", "heavy"):
        pdf.trailer.ID = pikepdf.Array([pikepdf.String(os.urandom(16).hex()),
                                      pikepdf.String(os.urandom(16).hex())])
        actions.append("Regenerated trailer /ID")
    if strength == "heavy":
        stripped = 0
        for page in pdf.pages:
            try:
                annots = page.get("/Annots", []) or []
                for annot in annots:
                    try:
                        a = annot.get("/A")
                        if a is not None and a.get("/URI"):
                            u = str(a["/URI"])
                            if "?" in u:
                                a["/URI"] = pikepdf.String(u.split("?")[0])
                                stripped += 1
                        for k in ("/T", "/NM", "/CreationDate", "/M"):
                            if k in annot:
                                del annot[k]
                    except Exception:
                        continue
            except Exception:
                continue
        if stripped:
            actions.append(f"Stripped query strings on {stripped} URI annotations")
        actions.append("Cleared annotation identity fields (/T, /NM, /CreationDate, /M)")

    buf = io.BytesIO()
    pdf.save(buf, linearize=False)
    pdf.close()
    actions.append("Saved without metadata (docinfo, XMP, pieceinfo removed)")
    return buf.getvalue(), ".pdf", actions


def _clean_ooxml(data: bytes, kind: str, strength: str, actions):
    src = zipfile.ZipFile(io.BytesIO(data))
    names = src.namelist()
    drop = set()

    def drop_part(name, why):
        if name in names:
            drop.add(name)
            actions.append(why)

    drop_part("docProps/custom.xml", "Removed docProps/custom.xml")
    for n in names:
        if n.startswith("docProps/thumbnail"):
            drop_part(n, "Removed document thumbnail")
    drop_part("word/people.xml", "Removed word/people.xml (comment authors)")

    def edit(name, text):
        return text

    buf = io.BytesIO()
    out = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
    for n in names:
        if n in drop:
            continue
        content = src.read(n)
        if n == "docProps/core.xml":
            t = content.decode("utf-8", "replace")
            title = re.search(r"<dc:title>[^<]*</dc:title>", t)
            t = ("<?xml version='1.0' encoding='UTF-8' standalone='yes'?>\n"
                 '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/'
                 'package/2006/metadata/core-properties" xmlns:dc="http://purl.org/'
                 'dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
                 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                 + (title.group(0) if title else "")
                 + "</cp:coreProperties>")
            content = t.encode()
            actions.append("Cleared core.xml (creator/dates/revision)")
        elif n == "docProps/app.xml":
            t = content.decode("utf-8", "replace")
            t = re.sub(r"<(Application|Company|Manager|TotalTime|Template|AppVersion|HyperlinkBase)>[^<]*</\1>", "", t)
            content = t.encode()
            actions.append("Removed app.xml identity fields")
        elif n == "word/settings.xml":
            t = content.decode("utf-8", "replace")
            t = re.sub(r'<w:rsids>.*?</w:rsids>', "", t, flags=re.S)
            t = re.sub(r'\sw1[45]:(docId|paraId|textId)="[^"]*"', "", t)
            content = t.encode()
            actions.append("Stripped rsids + docIds from settings.xml")
        elif n == "[Content_Types].xml":
            t = content.decode("utf-8", "replace")
            for d in drop:
                t = re.sub(r'<Override[^>]*PartName="/' + re.escape(d) + r'"[^>]*/>', "", t)
            content = t.encode()
        elif n == "_rels/.rels" or n.endswith(".rels"):
            t = content.decode("utf-8", "replace")
            for d in drop:
                t = re.sub(r'<Relationship[^>]*Target="[^"]*' + re.escape(os.path.basename(d)) + r'"[^>]*/>', "", t)
            if strength in ("standard", "heavy"):
                before = t
                t = re.sub(r'<Relationship[^>]*TargetMode="External"[^>]*/>', "", t)
                if t != before:
                    actions.append("Removed external relationships (linked images may show placeholders)")
            content = t.encode()
        elif n == "word/document.xml" or n.startswith("word/comments"):
            t = content.decode("utf-8", "replace")
            t2 = re.sub(r'\sw:rsid[A-Za-z]*="[^"]*"', "", t)
            t2 = re.sub(r'\sw:(author|initials)="[^"]*"', ' w:author="Author"', t2)
            t2 = re.sub(r'\sw:date="[^"]*"', "", t2)
            if t2 != t:
                actions.append(f"Neutralized author/rsid attributes in {n}")
            content = t2.encode()
        elif n in ("_rels/.rels",):
            content = content
        out.writestr(n, content)
    out.close()
    src.close()
    return buf.getvalue(), "." + kind, actions


def clean_document(data: bytes, strength: str = "standard", ctx: dict | None = None):
    ctx = ctx or {}
    strength = strength if strength in ("light", "standard", "heavy") else "standard"
    actions = []
    if data[:4] == b"%PDF":
        out, ext, a = _clean_pdf(data, strength, actions)
        return out, ext, a, {"in_bytes": len(data), "out_bytes": len(out)}
    zf = zipfile.ZipFile(io.BytesIO(data))
    kind = ooxml_kind(zf)
    zf.close()
    if kind:
        out, ext, a = _clean_ooxml(data, kind, strength, actions)
        return out, ext, a, {"in_bytes": len(data), "out_bytes": len(out)}
    raise ValueError("unrecognized document")
