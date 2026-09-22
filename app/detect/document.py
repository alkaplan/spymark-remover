import io
import re
import zipfile

from ..findings import Finding

PDF_EXTS = {".pdf"}
OOXML_EXTS = {".docx", ".xlsx", ".pptx"}


def _f(fid, cat, name, sev, conf, ev, rem=True, removal=""):
    return Finding(id=fid, category=cat, name=name, severity=sev,
                   confidence=conf, evidence=str(ev)[:200], removable=rem,
                   removal=removal)


# ---------------- PDF ----------------

def _pdf_findings(data: bytes):
    import pikepdf
    findings = []
    pdf = pikepdf.open(io.BytesIO(data))
    root = pdf.Root

    docinfo = {str(k): str(v) for k, v in (pdf.docinfo or {}).items()}
    ident = {k: v for k, v in docinfo.items()
             if k.lstrip("/") in ("Author", "Creator") and v}
    soft = {k: v for k, v in docinfo.items()
            if k.lstrip("/") in ("Producer", "CreatorTool") and v}
    times = {k: v for k, v in docinfo.items() if "Date" in k and v}
    if ident:
        findings.append(_f("owner-identity", "metadata", "PDF identity (docinfo)",
                           "high", 1.0, "; ".join(f"{k}={v}" for k, v in ident.items()),
                           removal="docinfo cleared"))
    if soft:
        findings.append(_f("software-tags", "metadata", "PDF producer software",
                           "low", 1.0, "; ".join(f"{k}={v}" for k, v in soft.items()),
                           removal="docinfo cleared"))
    if times:
        findings.append(_f("timestamps", "metadata", "PDF timestamps",
                           "low", 1.0, "; ".join(f"{k}={v}" for k, v in times.items()),
                           removal="docinfo cleared"))

    # XMP
    try:
        xmp = bytes(root.Metadata.read_bytes())
    except Exception:
        xmp = b""
    if xmp:
        hits = re.findall(rb"(?:Document|Instance|OriginalDocument)ID[^<\"]{0,60}|"
                          rb"dc:creator[^<]{0,120}|<dc:creator[^>]*>|<rdf:li>[^<]{0,120}",
                          xmp)
        findings.append(_f("unique-ids", "metadata", "PDF XMP metadata", "medium",
                           1.0, "; ".join(h.decode("utf-8", "replace")[:80]
                                          for h in hits[:5]) or "XMP packet present",
                           removal="/Metadata stream deleted"))
        findings[-1].id = "pdf-xmp"

    # trailer ID
    try:
        tid = pdf.trailer.get("/ID")
        if tid is not None:
            findings.append(_f("pdf-trailer-id", "metadata", "PDF trailer /ID",
                               "low", 1.0, str(tid)[:120],
                               removal="Fresh random /ID on save (standard+)"))
    except Exception:
        pass

    # names / embedded files
    try:
        names = root.get("/Names")
        if names and names.get("/EmbeddedFiles"):
            efs = names["/EmbeddedFiles"].get("/Names", [])
            ns = [str(x) for x in efs if not str(x).startswith("<<")]
            findings.append(_f("pdf-embedded-files", "metadata",
                               "Embedded files", "medium", 1.0,
                               ", ".join(ns[:6]) or "EmbeddedFiles present",
                               removal="EmbeddedFiles name tree removed"))
    except Exception:
        pass

    # JavaScript
    js = False
    try:
        allstr = " ".join(str(o) for o in [root.get("/Names"), root.get("/OpenAction"),
                                           root.get("/AA")])
        js = bool(re.search(r"/JavaScript|/JS\b", allstr))
    except Exception:
        pass
    if not js:
        try:
            for page in pdf.pages:
                if "/AA" in str(page.get("/Annots", "")) or re.search(
                        r"/JavaScript|/JS\b", str(page)):
                    js = True
                    break
        except Exception:
            pass
    if js:
        findings.append(_f("pdf-javascript", "tracking", "JavaScript actions",
                           "high", 0.9, "/JS or /JavaScript entries in catalog/page actions",
                           removal="OpenAction/AA JavaScript entries removed"))

    # remote links
    links = []
    try:
        for page in pdf.pages:
            for annot in page.get("/Annots", []) or []:
                try:
                    a = annot.get("/A")
                    uri = a.get("/URI") if a else annot.get("/URI")
                    if uri:
                        u = str(uri)
                        if "?" in u or re.search(r"utm_|track|pixel|1x1|beacon", u, re.I):
                            links.append(u[:100])
                    sub = str((a or {}).get("/S", ""))
                    if sub in ("/Launch", "/GoToR", "/SubmitForm", "/ImportData"):
                        links.append(f"{sub} action")
                except Exception:
                    continue
    except Exception:
        pass
    if links:
        findings.append(_f("pdf-remote-links", "tracking", "Remote/tracked links",
                           "medium", 0.9, "; ".join(links[:5]),
                           removal="Heavy: query strings stripped; annotation fields cleared"))

    # hidden text in content streams
    hidden = 0
    try:
        for page in pdf.pages:
            try:
                cs = page.Contents.read_bytes() if not isinstance(
                    page.Contents, pikepdf.Array) else b"".join(
                    c.read_bytes() for c in page.Contents)
            except Exception:
                continue
            hidden += len(re.findall(rb"\b3\s+Tr\b", cs))
            for m in re.findall(rb"/F\S+\s+([\d.]+)\s+Tf", cs):
                if float(m) < 0.5:
                    hidden += 1
    except Exception:
        pass
    if hidden:
        findings.append(_f("pdf-hidden-text", "tracking", "Invisible text runs",
                           "medium", 0.8,
                           f"{hidden} invisible/near-zero-size text operators in content streams",
                           rem=False,
                           removal="Report only — removing text runs risks altering the document"))

    if "/PieceInfo" in root:
        findings.append(_f("pdf-pieceinfo", "metadata", "Private /PieceInfo data",
                           "low", 1.0, str(list(root["/PieceInfo"].keys()))[:120],
                           removal="/PieceInfo deleted"))
    pm = sum(1 for page in pdf.pages if "/Metadata" in page)
    if pm:
        findings.append(_f("pdf-page-metadata", "metadata", "Per-page /Metadata",
                           "low", 1.0, f"{pm} pages carry /Metadata streams",
                           removal="Per-page /Metadata deleted"))
    pdf.close()
    return findings


def _pdf_dockind(_):
    return "document"


# ---------------- OOXML ----------------

def ooxml_kind(zf):
    names = set(zf.namelist())
    if any(n.startswith("word/") for n in names):
        return "docx"
    if any(n.startswith("xl/") for n in names):
        return "xlsx"
    if any(n.startswith("ppt/") for n in names):
        return "pptx"
    return None


def _ooxml_findings(data: bytes):
    findings = []
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()

    def read(n):
        try:
            return zf.read(n).decode("utf-8", "replace")
        except Exception:
            return ""

    core = read("docProps/core.xml")
    if core:
        ident = re.findall(r"<(?:dc:creator|cp:lastModifiedBy|dc:description)>([^<]{0,120})", core)
        times = re.findall(r"<(?:dcterms:created|dcterms:modified|cp:lastPrinted)[^>]*>([^<]{0,40})", core)
        if ident:
            findings.append(_f("owner-identity", "metadata", "Document author props",
                               "high", 1.0, "; ".join(ident[:5]),
                               removal="core.xml cleared (dc:title kept if present)"))
        if times:
            findings.append(_f("timestamps", "metadata", "Document timestamps",
                               "low", 1.0, "; ".join(times[:4]),
                               removal="created/modified/lastPrinted cleared"))
    app = read("docProps/app.xml")
    if app:
        soft = [(a, b) for a, b in re.findall(
            r"<(Application|AppVersion|Template|Company|Manager|TotalTime|HyperlinkBase)>([^<]{0,80})", app) if b.strip()]
        if soft:
            findings.append(_f("software-tags", "metadata", "App/company props",
                               "low", 1.0,
                               "; ".join(f"{a}={b}" for a, b in soft[:6]),
                               removal="app.xml fields removed"))
    custom = read("docProps/custom.xml")
    if custom:
        props = re.findall(r'name="([^"]{0,60})"', custom)
        findings.append(_f("custom-properties", "metadata", "Custom properties",
                           "medium", 1.0, ", ".join(props[:8]) or "custom.xml present",
                           removal="docProps/custom.xml deleted"))
    if "docProps/thumbnail.jpeg" in names or any(n.startswith("docProps/thumbnail") for n in names):
        findings.append(_f("thumbnail", "metadata", "Document thumbnail",
                           "low", 1.0, "docProps/thumbnail present",
                           removal="Thumbnail deleted"))
    if "word/people.xml" in names:
        ppl = re.findall(r'w:author="([^"]{0,60})"', read("word/people.xml"))
        findings.append(_f("people", "metadata", "People list (comment authors)",
                           "high", 1.0, ", ".join(ppl[:6]) or "people.xml present",
                           removal="word/people.xml deleted"))

    settings = read("word/settings.xml")
    ids = []
    if settings:
        for m in re.findall(r'w1[45]:(docId|paraId|textId)="([^"]{4,20})"', settings):
            ids.append(f"{m[0]}={m[1]}")
        nrsid = len(re.findall(r"<w:rsid\b", settings))
        if nrsid:
            ids.append(f"{nrsid} w:rsid entries")
    if "word/comments.xml" in names or "word/commentsExtended.xml" in names:
        cm = read("word/comments.xml")
        authors = [a for a in re.findall(r'w:author="([^"]{0,60})"', cm)
                   if a.strip()]
        dates = re.findall(r'w:date="[^"]+"', cm)
        if authors or dates:
            findings.append(_f("comments", "metadata", "Comments with authors/dates",
                               "medium", 1.0,
                               ", ".join(sorted(set(authors))[:5]) or "comments.xml present",
                               removal="Author names neutralized, dates dropped"))
    doc = read("word/document.xml")
    n_ins = 0
    for el in re.findall(r"<w:(?:ins|del)\b[^>]*>", doc):
        if re.search(r'\sw:date="', el):
            n_ins += 1
            continue
        m = re.search(r'\sw:author="([^"]*)"', el)
        if m and m.group(1).strip() and m.group(1) != "Author":
            n_ins += 1
    if n_ins:
        findings.append(_f("tracked-changes", "metadata", "Tracked changes",
                           "medium", 1.0, f"{n_ins} w:ins/w:del elements with author/date",
                           removal="Author names neutralized, dates dropped"))
    n_rsid = len(re.findall(r'\sw:rsid[A-Za-z]*="[^"]*"', doc))
    if n_rsid:
        ids.append(f"{n_rsid} rsid attributes in document.xml")
    if ids:
        findings.append(_f("unique-ids", "metadata", "Session/doc IDs",
                           "medium", 1.0, "; ".join(ids[:8]),
                           removal="rsid/docId values stripped"))

    remote = []
    for n in names:
        if n.endswith(".rels"):
            for m in re.findall(r'Target="([^"]+)"[^>]*TargetMode="External"|'
                                r'TargetMode="External"[^>]*Target="([^"]+)"',
                                read(n)):
                remote.append(m[0] or m[1])
    incp = re.findall(r"INCLUDEPICTURE[^<]{0,120}", doc)
    if remote or incp:
        findings.append(_f("remote-content", "tracking", "External/remote content",
                           "high", 0.95,
                           "; ".join((remote + incp)[:6]),
                           removal="External relationships removed (placeholder may remain)"))
    if any("vbaProject" in n for n in names):
        findings.append(_f("macros", "metadata", "Embedded macros", "medium", 1.0,
                           "vbaProject.bin present", rem=False,
                           removal="Report only — macros not auto-removed"))
    emb = [n for n in names if re.search(r"fonts?/.*\.(odttf|fntdata)", n)]
    if emb:
        findings.append(_f("embedded-fonts", "metadata", "Embedded fonts",
                           "low", 1.0, f"{len(emb)} obfuscated font parts",
                           rem=False, removal="Report only"))
    zf.close()
    return findings


def analyze_document(data: bytes, filename: str = "") -> dict:
    if data[:4] == b"%PDF":
        return {"kind": "document", "findings": _pdf_findings(data),
                "notes": [], "visual": None}
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        if ooxml_kind(zf):
            return {"kind": "document", "findings": _ooxml_findings(data),
                    "notes": [], "visual": None}
        zf.close()
    except zipfile.BadZipFile:
        pass
    return {"kind": "document", "findings": [],
            "notes": ["Unrecognized document container"], "visual": None}
