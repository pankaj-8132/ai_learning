"""AdaptiveTutor - document-grounded AI tutor with explainable recommendations.
Runs offline (BM25 retrieval + rule-based engines). Set ANTHROPIC_API_KEY to add LLM answers."""
import os, re, json, math, random, sqlite3, time, urllib.request
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "data.db")
app = Flask(__name__, static_folder=os.path.join(BASE, "static"))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED = {"pdf", "docx", "txt"}
STOP = set("a an the and or but if then of to in on at by for with from as is are was were be been being it its this that these those "
           "which who whom what when where why how can could should would may might will shall do does did not no yes so than too very "
           "also into over under about between such there their they them he she his her we you your our i have has had one more most "
           "other some any each all both only own same just".split())
WEAK, STRONG = 0.5, 0.75

# ---------- storage ----------
def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c
def sel(sql, a=()):
    c = db(); r = [dict(x) for x in c.execute(sql, a).fetchall()]; c.close(); return r
def ex(sql, a=()):
    c = db(); cur = c.execute(sql, a); c.commit(); i = cur.lastrowid; c.close(); return i
def init():
    c = db(); c.executescript("""
    CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, name TEXT, ext TEXT, text TEXT, ts REAL);
    CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, doc_id INT, idx INT, text TEXT, concept TEXT, prereq INT);
    CREATE TABLE IF NOT EXISTS chat(id INTEGER PRIMARY KEY, doc_id INT, role TEXT, content TEXT, sources TEXT, ts REAL);
    CREATE TABLE IF NOT EXISTS tests(id INTEGER PRIMARY KEY, doc_id INT, ts REAL);
    CREATE TABLE IF NOT EXISTS questions(id INTEGER PRIMARY KEY, doc_id INT, chunk_id INT, test_id INT, q TEXT, options TEXT, answer TEXT, sentence TEXT);
    CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, doc_id INT, chunk_id INT, question_id INT, test_id INT, correct INT, ts REAL);
    CREATE TABLE IF NOT EXISTS recs(doc_id INT, chunk_id INT, status TEXT, PRIMARY KEY(doc_id, chunk_id));
    """); c.commit(); c.close()

# ---------- text utils ----------
def tok(t): return [w for w in re.findall(r"[a-z][a-z0-9']+", t.lower()) if w not in STOP and len(w) > 2]
def sents(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", t)) if len(s.strip()) > 20]
def llm(prompt, system="You are a clear, patient tutor."):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key: return None
    try:
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            data=json.dumps({"model": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"), "max_tokens": 1500, "system": system,
                             "messages": [{"role": "user", "content": prompt}]}).encode())
        return json.load(urllib.request.urlopen(req, timeout=45))["content"][0]["text"]
    except Exception:
        return None

def extract(f, ext):
    if ext == "txt":
        return f.read().decode("utf-8", errors="replace")
    if ext == "pdf":
        from pypdf import PdfReader
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(f).pages)
    from docx import Document
    return "\n\n".join(p.text for p in Document(f).paragraphs)

def make_chunks(text):
    paras = [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text) if p.strip()]
    if len(paras) < 3: paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, buf, head = [], [], None
    def flush():
        nonlocal buf
        if buf: out.append((head, " ".join(buf))); buf = []
    for p in paras:
        p = re.sub(r"\s+", " ", p)
        if len(p.split()) <= 8 and not re.search(r"[.!?]$", p):
            flush(); head = p; continue
        buf.append(p)
        if sum(len(b.split()) for b in buf) >= 90: flush()
    flush()
    return out

def index_doc(doc_id, text):
    chunks = make_chunks(text)
    toks = [set(tok(c[1])) for c in chunks]; N = len(chunks)
    df = Counter(w for s in toks for w in s)
    ids = []
    for i, (head, t) in enumerate(chunks):
        tf = Counter(tok(t))
        top = sorted(tf, key=lambda w: -tf[w] * math.log(1 + N / df[w]))[:2]
        concept = head or " & ".join(w.title() for w in top) or f"Section {i+1}"
        pre = None
        if i:
            best = max(range(i), key=lambda j: len(toks[i] & toks[j]) / (1 + len(toks[j])) + (0.01 if j == i - 1 else 0))
            pre = ids[best] if toks[i] & toks[best] else ids[i - 1]
        ids.append(ex("INSERT INTO chunks(doc_id,idx,text,concept,prereq) VALUES(?,?,?,?,?)", (doc_id, i, t, concept, pre)))
    return N

# ---------- retrieval (BM25) ----------
def retrieve(doc_id, query, k=3):
    ch = sel("SELECT * FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,))
    docs = [tok(c["text"]) for c in ch]; N = len(ch) or 1
    avg = sum(map(len, docs)) / N or 1
    df = Counter(w for d in docs for w in set(d)); qt = list(dict.fromkeys(tok(query)))
    res = []
    for c, d in zip(ch, docs):
        tf = Counter(d); s = 0
        for w in qt:
            if w in tf:
                idf = math.log(1 + (N - df[w] + .5) / (df[w] + .5))
                s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (.25 + .75 * len(d) / avg))
        cov = sum(1 for w in qt if w in tf) / (len(qt) or 1)
        res.append((s, cov, c))
    res.sort(key=lambda x: -x[0])
    return res[:k]

def best_sentences(text, qt, n=2):
    ss = sents(text); sc = [(len(set(tok(s)) & set(qt)), -i, s) for i, s in enumerate(ss)]
    top = sorted(sc, reverse=True)[:n]
    return [s for _, _, s in sorted(top, key=lambda x: -x[1])]

# ---------- mastery ----------
def mastery(doc_id):
    out = {}
    for a in sel("SELECT chunk_id, correct FROM attempts WHERE doc_id=? ORDER BY id", (doc_id,)):
        out.setdefault(a["chunk_id"], []).append(a["correct"])
    res = {}
    for cid, h in out.items():
        res[cid] = {"mastery": round(.4 * sum(h) / len(h) + .6 * sum(h[-5:]) / len(h[-5:]), 3), "attempts": len(h),
                    "correct": sum(h), "recent": h[-3:]}
    return res

# ---------- routes ----------
@app.route("/")
def home(): return send_from_directory(app.static_folder, "index.html")

@app.errorhandler(413)
def too_big(e): return jsonify(error="File too large. Maximum size is 10 MB."), 413

@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename: return jsonify(error="No file selected."), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED: return jsonify(error=f"Unsupported type '.{ext}'. Upload PDF, DOCX or TXT."), 400
    try: text = extract(f, ext).strip()
    except ImportError as e: return jsonify(error=f"Missing library: {e}. Run pip install -r requirements.txt"), 500
    except Exception: return jsonify(error="Could not read this file. It may be corrupted or password-protected."), 400
    if len(text.split()) < 40:
        return jsonify(error="Almost no text found. Scanned/image-only PDFs are not supported (needs OCR)."), 400
    did = ex("INSERT INTO docs(name,ext,text,ts) VALUES(?,?,?,?)", (f.filename, ext, text, time.time()))
    return jsonify(id=did, sections=index_doc(did, text))

@app.get("/api/docs")
def docs(): return jsonify(sel("SELECT id,name,ext,length(text) AS chars FROM docs ORDER BY id DESC"))

@app.get("/api/doc/<int:d>")
def doc(d):
    r = sel("SELECT name,text FROM docs WHERE id=?", (d,))
    if not r: return jsonify(error="Document not found"), 404
    return jsonify(name=r[0]["name"], text=r[0]["text"], chunks=sel("SELECT id,idx,concept,prereq FROM chunks WHERE doc_id=? ORDER BY idx", (d,)))

@app.post("/api/ask")
def ask():
    j = request.json; d, qn = j["doc_id"], j["question"].strip()
    if not qn: return jsonify(error="Please type a question."), 400
    hist = sel("SELECT role,content FROM chat WHERE doc_id=? ORDER BY id DESC LIMIT 4", (d,))[::-1]
    ex("INSERT INTO chat(doc_id,role,content,sources,ts) VALUES(?,?,?,?,?)", (d, "user", qn, "[]", time.time()))
    hits = retrieve(d, qn)
    if not hits or hits[0][1] < .4 or hits[0][0] <= 0:
        ans, src, grounded = ("I couldn't find an answer to that in this document, so I won't guess. Try rephrasing with terms from the material, "
                              "or ask about a topic it covers."), [], False
    else:
        src = [{"chunk_id": c["id"], "concept": c["concept"], "excerpt": c["text"][:350]} for s, cv, c in hits if s > 0]
        ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, (s, cv, c) in enumerate(hits) if s > 0)
        h = "\n".join(f"{m['role']}: {m['content']}" for m in hist)
        a = llm(f"Answer ONLY from the context. If it is not there, reply exactly NOT_IN_DOCUMENT. Give: a simple explanation, one example, "
                f"a one-line summary. Cite sources like [1].\n\nContext:\n{ctx}\n\nHistory:\n{h}\n\nQuestion: {qn}")
        grounded = True
        if a and "NOT_IN_DOCUMENT" in a:
            ans, src, grounded = "The document doesn't appear to cover this, so I can't answer from the material.", [], False
        elif a: ans = a
        else:
            qt = tok(qn); sn = best_sentences(hits[0][2]["text"], qt, 2)
            ex_s = next((s for _, _, c in hits for s in sents(c["text"]) if re.search(r"for example|e\.g\.|such as|for instance", s, re.I)), None)
            ans = "Simple explanation:\n" + " ".join(sn) + (f"\n\nExample from the material:\n{ex_s}" if ex_s else "") + \
                  f"\n\nSummary: this is covered in \"{hits[0][2]['concept']}\"."
    ex("INSERT INTO chat(doc_id,role,content,sources,ts) VALUES(?,?,?,?,?)", (d, "assistant", ans, json.dumps(src), time.time()))
    return jsonify(answer=ans, sources=src, grounded=grounded)

@app.get("/api/history/<int:d>")
def history(d):
    r = sel("SELECT role,content,sources FROM chat WHERE doc_id=? ORDER BY id", (d,))
    for x in r: x["sources"] = json.loads(x["sources"])
    return jsonify(r)

@app.post("/api/quiz")
def quiz():
    j = request.json; d = j["doc_id"]; n = min(int(j.get("n", 8)), 20)
    ch = sel("SELECT * FROM chunks WHERE doc_id=?", (d,))
    if not ch: return jsonify(error="Document has no content."), 404
    ms = mastery(d); N = len(ch)
    toks = {c["id"]: set(tok(c["text"])) for c in ch}; df = Counter(w for s in toks.values() for w in s)
    pool = [w for w in df if len(w) >= 5]
    cand = []
    for c in ch:
        for s in sents(c["text"]):
            if 8 <= len(s.split()) <= 35:
                kws = [w for w in tok(s) if len(w) >= 5 and w in re.sub(r"[^a-z ]", " ", s.lower()).split()]
                if kws: cand.append((c, s, max(kws, key=lambda w: math.log(1 + N / df[w]))))
    if not cand: return jsonify(error="Not enough sentence-level content to build questions."), 400
    w = [1.5 - ms.get(c["id"], {"mastery": .5})["mastery"] for c, s, k in cand]  # adaptive: weak concepts first
    chosen, seen = [], set()
    for _ in range(n * 6):
        if len(chosen) >= n: break
        c, s, k = random.choices(cand, w)[0]
        if s not in seen: seen.add(s); chosen.append((c, s, k))
    tid = ex("INSERT INTO tests(doc_id,ts) VALUES(?,?)", (d, time.time())); qs = []
    for c, s, k in chosen:
        dis = random.sample([x for x in pool if x != k and x not in toks[c["id"]]] or [x for x in pool if x != k], min(3, max(len(pool) - 1, 0)))
        if len(dis) < 3: continue
        opts = dis + [k]; random.shuffle(opts)
        qtxt = re.sub(re.escape(k), "_____", s, count=1, flags=re.I)
        qid = ex("INSERT INTO questions(doc_id,chunk_id,test_id,q,options,answer,sentence) VALUES(?,?,?,?,?,?,?)",
                 (d, c["id"], tid, qtxt, json.dumps(opts), k, s))
        qs.append({"id": qid, "q": qtxt, "options": opts, "concept": c["concept"]})
    return jsonify(test_id=tid, questions=qs)

@app.post("/api/submit")
def submit():
    j = request.json; tid = j["test_id"]; ans = j["answers"]; out = []; ok = 0
    for q in sel("SELECT * FROM questions WHERE test_id=?", (tid,)):
        mine = ans.get(str(q["id"])); good = int(mine == q["answer"]); ok += good
        ex("INSERT INTO attempts(doc_id,chunk_id,question_id,test_id,correct,ts) VALUES(?,?,?,?,?,?)", (q["doc_id"], q["chunk_id"], q["id"], tid, good, time.time()))
        r = {"q": q["q"], "yours": mine, "answer": q["answer"], "correct": bool(good)}
        if not good:
            c = sel("SELECT concept,text FROM chunks WHERE id=?", (q["chunk_id"],))[0]
            r["concept"] = c["concept"]
            steps = (f"1. Read the sentence and note the missing word's role (it completes: \"{q['sentence'][:120]}...\").\n"
                     f"2. Go back to the section '{c['concept']}' and find the idea this sentence states.\n"
                     f"3. Correct answer: '{q['answer']}'. Source: \"{q['sentence']}\"\n"
                     f"4. Why '{mine}' is wrong: it does not fit the definition given in the passage.")
            r["how_to_solve"] = llm(f"A student answered '{mine}' to this fill-in question: \"{q['q']}\" (correct: '{q['answer']}'). "
                                    f"Using ONLY this passage, explain the concept and show step by step how to solve it.\n\nPassage: {c['text']}") or steps
        out.append(r)
    return jsonify(score=ok, total=len(out), results=out)

def analysis_data(d):
    ch = sel("SELECT * FROM chunks WHERE doc_id=? ORDER BY idx", (d,)); ms = mastery(d); rows = []
    for c in ch:
        m = ms.get(c["id"])
        st = "untested" if not m else "weak" if m["mastery"] < WEAK else "developing" if m["mastery"] < STRONG else "strong"
        rows.append({**c, "mastery": m["mastery"] if m else None, "attempts": m["attempts"] if m else 0,
                     "correct": m["correct"] if m else 0, "recent": m["recent"] if m else [], "status": st})
    tests = sel("SELECT test_id, SUM(correct) c, COUNT(*) n FROM attempts WHERE doc_id=? GROUP BY test_id ORDER BY test_id", (d,))
    return rows, [{"test": t["test_id"], "pct": round(100 * t["c"] / t["n"])} for t in tests]

@app.get("/api/analysis/<int:d>")
def analysis(d):
    rows, hist = analysis_data(d)
    weak = [r for r in rows if r["status"] == "weak"]
    wrong = {r["id"]: [q["sentence"] for q in sel("SELECT q.sentence FROM attempts a JOIN questions q ON q.id=a.question_id WHERE a.chunk_id=? AND a.correct=0", (r["id"],))] for r in weak}
    why = [f"'{r['concept']}': only {r['correct']}/{r['attempts']} correct. Missed ideas: " + " | ".join(wrong[r["id"]][:2]) for r in weak]
    trend = None
    if len(hist) >= 2: trend = hist[-1]["pct"] - hist[-2]["pct"]
    return jsonify(concepts=rows, history=hist, why_low=why, trend=trend)

@app.get("/api/recs/<int:d>")
def recs(d):
    rows, _ = analysis_data(d); byid = {r["id"]: r for r in rows}; saved = {r["chunk_id"]: r["status"] for r in sel("SELECT * FROM recs WHERE doc_id=?", (d,))}
    blocks = Counter(r["prereq"] for r in rows if r["prereq"] and r["status"] in ("weak", "developing"))
    out = []
    for r in rows:
        if r["status"] in ("strong",): continue
        m = r["mastery"]; ev = []; pr = (1 - (m if m is not None else .5)) * 60
        if m is None:
            ev.append("No assessment data yet for this section."); act = "Read the section, then take a diagnostic quiz."
        else:
            ev.append(f"Answered {r['correct']}/{r['attempts']} questions correctly (mastery {int(m*100)}%).")
            if r["recent"]: ev.append("Last answers: " + " ".join("✓" if x else "✗" for x in r["recent"]) + ("  (recent decline)" if len(r["recent"]) >= 2 and r["recent"][-1] == 0 and r["recent"][-2] == 0 else ""))
            if r["recent"] and sum(r["recent"][-2:]) == 0: pr += 10
            act = ("Study the full lesson (basic→advanced), then retake 5 questions." if m < WEAK else "Review the summary and do a targeted practice quiz.")
        p = byid.get(r["prereq"])
        if p and p["mastery"] is not None and p["mastery"] < WEAK:
            ev.append(f"Prerequisite '{p['concept']}' is weak ({int(p['mastery']*100)}%) — fix it first."); pr += 15; act = f"Complete '{p['concept']}' first. " + act
        if blocks[r["id"]]: ev.append(f"{blocks[r['id']]} later section(s) build on this."); pr += 8
        if m is None: pr = min(pr, 25)
        pr = round(pr); label = "High" if pr >= 45 else "Medium" if pr >= 25 else "Low"
        ss = sents(r["text"]); fq = f"In your own words, explain '{r['concept']}' and give one example from the material." if not ss else \
             f"Why does this matter: \"{ss[0][:110]}...\"? Explain it without looking at the text."
        out.append({"chunk_id": r["id"], "concept": r["concept"], "reason": f"Section is {r['status']}" + (f" (mastery {int(m*100)}%)" if m is not None else ""),
                    "evidence": ev, "activity": act, "priority": label, "score": pr, "status": saved.get(r["id"], "pending"), "follow_up": fq, "idx": r["idx"]})
    out.sort(key=lambda x: -x["score"])
    roadmap = [{"step": i + 1, "chunk_id": x["chunk_id"], "concept": x["concept"], "priority": x["priority"], "status": x["status"]}
               for i, x in enumerate(sorted([x for x in out if x["score"] >= 25 or x["status"] == "done"], key=lambda x: x["idx"]))]
    return jsonify(recommendations=out, roadmap=roadmap)

@app.post("/api/recs/<int:d>/<int:c>/status")
def rec_status(d, c):
    ex("INSERT OR REPLACE INTO recs(doc_id,chunk_id,status) VALUES(?,?,?)", (d, c, request.json.get("status", "done"))); return jsonify(ok=True)

@app.get("/api/lesson/<int:d>/<int:c>")
def lesson(d, c):
    r = sel("SELECT * FROM chunks WHERE id=? AND doc_id=?", (c, d))
    if not r: return jsonify(error="Not found"), 404
    r = r[0]; pre = sel("SELECT concept,text FROM chunks WHERE id=?", (r["prereq"],)) if r["prereq"] else []
    nxt = sel("SELECT concept,text FROM chunks WHERE doc_id=? AND prereq=? LIMIT 2", (d, c)); ss = sents(r["text"])
    gen = llm(f"Teach the concept '{r['concept']}' from basic to advanced using ONLY this material. Use sections: Prerequisite check, Basics, Core ideas, "
              f"Advanced connections, Worked example, Check yourself (2 questions).\n\nMaterial:\n{r['text']}\n\nPrerequisite: {pre[0]['text'] if pre else 'none'}\n"
              f"Next topics: {' '.join(n['text'][:300] for n in nxt)}")
    if gen: return jsonify(concept=r["concept"], lesson=gen)
    parts = [f"# {r['concept']}"]
    if pre: parts.append(f"## 0. Prerequisite check\nBefore this, make sure you know '{pre[0]['concept']}':\n{' '.join(sents(pre[0]['text'])[:2])}")
    parts.append("## 1. Basics\n" + " ".join(ss[:2]))
    parts.append("## 2. Core ideas\n" + " ".join(ss[2:6] or ss[:1]))
    if nxt: parts.append("## 3. Advanced — where this leads\n" + "\n".join(f"• {n['concept']}: {' '.join(sents(n['text'])[:1])}" for n in nxt))
    parts.append(f"## Check yourself\nExplain '{r['concept']}' in your own words, then retake a quiz.")
    return jsonify(concept=r["concept"], lesson="\n\n".join(parts))

if __name__ == "__main__":
    init(); print("Open http://127.0.0.1:5000  |  LLM mode:", "ON" if os.getenv("ANTHROPIC_API_KEY") else "OFF (offline rule-based mode)")
    app.run(debug=False, port=5000)
