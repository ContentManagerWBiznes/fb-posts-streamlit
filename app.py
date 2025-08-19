# app.py — Streamlit UI for your FB post generator (Polish output)

import io
import os
import json
import re
import random
import pandas as pd
import streamlit as st
from collections import Counter
from difflib import SequenceMatcher  
from openai import OpenAI

# ===================== CONFIG / SECRETS =====================
st.set_page_config(page_title="FB Post Generator", page_icon="🧡", layout="centered")
st.title("FB Post Generator 🧡")
st.caption("Wgraj dwa pliki CSV (historia i tematy), a aplikacja zwróci plik **posty_na_fb.csv**.")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Brakuje OPENAI_API_KEY. Dodaj sekret w Streamlit Cloud → App → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

CANDIDATE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
N_SHOTS = 40

# ====================== SYSTEM PROMPT (English instructions, Polish output) ======================
SYSTEM = """
You are Dr. Bożena Chocielowska, a compassionate, experienced gynecologist.
Write all outputs in natural, empathetic Polish only.

OBJECTIVE
- Create Facebook posts that educate, calm, and warmly invite readers to book a visit — never diagnose online.
- Each post must be between 700–1700 characters.

VOICE & EMPATHY
- Warm, supportive, respectful. Include at least one sentence that explicitly shows empathy (e.g., “Wiem, że…”, “Rozumiem, że…”).
- Speak directly to one reader (“Ty”), in simple, friendly language; explain difficult terms in one short sentence.
- Use emojis naturally: minimum 1 in the hook and minimum 1–2 in the main text.

STRUCTURE (Polish output; allow subtle variation so posts don’t feel templated):
1) Hook — one punchy sentence with an emoji; rotate styles across the batch (question, myth-busting, metaphor, vivid scene, micro-anecdote, reference to the practice). Avoid starting more than 2 posts in a batch with the same style.
2) Main content — 3–7 sentences: a clear fact, gentle advice, normalization of a symptom; everyday references are welcome.
3) CTA — exactly one sentence that invites the reader to **book an appointment**, explicitly tied to the topic (use the topic words naturally). No exclamation marks, do not repeat the phone number (it’s in the signature), keep it warm and concrete, and avoid duplicate phrasing across the batch.
4) Permanent signature — always exactly:
🌍 www.gabinetginekologiczny.com.pl
☎️ 502 552 371
5) Hashtags — 3–6 unique, topic-relevant Polish hashtags (avoid English unless unavoidable). Avoid repeating the same set across the batch.
6) Visual hint — a detailed, unique image suggestion for the designer: include scene, main subject, type of shot (close-up / medium / long), mood, color palette, season or time of day, and optional overlay text. No generic descriptions.

STYLE RULES
- Vary rhythm and sentence openings across posts.
- Avoid repetitive phrasing.
- No direct diagnoses or promises.
- Keep a calm, hopeful tone; never shame or alarm.

OUTPUT FORMAT (strict JSON, Polish values):
{
  "hook": "…",
  "post_text": "… (already includes CTA and signature at the end)",
  "hashtags": ["#tag1","#tag2","#tag3"],
  "visual_hint": "detailed scene description"
}
"""

# =================== UPLOADERS ===================
with st.expander("Wymagany format CSV", expanded=False):
    st.markdown("""
**history.csv** musi mieć nagłówki:
- `Data publikacji`, `Treść posta (tekst)`, `Temat/kategoria`, `Liczba reakcji`, `Liczba komentarzy`, `Liczba udostępnień`

**tematy.csv**:
- `data`, `temat`, (opcjonalnie) `kontekst`

Kodowanie: UTF-8 (najlepiej UTF-8 z BOM).
""")

hist_up = st.file_uploader("Wgraj *history.csv*", type=["csv"])
topics_up = st.file_uploader("Wgraj *tematy.csv*", type=["csv"])

# =================== HELPERS ===================
def call_chat(messages, max_tokens=1000):
    """Fallback przez listę modeli."""
    last = None
    for m in CANDIDATE_MODELS:
        try:
            resp = client.chat.completions.create(
                model=m, messages=messages, temperature=0.8, max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            last = e
            continue
    raise last

def safe_json_loads(text):
    start = text.find("{"); end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end+1])
    return json.loads(text)

BAN_STARTS_RE = re.compile(r'^\s*(wiem, że|rozumiem, że)\b', re.IGNORECASE)
def split_first_sentence(full_text: str):
    s = (full_text or "").strip()
    if not s: return "", ""
    parts = re.split(r'(?<=\.|!|\?|…)\s+', s, maxsplit=1)
    return (parts[0], "" if len(parts)==1 else parts[1])

def starts_with_banned(sentence: str) -> bool:
    return bool(BAN_STARTS_RE.search(sentence or ""))

ALLOWED_OPENERS = [
    "Czasem łatwo to odkładać…",
    "Bywa, że to temat pełen pytań.",
    "W gabinecie często słyszę podobne wątpliwości.",
    "Zdarza się, że pierwszym krokiem jest po prostu rozmowa.",
    "Dla wielu kobiet to ważny, ale niełatwy moment.",
    "Jeśli ten temat wraca do Ciebie co jakiś czas, to normalne.",
    "Kiedy myślimy o zdrowiu, liczy się czułość i małe kroki."
]

def paraphrase_intro_for_variety(topic: str, context: str, sentence: str, used_starts):
    prompt = f"""Rewrite this ONE Polish sentence so it uses a DIFFERENT opening (different first 3 words)
than the ones already used:
{chr(10).join('• ' + s for s in sorted(set(used_starts)))}
Keep meaning, empathy, and natural flow. Do not use “Wiem, że” or “Rozumiem, że”.
Return exactly ONE sentence in Polish, no quotes.
Topic: {topic}
Context: {context}
Sentence to rewrite: {sentence}"""
    out = call_chat(
        [{"role":"system","content":"Rewrite a single sentence in Polish with a different opener; avoid 'Wiem, że'/'Rozumiem, że'. Return one sentence only."},
         {"role":"user","content": prompt}],
        max_tokens=80
    )
    return (out or "").strip().replace("\n"," ")

SIG_LINE1 = "🌍 www.gabinetginekologiczny.com.pl"
SIG_LINE2 = "☎️ 502 552 371"

def extract_cta(text):
    s = str(text).rstrip()
    idx = s.rfind(SIG_LINE1)
    if idx == -1: return s, "", ""
    before = s[:idx].rstrip()
    signature = s[idx:].strip()
    parts = [ln for ln in before.split("\n") if ln.strip()]
    cta = parts[-1] if parts else ""
    body = "\n".join(parts[:-1]) if len(parts) > 1 else ""
    return body, cta, signature

APPT_VERB_HINT = "umów się|zapisz się|zarezerwuj termin|porozmawiajmy w gabinecie|umów konsultację"

def paraphrase_cta(topic, angle, cta, used_ctas):
    prompt = f"""Write ONE CTA sentence in Polish that invites the reader to book an appointment.
Requirements:
- directly invite to book (use a natural variant of: {APPT_VERB_HINT}),
- clearly tie to this topic/angle: {angle or topic},
- no phone number (it's in the signature), no exclamation marks,
- warm, specific, 7–16 words,
- avoid repeating any of these previous CTA lines:
{chr(10).join('• ' + c for c in sorted(set(used_ctas)) if c)}

Return only the new CTA sentence in Polish, no quotes."""
    out = call_chat(
        [{"role":"system","content":"Polish CTA writer. Return ONE Polish sentence inviting to book; no '!' and no phone; tie to the topic; no quotes."},
         {"role":"user","content": prompt}],
        max_tokens=90
    )
    return (out or "").strip().replace("\n"," ")

APPT_RE = re.compile(r"\b(umów\w*|zapisz\w*|zarezerwuj\w*|porozmawiajmy|konsultuj\w*|konsultacj\w*|wizyt\w*)\b", re.IGNORECASE)

def ensure_appointment_cta(topic, angle, cta, used_ctas):
    if not APPT_RE.search(cta or ""):
        return paraphrase_cta(topic, angle, cta or "(brak)", used_ctas)
    return re.sub(r"!+$", ".", cta).strip()

SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\?…])\s+')
def split_sentences_pl(s: str):
    return [x.strip() for x in re.split(SENT_SPLIT_RE, s or "") if x and x.strip()]
def split_paragraphs(s: str):
    return [p.strip() for p in str(s).split("\n") if p.strip()]

def isolate_cta_sentence(body: str, cta_line: str):
    body = body or ""; cta_line = (cta_line or "").strip()
    paras = split_paragraphs(body)
    source = cta_line if cta_line else (paras[-1] if paras else "")
    sents = split_sentences_pl(source)
    idxs = [i for i, s in enumerate(sents) if APPT_RE.search(s)]
    if idxs:
        chosen_i = idxs[-1]; chosen = sents[chosen_i].strip()
        remainder = " ".join(s for i, s in enumerate(sents) if i != chosen_i).strip()
    else:
        if len(sents) >= 2:
            chosen = sents[-1].strip(); remainder = " ".join(sents[:-1]).strip()
        else:
            chosen = cta_line; remainder = ""
    new_body = body
    if cta_line:
        if remainder:
            new_body = (body.rstrip() + ("\n\n" + remainder if body.strip() else remainder)).strip()
    else:
        if paras:
            last = paras[-1]
            last_new = last.replace(chosen, "", 1).strip()
            last_new = re.sub(r'\s{2,}', ' ', last_new)
            new_body = "\n\n".join(paras[:-1] + ([last_new] if last_new else [])).strip()
    return new_body, chosen

def enrich_visual_hint(vh_text, topic, body):
    ask = f"""Improve the 'visual hint' into a concrete brief for the designer.
Include: scene, main subject, shot type (close/medium/long), mood, color palette, time of day/season, and an optional 2–4 word overlay text.
Return 1–2 sentences in Polish.

Temat: {topic}
Post summary: {body[:400]}
Current visual hint: {vh_text}"""
    out = call_chat(
        [{"role":"system","content":"You are a (Polish-output) art director creating concise visual briefs. Return Polish."},
         {"role":"user","content": ask}],
        max_tokens=150
    )
    return out.strip().replace("\n"," ")

def fix_hashtags(existing, topic, body):
    items = [x.strip() for x in str(existing).split(",") if x.strip() and x.strip().startswith("#")]
    if len(items) >= 3:
        return items[:6]
    ask = f"""Suggest 3–6 short Polish hashtags (no spaces, with #) that fit this post.
Temat: {topic}
Post summary: {body[:300]}
Return ONLY a JSON array, e.g., ["#profilaktyka","#zdrowiekobiety"]."""
    out = call_chat(
        [{"role":"system","content":"Return only a valid JSON array of Polish hashtags."},
         {"role":"user","content": ask}],
        max_tokens=80
    )
    try:
        arr = safe_json_loads(out)
        arr = [h for h in arr if isinstance(h,str) and h.startswith("#")]
        return arr[:6] if len(arr)>=3 else items
    except:
        return items

def needs_enrichment(vh):
    s = (vh or "").lower()
    tests = [
        any(k in s for k in ["zbliżenie","średni","daleki","close","medium","long"]),
        any(k in s for k in ["ciepły","spokojny","delikatny","energetyczny","nostalgiczny","pogodny"]),
        any(k in s for k in ["pastel","kontrast","kolor","paleta","barw"]),
        any(k in s for k in ["poranek","wieczór","noc","wiosna","lato","jesień","zima"]),
    ]
    return sum(tests) < 2 or len(s) < 80

def first3_words(text):
    return " ".join(re.findall(r"\w+|\S", str(text).strip())[:3]).lower()

EMOJI_RE = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
def normalize_line(s: str) -> str:
    s = EMOJI_RE.sub('', str(s).lower())
    return re.sub(r'\s+', ' ', s).strip()

def ensure_body_emoji(body: str) -> str:
    if EMOJI_RE.search(body or ""): return body
    prompt = f"""You will lightly revise the following Polish paragraph to add exactly ONE emoji in a natural, mid-sentence placement.
Do NOT add a CTA or a signature, and do NOT change the meaning or length too much. Return only the revised paragraph in Polish.
Paragraph:
{body}"""
    out = call_chat(
        [{"role":"system","content":"Polish paragraph micro-editor. Insert exactly one emoji naturally (not at the very start or end). Return only the revised paragraph in Polish."},
         {"role":"user","content": prompt}],
        max_tokens=160
    )
    return (out or body).strip()

def expand_body_minimally(topic: str, context: str, body: str, desired_extra_chars: int = 120) -> str:
    prompt = f"""Add ONE or TWO Polish sentences to the following paragraph to gently extend it.
Keep the tone empathetic and consistent; keep facts correct; tie naturally to the topic: {topic} ({context}).
Do NOT add a CTA or signature. Keep it concise and natural.
Return only the updated paragraph in Polish.
Paragraph:
{body}"""
    out = call_chat(
        [{"role":"system","content":"Polish paragraph extender. Add 1–2 sentences, keep tone and coherence. No CTA/signature. Return only the updated paragraph."},
         {"role":"user","content": prompt}],
        max_tokens=220
    )
    return (out or body).strip()

def compress_body_minimally(topic: str, body: str, target_len: int) -> str:
    prompt = f"""Rewrite this Polish paragraph to be slightly shorter (about {target_len} characters) by removing redundancy.
Keep empathy, key facts, and any existing emojis. Do NOT add a CTA or signature.
Return only the shortened paragraph in Polish.
Paragraph:
{body}"""
    out = call_chat(
        [{"role":"system","content":"Polish paragraph compressor. Gently shorten to target length; preserve meaning/empathy/emojis. No CTA/signature. Return only the paragraph."},
         {"role":"user","content": prompt}],
        max_tokens=220
    )
    return (out or body).strip()

def assembled_length(body: str, cta: str, signature: str) -> int:
    return len((str(body).strip() + "\n\n" + str(cta).strip() + "\n" + str(signature)).strip())

EDITORIAL_ERROR_RE = re.compile(
    r"(Proszę .*tekst|Przykro mi|nie dostarczyłeś|podaj tekst|Provide the text|Return only|Give me the text|Wprowadź tekst|Podaj treść)",
    re.IGNORECASE
)
def looks_like_editorial_error(s: str) -> bool:
    return bool(EDITORIAL_ERROR_RE.search(s or ""))

def synthesize_body(topic: str, context: str, desired_chars: int = 800) -> str:
    prompt = f"""Write the MAIN BODY ONLY in Polish (no CTA, no signature), 3–7 sentences, about {desired_chars} characters.
Keep it warm, empathetic, informative; include at least one emoji naturally (not at start/end).
Tie clearly to the topic and context.
Topic: {topic}
Context: {context}"""
    out = call_chat(
        [{"role":"system","content":"Write only the Polish paragraph (main body). No CTA, no signature."},
         {"role":"user","content": prompt}],
        max_tokens=400
    )
    return (out or "").strip()

DUP_ARTEFACT_RE = re.compile(
    r"(?:^|\n)(?:Paragraph:|Paragraf:|Hook:|Temat:|Topic:|Context:|Kontekst:|CTA:|Podpis:|Signature:|Return only.*?|Wersja:|Oryginał:|Before:|After:|Przed:|Po:|Tekst:)",
    re.IGNORECASE
)
def remove_signature_blocks(text: str) -> str:
    s = str(text).replace(SIG_LINE1, "").replace(SIG_LINE2, "")
    return re.sub(r'\n{3,}', '\n\n', s).strip()

def remove_trailing_cta(body: str, cta: str) -> str:
    b = (body or "").rstrip(); c = (cta or "").strip()
    if not b or not c: return b
    lines = [ln for ln in b.split("\n") if ln.strip()]
    if lines and normalize_line(lines[-1]) == normalize_line(c):
        lines = lines[:-1]
    return "\n".join(lines).strip()

def collapse_duplicate_blocks(body: str) -> str:
    s = (body or "").strip()
    if len(s) < 200: return s
    mid = len(s)//2
    left, right = s[:mid].strip(), s[mid:].strip()
    if left and right and (left == right):  # identyczne połówki
        return left
    paras = [p.strip() for p in s.split("\n\n") if p.strip()]
    deduped, seen = [], set()
    for p in paras:
        if p not in seen:
            deduped.append(p); seen.add(p)
    return "\n\n".join(deduped).strip()

def _jaccard_char_ngrams(a: str, b: str, n: int = 3) -> float:
    a = a or ""; b = b or ""
    sa = {a[i:i+n] for i in range(max(0, len(a)-n+1))}
    sb = {b[i:i+n] for i in range(max(0, len(b)-n+1))}
    denom = len(sa | sb) or 1
    return len(sa & sb) / denom

def collapse_near_duplicate_halves(body: str, threshold: float = 0.82) -> str:
    s = (body or "").strip()
    if len(s) < 240: return s
    mid = len(s)//2
    left, right = s[:mid].strip(), s[mid:].strip()
    if not left or not right: return s
    try:
        score = SequenceMatcher(None, left, right).ratio()
    except Exception:
        score = _jaccard_char_ngrams(left, right, n=3)
    if score >= threshold:
        return left if len(left) >= len(right) else right
    return s

def strip_editorial_artefacts(body: str) -> str:
    s = DUP_ARTEFACT_RE.sub("\n", body or "").strip()
    return re.sub(r'\n{3,}', '\n\n', s)

def sanitize_body(body: str, cta: str) -> str:
    s = remove_signature_blocks(body)
    s = remove_trailing_cta(s, cta)
    s = strip_editorial_artefacts(s)
    s = collapse_duplicate_blocks(s)
    s = collapse_near_duplicate_halves(s, threshold=0.82)
    return s.strip()

def rebuild_post_text(raw_final_text: str, cta: str) -> str:
    body, _, _ = extract_cta(raw_final_text)
    body = sanitize_body(body, cta)
    body = ensure_body_emoji(body)
    body = sanitize_body(body, cta)
    signature = f"{SIG_LINE1}\n{SIG_LINE2}"
    return (body.strip() + "\n\n" + cta.strip() + "\n" + signature).strip()

# =================== GŁÓWNA FUNKCJA GENERUJĄCA ===================
def generate_posts(history_df: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    # (historia nie jest teraz aktywnie używana; można tu dodać few-shots)
    rows = []
    used_ctas, used_ctas_norm = set(), set()
    hook_counter = Counter()
    first_sentence_counter = Counter()

    for _, r in topics.iterrows():
        length = random.randint(700, 1700)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Temat: {r['temat']}\nKontekst: {r.get('kontekst','')}\nDługość: ok. {length} znaków."}
        ]
        raw = call_chat(messages, max_tokens=1200)
        post = safe_json_loads(raw)

        # FIRST SENTENCE variety
        orig_text = post.get("post_text", "").strip()
        first, rest = split_first_sentence(orig_text)
        need_rewrite = starts_with_banned(first)
        intro_start = first3_words(first)
        if first_sentence_counter[intro_start] >= 2:
            need_rewrite = True
        if need_rewrite:
            used_starts = [k for k,_ in first_sentence_counter.most_common(12)]
            fixed_first = paraphrase_intro_for_variety(r['temat'], r.get('kontekst',''), first, used_starts)
            if fixed_first: first = fixed_first
        post["post_text"] = (first + (" " + rest if rest else "")).strip()
        intro_start = first3_words(first)
        first_sentence_counter[intro_start] += 1

        hook = post.get("hook","").strip()
        text = post.get("post_text","").strip()
        hashtags = post.get("hashtags", [])
        visual_hint = post.get("visual_hint","").strip()

        # HOOK variety
        start = first3_words(hook)
        hook_counter[start] += 1
        if hook_counter[start] > 2:
            ask = f"""Rewrite this HOOK in Polish so it has a different opening (different first 3 words),
keep meaning and vibe, and include at least one emoji. Return one sentence.
HOOK to rewrite: {hook}"""
            new_hook = call_chat(
                [{"role":"system","content":"Rewrite Polish hooks with variety; return a single sentence with at least one emoji."},
                 {"role":"user","content": ask}],
                max_tokens=60
            ).strip().replace("\n"," ")
            hook = new_hook
            start = first3_words(hook)
            hook_counter[start] += 1

        # CTA & BODY
        body_no_cta, cta, signature = extract_cta(text)
        body_no_cta, cta = isolate_cta_sentence(body_no_cta, cta)

        if (not body_no_cta.strip()) or (len(body_no_cta) < 180) or looks_like_editorial_error(body_no_cta):
            body_no_cta = synthesize_body(r['temat'], r.get('kontekst',''), desired_chars=850)

        body_no_cta = sanitize_body(body_no_cta, cta)

        if not signature or SIG_LINE1 not in signature or SIG_LINE2 not in signature:
            signature = f"{SIG_LINE1}\n{SIG_LINE2}"

        angle_hint = r.get('kontekst') or r['temat']
        if not cta or normalize_line(cta) in used_ctas_norm:
            cta = paraphrase_cta(r['temat'], angle_hint, cta or "(brak)", used_ctas)
        cta = ensure_appointment_cta(r['temat'], angle_hint, cta, used_ctas)
        used_ctas.add(cta); used_ctas_norm.add(normalize_line(cta))

        body_no_cta = ensure_body_emoji(body_no_cta)
        body_no_cta = sanitize_body(body_no_cta, cta)

        total_len = assembled_length(body_no_cta, cta, signature)
        MIN_L, MAX_L = 700, 1700
        if total_len < MIN_L:
            body_no_cta = expand_body_minimally(r['temat'], r.get('kontekst',''), body_no_cta,
                                                desired_extra_chars=min(200, MIN_L - total_len + 60))
            body_no_cta = ensure_body_emoji(body_no_cta)
            body_no_cta = sanitize_body(body_no_cta, cta)
        elif total_len > MAX_L:
            overhead = len(("\n\n" + cta + "\n" + signature).strip())
            target_body_len = max(600, MAX_L - overhead - 5)
            body_no_cta = compress_body_minimally(r['temat'], body_no_cta, target_body_len)
            body_no_cta = sanitize_body(body_no_cta, cta)

        raw_final_text = (body_no_cta.strip() + "\n\n" + cta.strip() + "\n" + signature).strip()
        final_text = rebuild_post_text(raw_final_text, cta)

        if needs_enrichment(visual_hint):
            visual_hint = enrich_visual_hint(visual_hint, r['temat'], body_no_cta)

        if isinstance(hashtags, str):
            try: hashtags = safe_json_loads(hashtags)
            except: hashtags = [h.strip() for h in hashtags.split(",") if h.strip()]
        hashtags = [h for h in hashtags if isinstance(h,str) and h.startswith("#")]
        if len(hashtags) < 3 or len(hashtags) > 6:
            hashtags = fix_hashtags(hashtags, r['temat'], body_no_cta)

        rows.append({
            "data": r['data'],
            "temat": r['temat'],
            "hook": hook,
            "post_text": final_text,
            "hashtags": ", ".join(hashtags[:6]),
            "visual_hint": visual_hint
        })

    return pd.DataFrame(rows)

# =================== RUN BUTTON ===================
def read_csv_safely(file):
    if file is None: return None, "Brak pliku."
    last_err = ""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc), None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, f"Nie udało się odczytać CSV (próbowano UTF-8-SIG/UTF-8/Latin-1). Ostatni błąd: {last_err}"

def validate_history(df: pd.DataFrame):
    need = {"Data publikacji", "Treść posta (tekst)", "Temat/kategoria",
            "Liczba reakcji", "Liczba komentarzy", "Liczba udostępnień"}
    return [c for c in need if c not in set(df.columns)]

def validate_topics(df: pd.DataFrame):
    need = {"data", "temat"}
    return [c for c in need if c not in set(df.columns)]

run = st.button("Generuj posty", type="primary", disabled=not (hist_up and topics_up))
if run:
    with st.spinner("Przetwarzam…"):
        hist_df, err1 = read_csv_safely(hist_up)
        topics_df, err2 = read_csv_safely(topics_up)
        if err1: st.error(f"history.csv: {err1}"); st.stop()
        if err2: st.error(f"tematy.csv: {err2}"); st.stop()

        miss_h = validate_history(hist_df)
        miss_t = validate_topics(topics_df)
        if miss_h: st.error(f"history.csv brakuje kolumn: {', '.join(miss_h)}"); st.stop()
        if miss_t: st.error(f"tematy.csv brakuje kolumn: {', '.join(miss_t)}"); st.stop()

        # przygotuj historię (opcjonalnie; aktualnie nie używamy w few-shots)
        hist_df = hist_df.rename(columns={
            'Data publikacji':'date', 'Treść posta (tekst)':'text', 'Temat/kategoria':'category',
            'Liczba reakcji':'reactions', 'Liczba komentarzy':'comments', 'Liczba udostępnień':'shares'
        })
        for c in ['reactions','comments','shares']:
            hist_df[c] = pd.to_numeric(hist_df[c], errors='coerce').fillna(0).astype(int)
        hist_df = hist_df[hist_df['text'].str.len() > 30]
        hist_df['engagement'] = hist_df[['reactions','comments','shares']].sum(axis=1)
        hist_top = hist_df.sort_values('engagement', ascending=False).head(N_SHOTS)  # reserved

        try:
            df_out = generate_posts(hist_df, topics_df if "kontekst" in topics_df.columns else topics_df.assign(kontekst=""))
        except Exception as e:
            st.exception(e); st.stop()

        buf = io.BytesIO()
        df_out.to_csv(buf, index=False, encoding="utf-8-sig")
        buf.seek(0)
        st.success("Gotowe! Pobierz plik poniżej.")
        st.download_button("Pobierz posty_na_fb.csv", buf, file_name="posty_na_fb.csv", mime="text/csv")

with st.expander("Diagnostyka (opcjonalnie)"):
    st.write("history.csv wgrany:", hist_up is not None)
    st.write("tematy.csv wgrany:", topics_up is not None)
