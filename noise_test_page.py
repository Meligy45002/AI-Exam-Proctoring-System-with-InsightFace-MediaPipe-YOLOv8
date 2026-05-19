"""
Noise Robustness Test Page
Self-contained — no dependency on NoiseTester / GazeTracker / ObjectDetector.

Noise types use integer intensity levels: 1 (Low), 2 (Medium), 3 (High).
Imported and called from app.py as:
    from noise_test_page import render_noise_test
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# NOISE FUNCTIONS  (integer intensity 1 / 2 / 3)
# ══════════════════════════════════════════════════════════════

def _noise_gaussian(img, intensity):
    sigma = {1: 15, 2: 35, 3: 65}[intensity]
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def _noise_salt_pepper(img, intensity):
    density = {1: 0.03, 2: 0.08, 3: 0.18}[intensity]
    out = img.copy()
    n = int((img.size // 3) * density)
    xs = np.random.randint(0, img.shape[1], n * 2)
    ys = np.random.randint(0, img.shape[0], n * 2)
    out[ys[:n], xs[:n]] = 255
    out[ys[n:], xs[n:]] = 0
    return out

def _noise_blur(img, intensity):
    k = {1: 9, 2: 19, 3: 35}[intensity]
    return cv2.GaussianBlur(img, (k, k), 0)

def _noise_motion_blur(img, intensity):
    size = {1: 10, 2: 20, 3: 40}[intensity]
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(img, -1, kernel)

def _noise_dark(img, intensity):
    factor = {1: 0.55, 2: 0.30, 3: 0.10}[intensity]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def _noise_bright(img, intensity):
    factor = {1: 1.5, 2: 2.0, 3: 2.8}[intensity]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def _noise_occlude(img, intensity):
    out = img.copy()
    h, w = out.shape[:2]
    s = {1: 0.20, 2: 0.35, 3: 0.50}[intensity]
    y1, y2 = int(h * 0.45), int(h * (0.45 + s))
    x1, x2 = int(w * 0.25), int(w * 0.75)
    out[y1:y2, x1:x2] = [80, 80, 80]
    return out

def _noise_jpeg(img, intensity):
    quality = {1: 30, 2: 12, 3: 3}[intensity]
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def _noise_pixel(img, intensity):
    factor = {1: 6, 2: 12, 3: 22}[intensity]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)),
                       interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def _noise_color(img, intensity):
    amount = {1: 30, 2: 60, 3: 100}[intensity]
    out = img.astype(np.int32)
    for c in range(3):
        out[:, :, c] = np.clip(
            out[:, :, c] + np.random.randint(-amount, amount), 0, 255)
    return out.astype(np.uint8)

def _noise_rotate(img, intensity):
    angle = {1: 10, 2: 22, 3: 40}[intensity]
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


NOISE_CATALOG = {
    "Gaussian Noise":   (_noise_gaussian,    "Bad camera sensor"),
    "Salt & Pepper":    (_noise_salt_pepper,  "Pixel corruption"),
    "Blur":             (_noise_blur,         "Out-of-focus camera"),
    "Motion Blur":      (_noise_motion_blur,  "Fast head movement"),
    "Low Brightness":   (_noise_dark,         "Dark room"),
    "High Brightness":  (_noise_bright,       "Bright backlight"),
    "Face Occlusion":   (_noise_occlude,      "Hand covering face"),
    "JPEG Compression": (_noise_jpeg,         "Low-bandwidth stream"),
    "Pixelation":       (_noise_pixel,        "Extreme compression"),
    "Color Jitter":     (_noise_color,        "Bad white balance"),
    "Rotation":         (_noise_rotate,       "Tilted camera"),
}


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _get_students(data_dir="data/registered_faces"):
    students = {}
    p = Path(data_dir)
    if not p.exists():
        return students
    for d in sorted(p.iterdir()):
        if d.is_dir():
            photos = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))
            if photos:
                display = d.name.rsplit("_", 1)[0]
                students[display] = [str(ph) for ph in photos]
    return students


def _get_face_embedding(detector, img_bgr):
    """Extract face embedding using whatever engine is available."""
    try:
        return detector.get_embedding(img_bgr)
    except Exception:
        pass
    return None, 0


def _cosine_sim(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def _build_face_db(detector, students):
    """Build name -> [embeddings] from registered photos."""
    try:
        if hasattr(detector, 'build_face_db'):
            return detector.build_face_db()
    except Exception:
        pass
    db = {}
    for name, photos in students.items():
        embs = []
        for ph in photos:
            img = cv2.imread(ph)
            if img is None:
                continue
            emb, _ = _get_face_embedding(detector, img)
            if emb is not None:
                embs.append(emb)
        if embs:
            db[name] = embs
    return db


def _run_recognition(detector, img_bgr, face_db=None):
    """Recognise face — returns (name, similarity, face_count)."""
    try:
        if face_db is not None and hasattr(detector, 'identify'):
            return detector.identify(img_bgr, face_db, threshold=0.55)
    except Exception:
        pass

    if face_db:
        emb, n_faces = _get_face_embedding(detector, img_bgr)
        if n_faces == 0:
            return "Unknown", 0.0, 0
        if emb is not None:
            best_name, best_sim = "Unknown", 0.0
            for sname, embs in face_db.items():
                sim = max(_cosine_sim(emb, e) for e in embs)
                if sim > best_sim:
                    best_sim, best_name = sim, sname
            return (best_name, best_sim, n_faces) if best_sim >= 0.55 \
                   else ("Unknown", best_sim, n_faces)
        return "Face found (no ID)", 0.0, n_faces

    try:
        faces, name, conf = detector.detect_and_verify(img_bgr)
        return name or "Unknown", float(conf) if conf else 0.0, len(faces)
    except Exception:
        return "Error", 0.0, 0


def _is_match(expected, got):
    if not got or got in ("Unknown", "Error", "Face found (no ID)"):
        return False
    return (expected.lower().split()[0] in got.lower() or
            got.lower().split()[0] in expected.lower())


def _result_card(label, expected, got_name, confidence, faces):
    if faces == 0:
        icon, color, verdict = "😶", "#ef4444", "NO FACE DETECTED"
        bg, border = "rgba(239,68,68,0.12)", "#ef4444"
    elif _is_match(expected, got_name):
        icon, color, verdict = "✅", "#22c55e", f"RECOGNIZED AS: {got_name}"
        bg, border = "rgba(34,197,94,0.12)", "#22c55e"
    else:
        icon, color, verdict = "❌", "#f97316", f"WRONG: GOT '{got_name}'"
        bg, border = "rgba(249,115,22,0.12)", "#f97316"

    st.markdown(f"""
    <div style='background:{bg};border:2px solid {border};border-radius:12px;
                padding:16px 20px;'>
        <div style='font-size:1.15rem;font-weight:700;color:{color};
                    margin-bottom:8px;'>{icon} {verdict}</div>
        <div style='color:#e0e0e0;font-size:0.92rem;line-height:1.8;'>
            <b>Test:</b> {label}<br>
            <b>Expected name:</b> {expected}<br>
            <b>Model said:</b> {got_name}<br>
            <b>Confidence:</b> {confidence * 100:.1f}%<br>
            <b>Faces found:</b> {faces}
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════

def render_noise_test():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#0f3460);
                border-radius:14px;padding:24px 28px;margin-bottom:20px;
                border:1px solid rgba(99,179,237,0.3);'>
        <h1 style='color:#fff;margin:0;font-size:1.85rem;'>🧪 Noise Robustness Test</h1>
        <p style='color:#93c5fd;margin:8px 0 0 0;'>
            Select a registered student → their photo loads automatically →
            noise is applied → the model must still recognize them by name.
        </p>
    </div>""", unsafe_allow_html=True)

    # ── Load detector once ────────────────────────────────────────────────────
    if "nt_detector" not in st.session_state:
        with st.spinner("Loading face recognition model..."):
            try:
                from proctoring_engine import FaceDetector
                st.session_state.nt_detector = FaceDetector()
                st.success("✅ Face recognition engine loaded")
            except Exception as e:
                st.error(f"❌ Could not load FaceDetector: {e}")
                st.session_state.nt_detector = None

    detector = st.session_state.nt_detector
    if detector is None:
        st.error("❌ FaceDetector not loaded. Make sure proctoring_engine.py is present.")
        return

    students = _get_students()
    if not students:
        st.warning("⚠️ No registered students found. Go to 👤 Student Registration tab first.")
        return

    # ── Build face embedding DB once per session ──────────────────────────────
    if "nt_face_db" not in st.session_state:
        with st.spinner("Building face database from registered photos..."):
            st.session_state.nt_face_db = _build_face_db(detector, students)
        db_size = sum(len(v) for v in st.session_state.nt_face_db.values())
        if db_size == 0:
            st.warning("⚠️ Could not extract face embeddings from registered photos.")
        else:
            st.success(
                f"✅ Face database ready: {db_size} embeddings "
                f"for {len(st.session_state.nt_face_db)} students."
            )
    face_db = st.session_state.nt_face_db

    # ── STEP 1: Pick student ──────────────────────────────────────────────────
    st.markdown("### 👤 Step 1 — Select student")

    names = list(students.keys())
    btn_cols = st.columns(min(len(names), 4))
    for i, n in enumerate(names):
        if btn_cols[i % 4].button(f"👤 {n}", key=f"nt_btn_{n}",
                                   use_container_width=True):
            st.session_state["nt_selected"] = n

    typed = st.text_input("Or type a name:",
                          placeholder="e.g. Ahmed Ezzat",
                          help="Registered: " + ", ".join(names))
    if typed.strip():
        hit = next((n for n in names if typed.strip().lower() in n.lower()), None)
        if hit:
            st.session_state["nt_selected"] = hit
        else:
            st.error(f"❌ '{typed}' not found. Available: {', '.join(names)}")

    selected = st.session_state.get("nt_selected")
    if not selected:
        st.info("👆 Click a student button or type a name above.")
        return

    st.success(f"✅ Student: **{selected}**")

    photos = students[selected]
    photo_idx = 0
    if len(photos) > 1:
        photo_idx = st.select_slider(
            f"Photo ({len(photos)} available)",
            options=list(range(len(photos))),
            format_func=lambda x: f"Photo {x + 1}"
        )

    original_bgr = cv2.imread(photos[photo_idx])
    if original_bgr is None:
        st.error("Could not read photo file.")
        return

    # ── STEP 2: Baseline ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Step 2 — Baseline: recognition on clean photo")

    c1, c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB),
                 caption=f"Registered photo of {selected}",
                 use_container_width=True)
    with c2:
        base_name, base_conf, base_faces = _run_recognition(
            detector, original_bgr, face_db)
        _result_card("Clean image", selected, base_name, base_conf, base_faces)

    if base_faces == 0:
        st.warning("⚠️ No face detected in the clean photo. "
                   "Try a different photo number or re-register.")
        return

    # ── STEP 3: Choose noise ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌫️ Step 3 — Choose noise type and intensity")

    cn, ci = st.columns([2, 1])
    with cn:
        noise_type = st.selectbox(
            "Noise type", list(NOISE_CATALOG.keys()),
            format_func=lambda k: f"{k}  —  {NOISE_CATALOG[k][1]}"
        )
    with ci:
        intensity = st.select_slider(
            "Intensity", options=[1, 2, 3], value=2,
            format_func=lambda x: {1: "🟢 Low", 2: "🟡 Medium", 3: "🔴 High"}[x]
        )

    # ── STEP 4: Apply & test ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚡ Step 4 — Apply noise → test identity recognition")

    fn, _ = NOISE_CATALOG[noise_type]
    try:
        noisy_bgr = fn(original_bgr.copy(), intensity)
    except Exception as e:
        st.error(f"Noise failed: {e}")
        return

    cn2, cr2 = st.columns(2)
    with cn2:
        st.image(cv2.cvtColor(noisy_bgr, cv2.COLOR_BGR2RGB),
                 caption=f"{noise_type} — intensity {intensity}",
                 use_container_width=True)
    with cr2:
        noisy_name, noisy_conf, noisy_faces = _run_recognition(
            detector, noisy_bgr, face_db)
        _result_card(f"After {noise_type}", selected,
                     noisy_name, noisy_conf, noisy_faces)

    # ── STEP 5: Verdict ───────────────────────────────────────────────────────
    st.markdown("---")
    lbl = {1: "Low", 2: "Medium", 3: "High"}[intensity]
    correct = _is_match(selected, noisy_name)
    drop = (base_conf - noisy_conf) * 100

    if noisy_faces == 0:
        color, bg = "#ef4444", "rgba(239,68,68,0.12)"
        title = "💀 COMPLETE FAILURE — face not detected"
        body = (f"**{noise_type}** at **{lbl}** intensity destroyed the face entirely. "
                f"Model couldn't find **{selected}** at all.")
    elif correct:
        color, bg = "#22c55e", "rgba(34,197,94,0.1)"
        title = f"✅ ROBUST — model still recognizes {selected}"
        body = (f"Even with **{noise_type}** at **{lbl}** intensity, model correctly "
                f"identified **{selected}**. Confidence dropped by {drop:.1f}%.")
    else:
        color, bg = "#f97316", "rgba(249,115,22,0.12)"
        title = f"❌ CONFUSED — expected '{selected}', model said '{noisy_name}'"
        body = (f"**{noise_type}** at **{lbl}** intensity caused the model to lose "
                f"**{selected}**'s identity.")

    st.markdown(f"""
    <div style='background:{bg};border:2px solid {color};border-radius:14px;
                padding:18px 24px;'>
        <div style='font-size:1.2rem;font-weight:700;color:{color};'>{title}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(body)

    m1, m2, m3 = st.columns(3)
    m1.metric("Clean confidence",  f"{base_conf * 100:.1f}%")
    m2.metric("Noisy confidence",  f"{noisy_conf * 100:.1f}%",
              delta=f"-{drop:.1f}%")
    m3.metric("Faces detected",    noisy_faces)

    # ── STEP 6: Full batch test ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Step 6 — Full batch test (all noise × all intensities)")
    st.caption(
        f"Tests all {len(NOISE_CATALOG)} noise types at Low / Medium / High on "
        f"**{selected}**'s photo and checks if the model still says their name."
    )

    if st.button("🚀 Run full batch test", type="primary", key="nt_batch"):
        import pandas as pd
        rows = []
        total = len(NOISE_CATALOG) * 3
        bar = st.progress(0)
        done = 0
        lbl_map = {1: "Low", 2: "Medium", 3: "High"}

        for nname, (nfn, ndesc) in NOISE_CATALOG.items():
            for nint in [1, 2, 3]:
                try:
                    nimg = nfn(original_bgr.copy(), nint)
                    gname, gconf, gfaces = _run_recognition(detector, nimg, face_db)
                except Exception:
                    gname, gconf, gfaces = "Error", 0.0, 0

                ok = _is_match(selected, gname)
                if gfaces == 0:
                    status = "💀 No Face"
                elif ok:
                    status = "✅ Correct"
                else:
                    status = f"❌ Said '{gname}'"

                rows.append({
                    "Noise Type":  nname,
                    "Simulates":   ndesc,
                    "Intensity":   lbl_map[nint],
                    "Result":      status,
                    "Confidence":  f"{gconf * 100:.1f}%",
                    "Conf. Drop":  f"−{max(0, (base_conf - gconf) * 100):.1f}%",
                })
                done += 1
                bar.progress(done / total)

        bar.empty()
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        correct_n  = sum(1 for r in rows if "Correct" in r["Result"])
        confused_n = sum(1 for r in rows if "Said"    in r["Result"])
        noface_n   = sum(1 for r in rows if "No Face" in r["Result"])

        st.markdown("---")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total tests",          total)
        b2.metric(f"✅ '{selected}'",      correct_n)
        b3.metric("❌ Wrong identity",     confused_n)
        b4.metric("💀 No face",            noface_n)

        os.makedirs("data", exist_ok=True)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        rpath = (f"data/noise_report_{selected.replace(' ', '_')}_{ts}.json")
        with open(rpath, "w") as f:
            json.dump({
                "student":              selected,
                "timestamp":            datetime.now().isoformat(),
                "baseline_confidence":  round(base_conf * 100, 1),
                "results":              rows,
                "summary": {
                    "total":    total,
                    "correct":  correct_n,
                    "confused": confused_n,
                    "no_face":  noface_n
                }
            }, f, indent=2)
        st.success(f"📄 Report saved → `{rpath}`")
