"use strict";
// All untrusted text is inserted with textContent / text nodes. Never innerHTML.

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const state = { session: null, answers: [] };
const LABELS = { quote: "Verbatim quote", extrapolation: "Extrapolation", no_evidence: "No evidence" };

async function loadInfo() {
  const res = await fetch("/api/info");
  const info = await res.json();
  $("name").textContent = info.name;
  $("bio").textContent = info.bio || "";
  $("disclosure").textContent = info.disclosure;
  document.title = `${info.name} · Anyone`;
  if (info.provider === "extractive") {
    $("disclosure").textContent += " Offline mode: answers show the closest passages only.";
  }
}

function normalize(s) {
  return s.normalize("NFKC").replace(/[‘’‚‛]/g, "'").replace(/[“”„‟]/g, '"')
    .replace(/[–—−]/g, "-");
}

// Builds the passage with every verified quote from this answer highlighted.
function passage(text, quotes) {
  const p = el("p");
  const norm = normalize(text);
  const ranges = [];
  for (const q of quotes) {
    // Stored text is already NFKC-normalized at ingest, so offsets in `norm` map 1:1 onto `text`.
    const nq = normalize(q);
    const i = norm.indexOf(nq);
    if (i >= 0) ranges.push([i, i + nq.length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  let pos = 0;
  for (const [s, e] of ranges) {
    if (s < pos) continue;
    p.append(document.createTextNode(text.slice(pos, s)), el("mark", null, text.slice(s, e)));
    pos = e;
  }
  p.append(document.createTextNode(text.slice(pos)));
  return p;
}

function showEvidence(answerIdx, focusLabel) {
  const a = state.answers[answerIdx];
  const list = $("evidence-list");
  list.replaceChildren();
  $("evidence-empty").hidden = true;
  for (const [label, ev] of Object.entries(a.evidence)) {
    const quotes = a.segments.filter((s) => s.kind === "quote" && s.evidence_ids.includes(label)).map((s) => s.text);
    const card = el("article", "src" + (label === focusLabel ? " active" : ""));
    card.append(el("h3", null, `${label} · ${ev.title}`), el("div", "meta", `${ev.date || "undated"} · ${ev.source_id}`), passage(ev.text, quotes));
    list.append(card);
    if (label === focusLabel) card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  document.querySelectorAll(".cite").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.answer === String(answerIdx) && b.dataset.label === focusLabel));
  });
}

function renderAnswer(a) {
  const idx = state.answers.push(a) - 1;
  const li = el("li", "msg-ai");
  for (const seg of a.segments) {
    const box = el("div", `seg seg-${seg.kind}`);
    const quoteLabel = seg.reported_by ? `${LABELS.quote} ✓ · as reported by ${seg.reported_by}` : `${LABELS.quote} ✓`;
    const label = el("div", "label", seg.kind === "quote" ? quoteLabel : LABELS[seg.kind]);
    for (const id of seg.evidence_ids) {
      const b = el("button", "cite");
      b.type = "button";
      b.dataset.answer = String(idx);
      b.dataset.label = id;
      b.setAttribute("aria-pressed", "false");
      b.title = a.evidence[id] ? a.evidence[id].title : id;
      b.append(el("span", null, id));
      b.addEventListener("click", () => showEvidence(idx, id));
      label.append(b);
    }
    box.append(label, el("div", "text", seg.text));
    if (seg.note) box.append(el("div", "note muted", seg.note));
    li.append(box);
  }
  const v = a.validation || {};
  if (v.quotes_demoted || (v.removed && v.removed.length)) {
    li.append(el("div", "checks muted",
      `Checks: ${v.quotes_verified} quote(s) verified, ${v.quotes_demoted} relabeled as extrapolation, ${v.removed.length} unsupported segment(s) removed.`));
  }
  $("log").append(li);
  const firstCited = a.segments.flatMap((s) => s.evidence_ids)[0];
  if (firstCited) showEvidence(idx, firstCited);
  li.scrollIntoView({ block: "end", behavior: "smooth" });
}

async function ask(question) {
  $("log").append(el("li", "msg-user", question));
  const pending = el("li", "thinking", "Checking the record…");
  $("log").append(pending);
  pending.scrollIntoView({ block: "end" });
  $("send").disabled = true;
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session: state.session }),
    });
    const data = await res.json();
    if (data.session) state.session = data.session;
    pending.remove();
    if (!res.ok) {
      $("log").append(el("li", "error", data.message || data.error || "Something went wrong."));
    } else {
      renderAnswer(data);
    }
  } catch (err) {
    pending.remove();
    $("log").append(el("li", "error", "Could not reach the local Anyone server."));
  } finally {
    $("send").disabled = false;
    $("q").focus();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadInfo().catch(() => { $("name").textContent = "Anyone"; });
  $("ask").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("q").value.trim();
    if (!q || $("send").disabled) return;
    $("q").value = "";
    ask(q);
  });
  $("q").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("ask").requestSubmit(); }
  });
  $("reset").addEventListener("click", () => {
    state.session = null;
    state.answers = [];
    $("log").replaceChildren($("log").firstElementChild);
    $("evidence-list").replaceChildren();
    $("evidence-empty").hidden = false;
  });
});
