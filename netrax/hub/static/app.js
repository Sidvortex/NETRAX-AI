// Shared dashboard helpers: header, status bar, API calls, toasts. No framework.
"use strict";

const NX = (() => {
  const NAV = [["/", "Live"], ["/people", "People"], ["/events", "Events"], ["/settings", "Settings"]];
  const MODULE_LABELS = {
    face_detect: "Face detect", face_recog: "Face recog", head_pose: "Head pose",
    hands: "Hands", pose: "Pose", objects: "Objects", emotion: "Emotion",
  };

  function el(tag, attrs = {}, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v === true ? "" : v);
    }
    for (const k of kids.flat()) if (k !== null && k !== undefined) n.append(k.nodeType ? k : String(k));
    return n;
  }

  async function api(path, opts = {}) {
    const init = { headers: {}, ...opts };
    if (opts.json !== undefined) {
      init.method = init.method || "POST";
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.json);
      delete init.json;
    }
    const r = await fetch(path, init);
    const text = await r.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = text; }
    if (!r.ok) {
      let msg = (data && data.detail) || r.statusText;
      if (Array.isArray(msg)) msg = msg.map(d => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
      throw new Error(msg);
    }
    return data;
  }

  let toastTimer;
  function toast(msg, isErr = false) {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.classList.toggle("err", isErr);
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 3200);
  }

  function header() {
    const nav = el("nav", {}, NAV.map(([href, name]) =>
      el("a", { href, class: location.pathname === href ? "active" : "" }, name)));
    document.body.prepend(el("header", { class: "top" },
      el("div", { class: "brand" }, "NETRAX", el("small", {}, "vision hub")), nav));
    document.body.append(el("div", { id: "statusbar" }, "connecting…"), el("div", { id: "toast" }));
  }

  const listeners = [];
  function onStatus(fn) { listeners.push(fn); }

  async function pollStatus() {
    const bar = document.getElementById("statusbar");
    try {
      const s = await api("/api/status");
      const cam = s.camera || {};
      const parts = [
        el("span", {}, el("i", { class: `dot ${cam.state}` }), "camera ", el("b", {}, cam.state || "?"),
          cam.size && cam.size[0] ? ` ${cam.size[0]}×${cam.size[1]}` : ""),
        el("span", {}, "pipeline ", el("b", {}, `${s.fps} fps`), ` ${s.proc_ms} ms`),
        el("span", {}, "cpu ", el("b", {}, `${s.cpu}%`), ` (netrax ${s.proc_cpu}%)`),
      ];
      for (const [name, m] of Object.entries(s.modules || {})) {
        if (m.state === "off") continue;
        const detail = m.state === "ready" ? `${m.fps} fps · ${m.ms} ms` : m.state;
        parts.push(el("span", { class: "chip", title: m.error || "" },
          el("i", { class: `dot ${m.state}` }), MODULE_LABELS[name] || name, " ", el("b", {}, detail)));
      }
      if (s.mqtt) parts.push(el("span", {}, el("i", { class: `dot ${s.mqtt.connected ? "live" : "error"}` }), "mqtt"));
      bar.replaceChildren(...parts);
      listeners.forEach(fn => fn(s));
    } catch (e) {
      bar.replaceChildren(el("span", {}, el("i", { class: "dot error" }), "hub unreachable"));
    }
  }

  function fmtTime(ts) {
    const d = new Date(ts * 1000);
    return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  document.addEventListener("DOMContentLoaded", () => {
    header();
    pollStatus();
    setInterval(pollStatus, 1500);
  });

  return { el, api, toast, onStatus, fmtTime, MODULE_LABELS };
})();
