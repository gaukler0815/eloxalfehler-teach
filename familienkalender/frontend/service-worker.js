/* Service Worker: Offline-Caching + Web-Push-Benachrichtigungen */
const CACHE = "familienkalender-v2";
const ASSETS = [
  "/",
  "/index.html",
  "/styles.css",
  "/app.js",
  "/manifest.json",
  "/icons/icon-192.png",
  "/icons/icon-512.png"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(ASSETS)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

/* Netzwerk zuerst für die API, Cache-Fallback für statische Dateien */
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.pathname.startsWith("/api")) {
    return; // API immer live
  }
  event.respondWith(
    fetch(event.request)
      .then((resp) => {
        const copy = resp.clone();
        caches.open(CACHE).then((c) => c.put(event.request, copy)).catch(() => {});
        return resp;
      })
      .catch(() => caches.match(event.request).then((r) => r || caches.match("/")))
  );
});

/* Eingehende Push-Nachricht anzeigen */
self.addEventListener("push", (event) => {
  let data = { title: "Familienkalender", body: "", url: "/" };
  try {
    if (event.data) data = Object.assign(data, event.data.json());
  } catch (e) {
    if (event.data) data.body = event.data.text();
  }
  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: "/icons/icon-192.png",
      badge: "/icons/icon-192.png",
      tag: data.tag || "familienkalender",
      data: { url: data.url || "/", eventId: data.eventId || null },
      vibrate: [120, 60, 120]
    })
  );
});

/* Klick auf Benachrichtigung: direkt den kompletten Termin öffnen */
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const d = event.notification.data || {};
  const url = d.url || "/";
  const eventId = d.eventId || null;
  event.waitUntil((async () => {
    const list = await clients.matchAll({ type: "window", includeUncontrolled: true });
    for (const client of list) {
      if ("focus" in client) {
        await client.focus();
        // App läuft schon -> Termin-Detail per Nachricht öffnen
        client.postMessage({ type: "open-event", eventId: eventId, url: url });
        return;
      }
    }
    // App nicht offen -> mit Deep-Link starten
    if (clients.openWindow) return clients.openWindow(url);
  })());
});
