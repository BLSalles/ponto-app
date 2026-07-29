// Service worker mínimo — necessário para que o iOS (Safari) permita
// notificações locais quando o app é instalado na Tela de Início.
// Não faz cache/offline por enquanto, só precisa existir e ficar ativo.

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

// Recebe a notificação enviada pelo servidor (Web Push de verdade — funciona
// mesmo com o app fechado, desde que o dispositivo tenha internet).
self.addEventListener("push", (event) => {
  let dados = { title: "Lembrete de ponto", body: "Não esqueça de bater o ponto." };
  try {
    if (event.data) dados = event.data.json();
  } catch (e) {
    // ignora payload malformado, usa o texto padrão acima
  }

  event.waitUntil(
    self.registration.showNotification(dados.title || "Lembrete de ponto", {
      body: dados.body || "",
      icon: "/static/icons/icon-192.png",
      badge: "/static/icons/icon-192.png",
    })
  );
});

// Permite fechar a notificação ao tocar nela e focar/abrir o app.
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  event.waitUntil(
    self.clients.matchAll({ type: "window" }).then((clientList) => {
      for (const client of clientList) {
        if ("focus" in client) return client.focus();
      }
      if (self.clients.openWindow) {
        return self.clients.openWindow("/ponto");
      }
    })
  );
});
