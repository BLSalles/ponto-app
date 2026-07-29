// Service worker mínimo — necessário para que o iOS (Safari) permita
// notificações locais quando o app é instalado na Tela de Início.
// Não faz cache/offline por enquanto, só precisa existir e ficar ativo.

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
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
