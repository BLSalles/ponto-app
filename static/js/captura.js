// Captura um frame do <video> preservando a proporção real da câmera.
//
// Antes, tanto o cadastro quanto a batida do ponto criavam um canvas fixo de 640x480 e
// chamavam drawImage(video, 0, 0, 640, 480), o que ESTICA o quadro até esse tamanho.
// No celular segurado em pé a câmera entrega um quadro em pé (ex.: 480x640), então o
// rosto era espremido na horizontal em ~1,8x antes de sair a foto. O encoding facial
// gerado a partir de um rosto distorcido fica bem diferente do encoding cadastrado — e
// pior ainda quando o cadastro foi feito no computador (quadro deitado, sem distorção) e
// a batida é no celular (quadro em pé, distorcido). Era a causa principal de "o rosto não
// confere" com a pessoa certa na frente da câmera.
//
// Aqui o canvas acompanha a proporção do vídeo, só reduzindo a resolução quando o quadro
// é maior que `ladoMaximo` (para a foto não ficar pesada demais no upload).
window.capturarFrameFacial = function (video, ladoMaximo) {
  ladoMaximo = ladoMaximo || 720;

  const larguraFonte = video.videoWidth || 640;
  const alturaFonte = video.videoHeight || 480;
  const escala = Math.min(1, ladoMaximo / Math.max(larguraFonte, alturaFonte));

  const canvas = document.createElement('canvas');
  canvas.width = Math.round(larguraFonte * escala);
  canvas.height = Math.round(alturaFonte * escala);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

  // Qualidade 0.9 (antes 0.85): o artefato do JPEG em volta dos olhos/boca atrapalha a
  // detecção do rosto, e a diferença de tamanho do upload é pequena.
  return canvas.toDataURL('image/jpeg', 0.9);
};
