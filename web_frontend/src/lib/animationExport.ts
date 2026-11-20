export type AnimationOptions = {
  fps: number;
  frames: number;
  onSeek: (timePercent: number) => void;
  onProgress?: (fraction: number) => void;
  signal?: AbortSignal;
};

export type AnimationResult = { blob: Blob; extension: string };

const CANDIDATE_TYPES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm'
];

export function animationSupported(): boolean {
  return (
    typeof MediaRecorder !== 'undefined' &&
    typeof HTMLCanvasElement !== 'undefined' &&
    typeof HTMLCanvasElement.prototype.captureStream === 'function'
  );
}

function pickMimeType(): string | null {
  for (const type of CANDIDATE_TYPES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return null;
}

const wait = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

export async function recordAnimation(
  canvas: HTMLCanvasElement,
  options: AnimationOptions
): Promise<AnimationResult> {
  if (!animationSupported()) {
    throw new Error('This browser cannot record the canvas. Try Chrome, Edge or Firefox.');
  }
  const mimeType = pickMimeType();
  if (!mimeType) throw new Error('This browser has no WebM encoder available.');

  const fps = Math.max(1, Math.min(60, Math.round(options.fps)));
  const steps = Math.max(2, Math.round(options.frames));
  const stream = canvas.captureStream(fps);
  const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8_000_000 });
  const chunks: BlobPart[] = [];
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };

  const finished = new Promise<void>((resolve, reject) => {
    recorder.onstop = () => resolve();
    recorder.onerror = () => reject(new Error('Recording failed.'));
  });

  recorder.start();
  try {
    const frameMs = 1000 / fps;
    for (let index = 0; index < steps; index += 1) {
      if (options.signal?.aborted) break;
      options.onSeek((index / (steps - 1)) * 100);
      options.onProgress?.(index / (steps - 1));
      await wait(frameMs);
    }
    await wait(frameMs * 2);
  } finally {
    if (recorder.state !== 'inactive') recorder.stop();
    stream.getTracks().forEach((track) => track.stop());
  }

  await finished;
  if (!chunks.length) throw new Error('The recorder produced no data.');
  return { blob: new Blob(chunks, { type: mimeType }), extension: 'webm' };
}
