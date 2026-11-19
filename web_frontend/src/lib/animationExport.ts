/**
 * Animation export, recorded straight off the WebGL canvas.
 *
 * The desktop writes GIF through imageio and MP4 through OpenCV. Neither has a
 * browser equivalent without shipping an encoder, and the contract forbids new
 * dependencies — but every browser that can run this app already has a video
 * encoder wired to `MediaStream`. So the canvas is captured with
 * `captureStream` and encoded by `MediaRecorder` into WebM, which plays in any
 * modern browser and drops into a slide deck.
 *
 * The recording is real time: the playback is driven frame by frame at the
 * requested rate and the recorder captures what is actually on screen, so what
 * you export is exactly what you watched.
 */

export type AnimationOptions = {
  /** Frames per second for both playback stepping and the encoder. */
  fps: number;
  /** How many steps to walk from 0 to 100 percent. */
  frames: number;
  /** Called with each step's percentage; must update the viewport. */
  onSeek: (timePercent: number) => void;
  /** Progress feedback, 0..1. */
  onProgress?: (fraction: number) => void;
  signal?: AbortSignal;
};

export type AnimationResult = { blob: Blob; extension: string };

/** WebM codecs in preference order; the first the browser admits to wins. */
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

/**
 * Record a sweep of the time slider off the given canvas.
 *
 * Throws rather than returning a broken file when the browser cannot encode,
 * so the caller can tell the user why instead of downloading nothing.
 */
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
      // The recorder samples the canvas as the browser paints it, so each step
      // has to stay on screen for a frame's worth of wall clock.
      await wait(frameMs);
    }
    // Hold the last frame so it is not clipped by the stop.
    await wait(frameMs * 2);
  } finally {
    if (recorder.state !== 'inactive') recorder.stop();
    stream.getTracks().forEach((track) => track.stop());
  }

  await finished;
  if (!chunks.length) throw new Error('The recorder produced no data.');
  return { blob: new Blob(chunks, { type: mimeType }), extension: 'webm' };
}
