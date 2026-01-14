const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const jpeg = require("jpeg-js");

// ---- Teachable Machine metadata ----
const LABELS = ["TEACH", "STRONG", "STOP", "SORRY", "PLEASE"];
const IMAGE_SIZE = 224;

const MODEL_DIR = path.join(__dirname, "model");
const MODEL_JSON_PATH = path.join(MODEL_DIR, "model.json");

const DSHOW_CAMERA_NAME = "A4tech PC Camera";

// Capture settings
const WIDTH = 224;
const HEIGHT = 224;
const FPS = 10;

function topK(probs, k = 2) {
  const arr = Array.from(probs).map((p, i) => ({
    label: LABELS[i] ?? String(i),
    p,
  }));
  arr.sort((a, b) => b.p - a.p);
  return arr.slice(0, k);
}

// ---- Load Layers model from disk (pure tfjs) ----
function loadLayersModelFromFS(modelJsonPath) {
  const modelDir = path.dirname(modelJsonPath);
  const handler = {
    load: async () => {
      const modelJSON = JSON.parse(fs.readFileSync(modelJsonPath, "utf8"));
      const weightPaths = modelJSON.weightsManifest.flatMap((m) => m.paths);
      const bufs = weightPaths.map(
        (p) => fs.readFileSync(path.join(modelDir, p)).buffer
      );

      const total = bufs.reduce((s, b) => s + b.byteLength, 0);
      const out = new Uint8Array(total);
      let off = 0;
      for (const b of bufs) {
        out.set(new Uint8Array(b), off);
        off += b.byteLength;
      }

      return {
        modelTopology: modelJSON.modelTopology,
        format: modelJSON.format,
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy,
        weightSpecs: modelJSON.weightsManifest.flatMap((m) => m.weights),
        weightData: out.buffer,
      };
    },
  };
  return tf.loadLayersModel(handler);
}

// ---- Start ONE ffmpeg that opens the camera and outputs MJPEG to stdout ----
function startFFmpegMJPEG() {
  const args = [
    "-hide_banner",
    "-loglevel",
    "error",
    "-f",
    "dshow",
    "-i",
    `video=${DSHOW_CAMERA_NAME}`,
    "-vf",
    `fps=${FPS},scale=${WIDTH}:${HEIGHT}`,
    "-q:v",
    "5",
    "-f",
    "mjpeg",
    "pipe:1",
  ];

  const ff = spawn("ffmpeg", args, { stdio: ["ignore", "pipe", "pipe"] });

  ff.stderr.on("data", (d) => console.error("[ffmpeg]", d.toString().trim()));
  ff.on("close", (code) => {
    console.error("ffmpeg exited:", code);
    process.exit(code ?? 1);
  });

  return ff;
}

// ---- Preview: ffplay reads MJPEG from stdin (does NOT open camera) ----
function startFFplayFromStdin() {
  const args = [
    "-hide_banner",
    "-loglevel",
    "error",
    "-fflags",
    "nobuffer",
    "-f",
    "mjpeg",
    "-i",
    "pipe:0",
  ];

  const p = spawn("ffplay", args, { stdio: ["pipe", "inherit", "inherit"] });
  p.on("close", (code) => console.log("ffplay closed:", code));
  return p;
}

// ---- Extract JPEG frames from MJPEG byte stream ----
// We detect JPEG start/end markers: SOI (FFD8) ... EOI (FFD9)
function createMJPEGFrameParser(onFrameJpegBuffer) {
  let buf = Buffer.alloc(0);

  return (chunk) => {
    buf = Buffer.concat([buf, chunk]);

    while (true) {
      const soi = buf.indexOf(Buffer.from([0xff, 0xd8])); // start
      if (soi === -1) {
        // keep small tail to avoid memory growth
        if (buf.length > 1024 * 1024) buf = buf.subarray(buf.length - 1024);
        return;
      }
      const eoi = buf.indexOf(Buffer.from([0xff, 0xd9]), soi + 2); // end
      if (eoi === -1) return;

      const frame = buf.subarray(soi, eoi + 2);
      buf = buf.subarray(eoi + 2);

      onFrameJpegBuffer(frame);
    }
  };
}

async function main() {
  console.log("Loading model:", MODEL_JSON_PATH);
  const model = await loadLayersModelFromFS(MODEL_JSON_PATH);
  console.log("Model loaded.");

  // warmup
  tf.tidy(() => model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])));

  console.log("Opening preview window (ffplay)...");
  const player = startFFplayFromStdin();

  console.log("Starting camera via ffmpeg (single open)...");
  const ff = startFFmpegMJPEG();

  let lastLog = 0;
  let busy = false; // drop frames if inference is still running

  const parse = createMJPEGFrameParser(async (jpegBuf) => {
    // send to preview window
    if (!player.killed) {
      player.stdin.write(jpegBuf);
    }

    // Don’t queue infinite inferences
    if (busy) return;
    busy = true;

    try {
      // Decode JPEG -> RGBA
      const decoded = jpeg.decode(jpegBuf, { useTArray: true }); // {width,height,data(RGBA)}
      if (!decoded || !decoded.data) return;

      // Convert RGBA -> RGB tensor
      const input = tf.tidy(() => {
        const rgba = decoded.data; // Uint8Array length = w*h*4
        // Build RGB array
        const rgb = new Uint8Array(WIDTH * HEIGHT * 3);
        for (let i = 0, j = 0; i < rgba.length; i += 4) {
          rgb[j++] = rgba[i]; // R
          rgb[j++] = rgba[i + 1]; // G
          rgb[j++] = rgba[i + 2]; // B
        }
        const img = tf.tensor3d(rgb, [HEIGHT, WIDTH, 3], "int32");
        return img.toFloat().div(255).expandDims(0);
      });

      const output = model.predict(input);
      const probsTensor = tf.tidy(() => tf.softmax(output));
      const probs = await probsTensor.data();

      tf.dispose([input, output, probsTensor]);

      const now = Date.now();
      if (now - lastLog > 500) {
        lastLog = now;
        const [best, second] = topK(probs, 2);
        console.log(
          `${new Date().toISOString()}  ${best.label}: ${(best.p * 100).toFixed(
            1
          )}%` +
            (second
              ? ` | ${second.label}: ${(second.p * 100).toFixed(1)}%`
              : "")
        );
      }
    } catch (e) {
      // ignore occasional decode errors
      // console.error("Frame error:", e);
    } finally {
      busy = false;
    }
  });

  ff.stdout.on("data", (chunk) => parse(chunk));

  process.on("SIGINT", () => {
    console.log("\nStopping...");
    try {
      ff.kill("SIGINT");
    } catch {}
    try {
      player.kill("SIGINT");
    } catch {}
    process.exit(0);
  });
}

main().catch(console.error);
