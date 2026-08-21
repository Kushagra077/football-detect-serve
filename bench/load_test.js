// k6 load test for /predict.
//
// Ramps arrival rate so you can find the knee where the batching window stops
// helping and queue wait starts dominating.
//
//   k6 run bench/load_test.js
//   BASE_URL=http://localhost:8000 IMAGE=samples/frame.jpg k6 run bench/load_test.js
//   k6 run -e SCENARIO=constant -e RATE=50 -e DURATION=2m bench/load_test.js
//
// Compare the k6 numbers against the server's own view:
//   curl -s localhost:8000/metrics | grep -E 'fds_(queue_wait|batch_size|inference)'

import http from 'k6/http';
import { check, fail } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';
import encoding from 'k6/encoding';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const IMAGE_PATH = __ENV.IMAGE || '';
const MODE = (__ENV.MODE || 'multipart').toLowerCase(); // multipart | json
const SCENARIO = (__ENV.SCENARIO || 'ramp').toLowerCase(); // ramp | constant | smoke
const RATE = parseInt(__ENV.RATE || '20', 10);
const DURATION = __ENV.DURATION || '1m';

// A 1x1 PNG keeps the harness runnable with no fixtures; pass IMAGE= for real
// frames, otherwise you are measuring an empty-image fast path.
const FALLBACK_PNG_B64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx' +
  '0gAAAABJRU5ErkJggg==';

const imageBin = IMAGE_PATH
  ? open(IMAGE_PATH, 'b')
  : encoding.b64decode(FALLBACK_PNG_B64);

// --- custom metrics: server-reported timings, not just wall clock ---
const serverInferMs = new Trend('server_inference_ms', true);
const serverTotalMs = new Trend('server_total_ms', true);
const observedBatch = new Trend('server_batch_size');
const detections = new Trend('detections_per_image');
const emptyResponses = new Counter('empty_detection_responses');
const errorRate = new Rate('predict_errors');

const scenarios = {
  smoke: {
    executor: 'constant-vus',
    vus: 1,
    duration: '20s',
  },
  constant: {
    executor: 'constant-arrival-rate',
    rate: RATE,
    timeUnit: '1s',
    duration: DURATION,
    preAllocatedVUs: Math.max(10, RATE),
    maxVUs: Math.max(50, RATE * 4),
  },
  ramp: {
    executor: 'ramping-arrival-rate',
    startRate: 5,
    timeUnit: '1s',
    preAllocatedVUs: 20,
    maxVUs: 300,
    stages: [
      { target: 10, duration: '30s' },
      { target: 25, duration: '30s' },
      { target: 50, duration: '30s' },
      { target: 100, duration: '30s' },
      { target: 100, duration: '30s' }, // hold at peak
      { target: 5, duration: '15s' },   // recovery: does p95 come back down?
    ],
  },
};

export const options = {
  scenarios: { main: scenarios[SCENARIO] || scenarios.ramp },
  thresholds: {
    'http_req_failed': ['rate<0.01'],
    'predict_errors': ['rate<0.01'],
    'http_req_duration{endpoint:predict}': ['p(95)<500', 'p(99)<1000'],
    'server_inference_ms': ['p(95)<250'],
  },
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

export function setup() {
  const res = http.get(`${BASE_URL}/healthz`);
  if (res.status !== 200) {
    fail(`server not healthy at ${BASE_URL}/healthz (status ${res.status})`);
  }
  const body = res.json();
  console.log(`backend=${body.backend} model=${body.model} imgsz=${body.imgsz}`);
  return { backend: body.backend };
}

function buildRequest() {
  if (MODE === 'json') {
    return {
      body: JSON.stringify({ image_b64: encoding.b64encode(imageBin) }),
      params: {
        headers: { 'Content-Type': 'application/json' },
        tags: { endpoint: 'predict', mode: 'json' },
      },
    };
  }
  return {
    body: { file: http.file(imageBin, 'frame.jpg', 'image/jpeg') },
    params: { tags: { endpoint: 'predict', mode: 'multipart' } },
  };
}

export default function () {
  const { body, params } = buildRequest();
  const res = http.post(`${BASE_URL}/predict`, body, params);

  const ok = check(res, {
    'status is 200': (r) => r.status === 200,
    'has detections array': (r) => {
      try {
        return Array.isArray(r.json('detections'));
      } catch (e) {
        return false;
      }
    },
  });

  errorRate.add(!ok);
  if (!ok) {
    if (res.status === 503) {
      console.warn('503 — queue saturated; server is shedding load');
    }
    return;
  }

  const payload = res.json();
  serverInferMs.add(payload.inference_ms);
  serverTotalMs.add(payload.total_ms);
  observedBatch.add(payload.batch_size);
  detections.add(payload.num_detections);
  if (payload.num_detections === 0) {
    emptyResponses.add(1);
  }
}

export function teardown() {
  // Scrape the server's own histograms at the end of the run for the report.
  const res = http.get(`${BASE_URL}/metrics`);
  if (res.status === 200) {
    const lines = res.body
      .split('\n')
      .filter((l) => l.startsWith('fds_') && !l.includes('_bucket'));
    console.log('--- server metrics ---\n' + lines.join('\n'));
  }
}
