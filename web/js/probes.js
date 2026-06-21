// Team-shape "probes" -- interpretable geometric descriptors of a formation,
// computed live as players are dragged. Mirrors value/value/evaluation/shape.py
// and the team-shape features the network itself responds to (SoccerNetV7).
//
// All inputs are sim coordinates: x in [-50, 50], y in [-30, 30]; blue attacks
// +x (own goal at x=-50), red attacks -x (own goal at x=+50).

function centroid(pts) {
  let sx = 0, sy = 0;
  for (const [x, y] of pts) { sx += x; sy += y; }
  return [sx / pts.length, sy / pts.length];
}

// Convex-hull area via Andrew's monotone chain + shoelace.
export function hullArea(pts) {
  if (pts.length < 3) return 0;
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  let area = 0;
  for (let i = 0; i < hull.length; i++) {
    const [x1, y1] = hull[i], [x2, y2] = hull[(i + 1) % hull.length];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

// `team` is the array of [x,y]; index 0 is treated as the goalkeeper and
// excluded from outfield-shape metrics. `side` is +1 if the team attacks +x
// (blue) or -1 if it attacks -x (red). `ball` is [x,y].
export function teamProbes(team, side, ball) {
  const outfield = team.slice(1);            // drop keeper for shape metrics
  const c = centroid(outfield);

  // Compactness / stretch: mean distance to centroid (lower = more compact).
  let stretch = 0;
  for (const [x, y] of outfield) stretch += Math.hypot(x - c[0], y - c[1]);
  stretch /= outfield.length;

  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const [x, y] of outfield) {
    if (x < xmin) xmin = x; if (x > xmax) xmax = x;
    if (y < ymin) ymin = y; if (y > ymax) ymax = y;
  }
  const width = ymax - ymin;
  const depth = xmax - xmin;
  const area = hullArea(outfield);

  // Defensive line height: signed distance the rearmost defender holds from
  // the halfway line, measured up-pitch. For a +x team the "deepest" defender
  // is the smallest x; line height = how far forward that line sits.
  const ownGoalX = -50 * side;
  // rearmost outfielder relative to own goal:
  let lineHeight = -Infinity;
  for (const [x] of outfield) {
    const fromGoal = side > 0 ? x - ownGoalX : ownGoalX - x; // >0 = up-pitch
    if (fromGoal > lineHeight) lineHeight = -Infinity; // placeholder
  }
  // simpler & meaningful: mean up-pitch position of the back unit (4 rearmost)
  const byDepth = outfield
    .map(([x]) => (side > 0 ? x : -x))         // up-pitch coordinate
    .sort((a, b) => a - b);
  const backN = Math.min(4, byDepth.length);
  let backMean = 0;
  for (let i = 0; i < backN; i++) backMean += byDepth[i];
  backMean /= backN;
  lineHeight = backMean - (side > 0 ? ownGoalX : -ownGoalX); // distance from own goal

  // Defenders goalside of the ball (between ball and own goal).
  const ballUp = side > 0 ? ball[0] : -ball[0];
  let goalside = 0;
  for (const [x] of outfield) {
    const up = side > 0 ? x : -x;
    if (up < ballUp) goalside++;
  }

  const centroidToBall = Math.hypot(c[0] - ball[0], c[1] - ball[1]);

  return { stretch, width, depth, area, lineHeight, goalside, centroidToBall };
}
