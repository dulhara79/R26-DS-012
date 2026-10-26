// One source of geometry for the decorative SVG trail and its interactive stops.
// The highlighted trail indicates the selected stage, not verified completion.
export const JOURNEY_WIDTH = 700;
export const JOURNEY_ROW_HEIGHT = 188;
export const JOURNEY_START_Y = 145;
export const JOURNEY_MOBILE_WIDTH = 104;

export function journeyHeight(count) {
  return JOURNEY_START_Y + Math.max(0, count - 1) * JOURNEY_ROW_HEIGHT + 120;
}

export function journeyStops(count, mobile = false) {
  return Array.from({ length: count }, (_, index) => ({
    index,
    side: index % 2 === 0 ? "left" : "right",
    x: mobile ? (index % 2 === 0 ? 46 : 65) : (index % 2 === 0 ? 270 : 430),
    y: JOURNEY_START_Y + index * JOURNEY_ROW_HEIGHT,
  }));
}

export function journeyPath(points) {
  if (!points.length) return "";
  let path = `M ${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length; i += 1) {
    const before = points[i - 1];
    const current = points[i];
    const middleY = (before.y + current.y) / 2;
    path += ` C ${before.x} ${middleY}, ${current.x} ${middleY}, ${current.x} ${current.y}`;
  }
  return path;
}
