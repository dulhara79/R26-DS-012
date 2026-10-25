// Pure navigation scroll state: enough directional travel is required to
// change visibility, so tiny wheel/touchpad movements do not make the bar jitter.
export const NAV_TOP_BOUNDARY = 84;
export const NAV_HIDE_BOUNDARY = 110;
export const NAV_HIDE_TRAVEL = 14;
export const NAV_SHOW_TRAVEL = 10;

function position(value) {
  return Number.isFinite(value) ? Math.max(0, value) : 0;
}

export function initialNavScroll(scrollY = 0) {
  const y = position(scrollY);
  return {
    y,
    mode: y <= NAV_TOP_BOUNDARY ? "full" : "compact",
    direction: null,
    travel: 0,
  };
}

export function advanceNavScroll(previous, scrollY) {
  const y = position(scrollY);
  if (y <= NAV_TOP_BOUNDARY) return initialNavScroll(y);

  const delta = y - previous.y;
  if (Math.abs(delta) < 0.5) return { ...previous, y };
  const direction = delta > 0 ? "down" : "up";
  const travel = (direction === previous.direction ? previous.travel : 0) + Math.abs(delta);

  let mode = previous.mode === "full" ? "compact" : previous.mode;
  if (direction === "down" && y >= NAV_HIDE_BOUNDARY && travel >= NAV_HIDE_TRAVEL) {
    mode = "hidden";
  } else if (direction === "up" && travel >= NAV_SHOW_TRAVEL) {
    mode = "compact";
  }
  return { y, mode, direction, travel };
}
