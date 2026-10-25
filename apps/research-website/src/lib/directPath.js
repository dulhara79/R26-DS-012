// Vercel serves index.html for direct SPA paths, but this website retains
// HashRouter links. Normalize direct paths before React mounts so both forms
// resolve through the same route tree and custom unknown-route page.
export function hashUrlForDirectPath({
  pathname = "/",
  search = "",
  hash = "",
} = {}) {
  if (pathname === "/" || pathname === "/index.html") return null;

  // If a hash route is already provided, honor it rather than taking an
  // unrelated URL pathname. Avoid treating ordinary #section anchors as routes.
  if (hash.startsWith("#/")) return "/" + hash;

  // A path like /research#literature-title becomes
  // /#/research#literature-title, retaining the router's section hash.
  return "/#" + pathname + search + hash;
}

export function normalizeDirectPath(location, browserHistory) {
  const canonical = hashUrlForDirectPath(location);
  if (canonical) {
    browserHistory.replaceState(browserHistory.state, "", canonical);
  }
  return canonical;
}
