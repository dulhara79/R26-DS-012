// Resolve the active main navigation tab for each visible route.
// Research detail pages belong to Domain, rather than leaving every tab inactive.
export function navParentForPath(pathname) {
  const path = (pathname || "/").replace(/\/+$/, "") || "/";
  if (path === "/research" || path === "/components" || path.startsWith("/components/") ||
      path === "/system" || path === "/findings" || path === "/results") {
    return "/research";
  }
  if (path === "/publications") return "/documents";
  return path;
}
