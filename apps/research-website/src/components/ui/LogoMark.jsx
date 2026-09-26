// Brand mark: a heartbeat trace resolving into one calm point. Kept identical
// to public/favicon.svg.
export default function LogoMark({ size = 34 }) {
  return (
    <svg className="logo-mark" width={size} height={size} viewBox="0 0 64 64" aria-hidden="true" focusable="false">
      <rect width="64" height="64" rx="16" fill="#132f2f" />
      <path
        d="M9 36 H20 L24 29 L29 46 L34 16 L39 38 L42 33 H47"
        fill="none"
        stroke="#fdf1e1"
        strokeWidth="4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="53" cy="33" r="4.5" fill="#e8b9a9" />
    </svg>
  );
}
