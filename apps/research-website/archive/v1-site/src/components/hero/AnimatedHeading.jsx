import { useReducedMotion } from "framer-motion";

export default function AnimatedHeading({
  lines = ["Understanding anxiety", "beyond a single moment."],
  initialDelay = 200,
  charDelay = 30,
  duration = 500,
}) {
  const reduced = useReducedMotion();
  let cursor = 0;
  return (
    <h1
      className="animated-heading hero-heading-safe"
      aria-label={lines.join(" ")}
    >
      {lines.map((line, lineIndex) => (
        <span className="heading-line" aria-hidden="true" key={lineIndex}>
          {[...line].map((char, charIndex) => {
            const delay = reduced ? 0 : initialDelay + cursor++ * charDelay;
            return (
              <span
                className="heading-char"
                key={`${lineIndex}-${charIndex}`}
                style={{
                  animationDelay: `${delay}ms`,
                  animationDuration: `${reduced ? 0 : duration}ms`,
                }}
              >
                {char === " " ? "\u00A0" : char}
              </span>
            );
          })}
        </span>
      ))}
    </h1>
  );
}
