import { motion, useReducedMotion } from "framer-motion";

const PRESETS = {
  eyebrow: { y: 10, x: 0, blur: 4, duration: 0.48, amount: 0.72 },
  heading: { y: 34, x: 0, blur: 8, duration: 0.78, amount: 0.34 },
  body: { y: 22, x: 0, blur: 6, duration: 0.68, amount: 0.38 },
  support: { y: 14, x: 0, blur: 4, duration: 0.56, amount: 0.52 },
};

export default function TextReveal({
  children,
  as = "div",
  preset = "body",
  className = "",
  delay = 0,
  x,
  y,
  amount,
}) {
  const reduced = useReducedMotion();
  const MotionTag = motion[as] || motion.div;
  const settings = PRESETS[preset] || PRESETS.body;
  const hiddenX = x ?? settings.x;
  const hiddenY = y ?? settings.y;

  return (
    <MotionTag
      className={`text-reveal text-reveal--${preset} ${className}`.trim()}
      data-text-reveal={preset}
      initial={
        reduced
          ? false
          : {
              opacity: 0,
              x: hiddenX,
              y: hiddenY,
              filter: `blur(${settings.blur}px)`,
            }
      }
      whileInView={
        reduced ? undefined : { opacity: 1, x: 0, y: 0, filter: "blur(0px)" }
      }
      viewport={{
        once: false,
        amount: amount ?? settings.amount,
        margin: "-8% 0px -10% 0px",
      }}
      transition={{
        duration: settings.duration,
        delay,
        ease: [0.22, 1, 0.36, 1],
      }}
    >
      {children}
    </MotionTag>
  );
}
