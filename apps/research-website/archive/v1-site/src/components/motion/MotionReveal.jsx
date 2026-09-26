import { motion, useReducedMotion } from "framer-motion";

export default function MotionReveal({
  children,
  className = "",
  delay = 0,
  y = 28,
  amount = 0.2,
  as = "div",
}) {
  const reduced = useReducedMotion();
  const MotionTag = motion[as] || motion.div;
  return (
    <MotionTag
      className={`motion-reveal ${className}`.trim()}
      initial={reduced ? false : { opacity: 0, y }}
      whileInView={reduced ? undefined : { opacity: 1, y: 0 }}
      viewport={{ once: false, amount, margin: "-8% 0px -10% 0px" }}
      transition={{ duration: 0.72, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </MotionTag>
  );
}
