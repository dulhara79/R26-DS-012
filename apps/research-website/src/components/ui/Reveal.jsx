import { motion, useReducedMotion } from "framer-motion";

// Fades content up once as it enters the viewport.
export default function Reveal({
  as = "div",
  delay = 0,
  className,
  children,
  ...rest
}) {
  const reduced = useReducedMotion();
  const Component = motion[as] ?? motion.div;
  return (
    <Component
      className={className}
      initial={reduced ? false : { opacity: 0, y: 28 }}
      whileInView={reduced ? undefined : { opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "0px 0px -12% 0px" }}
      transition={{ duration: 0.8, delay, ease: [0.22, 1, 0.36, 1] }}
      {...rest}
    >
      {children}
    </Component>
  );
}
