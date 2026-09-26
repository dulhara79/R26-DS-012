import { useState } from "react";
import { images } from "../../data/images";

// Shows the image registered under `name` once its file exists in public/.
// Until then it renders a styled placeholder; in dev it also shows what to add.
export default function ImageSlot({ name, alt = "", className = "", ratio }) {
  const slot = images[name];
  const [missing, setMissing] = useState(false);
  if (!slot) return null;

  return (
    <div
      className={`image-slot ${missing ? "image-slot--missing" : ""} ${className}`}
      style={{ aspectRatio: ratio ?? slot.ratio }}
    >
      {!missing && (
        <img
          src={`${import.meta.env.BASE_URL}${slot.path.replace(/^\//, "")}`}
          alt={alt}
          loading="lazy"
          decoding="async"
          onError={() => setMissing(true)}
        />
      )}
      {missing && import.meta.env.DEV && (
        <div className="image-slot-guide">
          <strong>Add image · {slot.size}</strong>
          <code>public{slot.path}</code>
        </div>
      )}
    </div>
  );
}
