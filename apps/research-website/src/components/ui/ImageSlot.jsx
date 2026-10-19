import { useState } from "react";
import { imageDriveIds, imageSrc, images } from "../../data/images";

export default function ImageSlot({ name, alt = "", className = "", ratio }) {
  const slot = images[name];
  const [missing, setMissing] = useState(false);
  if (!slot) return null;
  const fileId = imageDriveIds[name];
  return (
    <div className={`image-slot ${missing ? "image-slot--missing" : ""} ${className}`} style={{ aspectRatio: ratio ?? slot.ratio }}>
      {!missing && (
        <img src={imageSrc(name, import.meta.env.BASE_URL)} alt={alt}
          loading={name.startsWith("banner") ? "eager" : "lazy"}
          decoding="async" referrerPolicy="no-referrer"
          onError={() => setMissing(true)} />
      )}
      {missing && fileId && (
        <a className="image-slot-guide"
          href={`https://drive.google.com/file/d/${fileId}/view`}
          target="_blank" rel="noopener noreferrer"
          aria-label={alt ? `Open ${alt} on Google Drive` : "Open image on Google Drive"}>
          Image unavailable here · View on Drive ↗
        </a>
      )}
      {missing && !fileId && import.meta.env.DEV && (
        <div className="image-slot-guide"><strong>Add image · {slot.size}</strong><code>public{slot.path}</code></div>
      )}
    </div>
  );
}
