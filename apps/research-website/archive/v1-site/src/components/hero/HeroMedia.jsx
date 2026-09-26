import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";

const HERO_PLAYBACK_RATE = 0.3;
const SEAMLESS_LOOP_FADE_SECONDS = 4;
const LOOP_START_SECONDS = 0.9;
const LOOP_END_TRIM_SECONDS = 0.9;
const HANDOFF_BUFFER_MS = 140;

export default function HeroMedia({
  webmSrc = "/media/research-hero.webm",
  mp4Src = "/media/research-hero.mp4",
  posterSrc = "/media/research-hero-poster.webp",
}) {
  const videoRefs = useRef([]);
  const transitioningRef = useRef(false);
  const cleanupTimerRef = useRef(null);
  const revealTimerRef = useRef(null);
  const reduced = useReducedMotion();
  const [activeLayer, setActiveLayer] = useState(0);
  const [mediaState, setMediaState] = useState("loading");

  const getLoopBounds = (video) => {
    const duration = Number.isFinite(video?.duration) ? video.duration : 0;
    if (duration <= 0) return { loopStart: 0, loopEnd: 0, fadeMediaSeconds: 0 };

    const maxSafeTrim = Math.max(0, duration * 0.12);
    const loopStart = Math.min(LOOP_START_SECONDS, maxSafeTrim);
    const loopEndTrim = Math.min(LOOP_END_TRIM_SECONDS, maxSafeTrim);
    const loopEnd = Math.max(loopStart + 1, duration - loopEndTrim);
    const availableMedia = Math.max(0.25, loopEnd - loopStart);
    const desiredFadeMedia = SEAMLESS_LOOP_FADE_SECONDS * HERO_PLAYBACK_RATE;
    const fadeMediaSeconds = Math.min(desiredFadeMedia, availableMedia * 0.32);

    return { loopStart, loopEnd, fadeMediaSeconds };
  };

  const configureVideo = (video) => {
    if (!video) return;
    video.muted = true;
    video.defaultPlaybackRate = HERO_PLAYBACK_RATE;
    video.playbackRate = HERO_PLAYBACK_RATE;
  };

  const safelyPlay = (video) => {
    if (!video || reduced) return;
    configureVideo(video);
    const playback = video.play();
    playback?.catch?.(() => setMediaState("blocked"));
  };

  const prepareHiddenLayer = (video) => {
    if (!video || !Number.isFinite(video.duration)) return;
    configureVideo(video);
    const { loopStart } = getLoopBounds(video);
    video.pause();
    if (Math.abs(video.currentTime - loopStart) > 0.04)
      video.currentTime = loopStart;
  };

  const revealPreparedLayer = (current, next, nextLayer) => {
    if (!transitioningRef.current) return;

    safelyPlay(next);
    const reveal = () => {
      window.clearTimeout(revealTimerRef.current);
      setActiveLayer(nextLayer);
    };

    if (typeof next.requestVideoFrameCallback === "function") {
      next.requestVideoFrameCallback(reveal);
      revealTimerRef.current = window.setTimeout(reveal, HANDOFF_BUFFER_MS);
    } else {
      revealTimerRef.current = window.setTimeout(reveal, 40);
    }

    window.clearTimeout(cleanupTimerRef.current);
    cleanupTimerRef.current = window.setTimeout(
      () => {
        current.pause();
        prepareHiddenLayer(current);
        transitioningRef.current = false;
      },
      SEAMLESS_LOOP_FADE_SECONDS * 1000 + HANDOFF_BUFFER_MS,
    );
  };

  const handoffToNextLayer = (fromLayer) => {
    if (reduced || transitioningRef.current) return;

    const current = videoRefs.current[fromLayer];
    const nextLayer = fromLayer === 0 ? 1 : 0;
    const next = videoRefs.current[nextLayer];
    if (
      !current ||
      !next ||
      !Number.isFinite(current.duration) ||
      !Number.isFinite(next.duration)
    )
      return;

    transitioningRef.current = true;
    configureVideo(next);
    const { loopStart } = getLoopBounds(next);

    const beginReveal = () => {
      next.removeEventListener("seeked", beginReveal);
      revealPreparedLayer(current, next, nextLayer);
    };

    next.pause();
    next.addEventListener("seeked", beginReveal, { once: true });
    if (Math.abs(next.currentTime - loopStart) <= 0.04) {
      next.removeEventListener("seeked", beginReveal);
      revealPreparedLayer(current, next, nextLayer);
    } else {
      next.currentTime = loopStart;
      revealTimerRef.current = window.setTimeout(() => {
        next.removeEventListener("seeked", beginReveal);
        revealPreparedLayer(current, next, nextLayer);
      }, HANDOFF_BUFFER_MS);
    }
  };

  const handleTimeUpdate = (layer) => {
    if (layer !== activeLayer || transitioningRef.current || reduced) return;
    const video = videoRefs.current[layer];
    if (!video || !Number.isFinite(video.duration) || video.duration <= 0)
      return;

    const { loopEnd, fadeMediaSeconds } = getLoopBounds(video);
    if (video.currentTime >= loopEnd - fadeMediaSeconds)
      handoffToNextLayer(layer);
  };

  const handleLoadedMetadata = (layer) => {
    const video = videoRefs.current[layer];
    if (!video) return;
    configureVideo(video);
    const { loopStart } = getLoopBounds(video);

    if (layer === 0) {
      if (video.currentTime < loopStart - 0.04) video.currentTime = loopStart;
      if (!reduced) safelyPlay(video);
    } else {
      video.pause();
      if (Math.abs(video.currentTime - loopStart) > 0.04)
        video.currentTime = loopStart;
    }
  };

  useEffect(() => {
    const videos = videoRefs.current.filter(Boolean);
    if (!videos.length) return undefined;

    videos.forEach(configureVideo);

    if (reduced) {
      videos.forEach((video) => video.pause());
      setMediaState((state) => (state === "failed" ? state : "paused"));
      return undefined;
    }

    const active = videoRefs.current[activeLayer];
    safelyPlay(active);
    return undefined;
  }, [reduced, activeLayer]);

  useEffect(
    () => () => {
      window.clearTimeout(cleanupTimerRef.current);
      window.clearTimeout(revealTimerRef.current);
    },
    [],
  );

  return (
    <div
      className="hero-media"
      aria-hidden="true"
      data-media-state={mediaState}
    >
      <div className="hero-fallback">
        <i />
        <i />
        <i />
        <i />
      </div>
      {[0, 1].map((layer) => (
        <motion.video
          key={layer}
          className="seamless-video-layer"
          ref={(node) => {
            videoRefs.current[layer] = node;
          }}
          muted
          playsInline
          preload="auto"
          poster={posterSrc}
          initial={false}
          animate={{ opacity: activeLayer === layer ? 1 : 0 }}
          transition={{
            duration: reduced ? 0 : SEAMLESS_LOOP_FADE_SECONDS,
            ease: [0.45, 0, 0.2, 1],
          }}
          onLoadedMetadata={() => handleLoadedMetadata(layer)}
          onLoadedData={() => {
            if (layer === 0) setMediaState(reduced ? "paused" : "ready");
          }}
          onCanPlay={() => {
            if (layer === activeLayer)
              setMediaState(reduced ? "paused" : "ready");
          }}
          onPlaying={() => {
            if (layer === activeLayer) setMediaState("ready");
          }}
          onTimeUpdate={() => handleTimeUpdate(layer)}
          onEnded={() => handoffToNextLayer(layer)}
          onError={() => setMediaState("failed")}
        >
          <source src={mp4Src} type="video/mp4" />
          {webmSrc && <source src={webmSrc} type="video/webm" />}
        </motion.video>
      ))}
    </div>
  );
}
