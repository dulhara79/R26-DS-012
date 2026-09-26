export function chapterIndexFromProgress(progress, chapterCount) {
  if (!Number.isFinite(chapterCount) || chapterCount <= 0) return 0;
  const normalized = Math.min(
    1,
    Math.max(0, Number.isFinite(progress) ? progress : 0),
  );
  return Math.min(chapterCount - 1, Math.floor(normalized * chapterCount));
}
