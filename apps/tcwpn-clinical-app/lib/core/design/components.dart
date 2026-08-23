// lib/core/design/components.dart
//
// The shared component vocabulary. Every screen in both research components
// builds from these, which is the mechanical reason the app reads as one
// product.
//
// The signature element is FusionBar: a single stacked bar where each segment's
// width is that modality's weighted contribution to the composite. It is the
// proposal's §5.1 late-fusion equation drawn at full size, and it is the one
// thing a clinician will remember about this app.

import 'package:flutter/material.dart';

import 'theme.dart';
import 'tokens.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Surfaces
// ─────────────────────────────────────────────────────────────────────────────

class Panel extends StatelessWidget {
  final Widget child;
  final EdgeInsets padding;
  final Color? background;
  final Color? borderColor;
  final VoidCallback? onTap;

  const Panel({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(Ds.s4),
    this.background,
    this.borderColor,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final body = Container(
      width: double.infinity,
      padding: padding,
      decoration: BoxDecoration(
        color: background ?? Ds.surface,
        borderRadius: BorderRadius.circular(Ds.rLg),
        border: Border.all(color: borderColor ?? Ds.hairline),
      ),
      child: child,
    );
    if (onTap == null) return body;
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(Ds.rLg),
        child: body,
      ),
    );
  }
}

/// Section eyebrow. The rule and the label together classify the block below,
/// so it earns its place rather than decorating.
class SectionLabel extends StatelessWidget {
  final String text;
  final String? trailing;
  final VoidCallback? onTrailing;

  const SectionLabel(this.text, {super.key, this.trailing, this.onTrailing});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(bottom: Ds.s3),
        child: Row(
          children: [
            Text(text.toUpperCase(), style: AppTheme.eyebrow),
            const SizedBox(width: Ds.s3),
            const Expanded(child: Divider(color: Ds.hairline)),
            if (trailing != null) ...[
              const SizedBox(width: Ds.s3),
              TextButton(
                onPressed: onTrailing,
                style: TextButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: Ds.s2),
                  minimumSize: Size.zero,
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                ),
                child: Text(trailing!),
              ),
            ],
          ],
        ),
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// Indicators
// ─────────────────────────────────────────────────────────────────────────────

class BandChip extends StatelessWidget {
  final AlertBand band;
  final bool large;
  final bool showProtocolName;

  const BandChip({
    super.key,
    required this.band,
    this.large = false,
    this.showProtocolName = false,
  });

  @override
  Widget build(BuildContext context) => Container(
        padding: EdgeInsets.symmetric(
          horizontal: large ? Ds.s4 : Ds.s3,
          vertical: large ? 7 : 4,
        ),
        decoration: BoxDecoration(
          color: band.bg,
          borderRadius: BorderRadius.circular(Ds.rPill),
          border: Border.all(color: band.fg.withValues(alpha: 0.24)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: large ? 8 : 6,
              height: large ? 8 : 6,
              decoration: BoxDecoration(color: band.fg, shape: BoxShape.circle),
            ),
            SizedBox(width: large ? Ds.s2 : 6),
            Text(
              showProtocolName ? band.protocolName : band.label,
              style: TextStyle(
                fontSize: large ? 13 : 11.5,
                fontWeight: FontWeight.w600,
                color: band.fg,
                letterSpacing: 0.2,
              ),
            ),
          ],
        ),
      );
}

/// A labelled measurement. Uses the mono register — this is instrument output.
class Readout extends StatelessWidget {
  final String label;
  final String value;
  final String? unit;
  final Color? valueColor;
  final CrossAxisAlignment align;

  const Readout({
    super.key,
    required this.label,
    required this.value,
    this.unit,
    this.valueColor,
    this.align = CrossAxisAlignment.start,
  });

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: align,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(label, style: AppTheme.eyebrow),
          const SizedBox(height: Ds.s1),
          Row(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Text(value,
                  style: AppTheme.data(
                      size: 18,
                      weight: FontWeight.w600,
                      color: valueColor ?? Ds.ink)),
              if (unit != null) ...[
                const SizedBox(width: 3),
                Text(unit!, style: AppTheme.data(size: 11, color: Ds.inkFaint)),
              ],
            ],
          ),
        ],
      );
}

class MetricTile extends StatelessWidget {
  final String label;
  final String value;
  final String? caption;
  final IconData icon;
  final Color accent;

  const MetricTile({
    super.key,
    required this.label,
    required this.value,
    required this.icon,
    this.caption,
    this.accent = Ds.brand,
  });

  @override
  Widget build(BuildContext context) => Panel(
        padding: const EdgeInsets.all(Ds.s4),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 17, color: accent),
            const SizedBox(height: Ds.s3),
            Text(value,
                style: AppTheme.data(
                    size: 22, weight: FontWeight.w600, color: Ds.ink)),
            const SizedBox(height: 2),
            Text(label,
                style: const TextStyle(
                    fontSize: 12, fontWeight: FontWeight.w600, color: Ds.ink)),
            if (caption != null)
              Text(caption!,
                  style: const TextStyle(fontSize: 11, color: Ds.inkFaint),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis),
          ],
        ),
      );
}

/// Circular confidence indicator. Deliberately does not colour by confidence —
/// confidence is not risk, and colouring it that way invites misreading.
class ConfidenceDial extends StatelessWidget {
  final double value; // 0..1
  final double size;
  final String label;

  const ConfidenceDial({
    super.key,
    required this.value,
    this.size = 56,
    this.label = 'Confidence',
  });

  @override
  Widget build(BuildContext context) {
    final low = value < 0.60;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: size,
          height: size,
          child: Stack(
            alignment: Alignment.center,
            children: [
              TweenAnimationBuilder<double>(
                tween: Tween(begin: 0, end: value.clamp(0.0, 1.0)),
                duration: Ds.med,
                curve: Curves.easeOutCubic,
                builder: (_, v, __) => CircularProgressIndicator(
                  value: v,
                  strokeWidth: 4,
                  strokeCap: StrokeCap.round,
                  backgroundColor: Ds.surfaceSunken,
                  valueColor: AlwaysStoppedAnimation(low ? Ds.amber : Ds.brand),
                ),
              ),
              Text('${(value * 100).round()}',
                  style: AppTheme.data(size: 14, weight: FontWeight.w600)),
            ],
          ),
        ),
        const SizedBox(height: Ds.s2),
        Text(label, style: AppTheme.eyebrow),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Signature: the fusion bar
// ─────────────────────────────────────────────────────────────────────────────

class FusionSegment {
  final String key;
  final String label;

  /// The server's own contribution figure. Null when this modality did not
  /// contribute — which is not the same as contributing zero.
  final double? contribution;
  final double weight;
  final double? score;
  final Color color;

  /// Excluded from the composite by pre-registered rule rather than by absence
  /// (c2_behavioral). Rendered differently from "no reading", because the
  /// distinction is the whole point of keeping it visible.
  final bool excluded;

  /// The server's reason this modality is not contributing, shown verbatim.
  final String? unavailableReason;

  const FusionSegment({
    required this.key,
    required this.label,
    required this.contribution,
    required this.weight,
    required this.score,
    required this.color,
    this.excluded = false,
    this.unavailableReason,
  });

  bool get available => score != null && contribution != null;
}

/// The composite anxiety risk score, drawn as the sum of its parts.
///
/// Each segment's width is that modality's weighted contribution. The empty
/// remainder to the right is "risk not accounted for". Unavailable modalities
/// appear as hatched gaps rather than being silently dropped, so a clinician can
/// see at a glance that the wearable has not reported.
class FusionBar extends StatelessWidget {
  final List<FusionSegment> segments;

  /// NULLABLE. The backend returns `composite: null` with `band: "GREY"`
  /// whenever the fusion gate blocks. Rendering that as 0.000 would present an
  /// assessment the server refused to make as a low-risk result.
  final double? composite;
  final AlertBand band;
  final bool renormalised;
  final bool compact;

  /// Shown in place of the number when there is no composite — the server's own
  /// gate reason, e.g. "insufficient evidence: 1 usable modality, need 2".
  final String? blockedReason;

  /// Backend wire keys. See the mapping table in domain/models.dart: the paper
  /// numbers TC-WPN as Component 4, the backend keys it as c3_clinical_nlp.
  static const Map<String, Color> palette = {
    'c1_physiological': Ds.c1Physiological,
    'c2_behavioral': Ds.c2Behavioral,
    'c3_clinical_nlp': Ds.c3ClinicalNlp,
    'c4_demographic': Ds.c4Demographic,
  };

  const FusionBar({
    super.key,
    required this.segments,
    required this.composite,
    required this.band,
    this.renormalised = false,
    this.compact = false,
    this.blockedReason,
  });

  bool get _hasComposite => composite != null;

  @override
  Widget build(BuildContext context) {
    final present = segments.where((s) => s.available).toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (!compact) ...[
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('COMPOSITE RISK', style: AppTheme.eyebrow),
                  const SizedBox(height: 2),
                  Text(
                    // Em dash, not 0.000. There is no composite to show.
                    _hasComposite ? composite!.toStringAsFixed(3) : '\u2014',
                    style: AppTheme.data(
                        size: 40, weight: FontWeight.w600, color: band.fg),
                  ),
                ],
              ),
              const Spacer(),
              BandChip(band: band, large: true),
            ],
          ),
          const SizedBox(height: Ds.s4),
        ],

        // The bar itself.
        LayoutBuilder(builder: (context, box) {
          final w = box.maxWidth;
          return TweenAnimationBuilder<double>(
            tween: Tween(begin: 0, end: 1),
            duration: Ds.med,
            curve: Curves.easeOutCubic,
            builder: (_, t, __) => SizedBox(
              height: compact ? 8 : 14,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(Ds.rSm),
                child: Stack(
                  children: [
                    Container(color: Ds.surfaceSunken),
                    Row(
                      children: [
                        if (_hasComposite)
                          for (final s in present)
                            SizedBox(
                              width: (s.contribution!.clamp(0.0, 1.0) * w * t),
                              child: Container(color: s.color),
                            ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          );
        }),

        if (!compact) ...[
          if (!_hasComposite) ...[
            const SizedBox(height: Ds.s3),
            InlineNotice(
              icon: Icons.block_flipped,
              tone: Ds.grey,
              text: blockedReason ??
                  'No composite could be computed from the readings available. '
                      'This is missing evidence, not a low score.',
            ),
          ],
          const SizedBox(height: Ds.s4),
          // Legend doubles as the per-modality breakdown.
          ...segments.map((s) => _legendRow(s)),
          if (renormalised) ...[
            const SizedBox(height: Ds.s3),
            const InlineNotice(
              icon: Icons.balance_rounded,
              text:
                  'One or more modalities have no current reading. Weights were '
                  'rescaled across the modalities that reported, so this composite '
                  'rests on partial evidence.',
            ),
          ],
        ],
      ],
    );
  }

  Widget _legendRow(FusionSegment s) => Padding(
        padding: const EdgeInsets.only(bottom: Ds.s3),
        child: Row(
          children: [
            Container(
              width: 10,
              height: 10,
              decoration: BoxDecoration(
                color: s.available ? s.color : Colors.transparent,
                border:
                    s.available ? null : Border.all(color: Ds.hairlineStrong),
                borderRadius: BorderRadius.circular(3),
              ),
            ),
            const SizedBox(width: Ds.s3),
            Expanded(
              child: Text(
                s.label,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                  color: s.available ? Ds.ink : Ds.inkFaint,
                ),
              ),
            ),
            if (s.available) ...[
              Text('w ${s.weight.toStringAsFixed(2)}',
                  style: AppTheme.data(size: 11, color: Ds.inkFaint)),
              const SizedBox(width: Ds.s3),
              Text('× ${s.score!.toStringAsFixed(2)}',
                  style: AppTheme.data(size: 11, color: Ds.inkMuted)),
              const SizedBox(width: Ds.s3),
              SizedBox(
                width: 46,
                child: Text(
                  s.contribution!.toStringAsFixed(3),
                  textAlign: TextAlign.right,
                  style: AppTheme.data(size: 12.5, weight: FontWeight.w600),
                ),
              ),
            ] else if (s.excluded)
              // A pre-registered exclusion is not a missing reading. Saying
              // "no reading" here would hide the fact that the model reported
              // and was deliberately not used.
              Text('excluded', style: AppTheme.data(size: 11, color: Ds.amber))
            else if (s.score != null)
              Text('not used',
                  style: AppTheme.data(size: 11, color: Ds.inkFaint))
            else
              Text('no reading',
                  style: AppTheme.data(size: 11, color: Ds.inkFaint)),
          ],
        ),
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// Messaging
// ─────────────────────────────────────────────────────────────────────────────

/// Quiet in-context notice. Not an alert — it explains a condition of the data.
class InlineNotice extends StatelessWidget {
  final IconData icon;
  final String text;
  final Color? tone;

  const InlineNotice({
    super.key,
    required this.icon,
    required this.text,
    this.tone,
  });

  @override
  Widget build(BuildContext context) {
    final c = tone ?? Ds.brand;
    return Container(
      padding: const EdgeInsets.all(Ds.s3),
      decoration: BoxDecoration(
        color: c.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(Ds.rMd),
        border: Border.all(color: c.withValues(alpha: 0.18)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 15, color: c),
          const SizedBox(width: Ds.s3),
          Expanded(
            child: Text(text,
                style: TextStyle(fontSize: 12.5, color: c, height: 1.45)),
          ),
        ],
      ),
    );
  }
}

/// An empty screen is an invitation to act, so every empty state names the
/// action that fills it.
class EmptyState extends StatelessWidget {
  final IconData icon;
  final String title;
  final String body;
  final String? actionLabel;
  final VoidCallback? onAction;

  const EmptyState({
    super.key,
    required this.icon,
    required this.title,
    required this.body,
    this.actionLabel,
    this.onAction,
  });

  @override
  Widget build(BuildContext context) => Center(
        child: Padding(
          padding: const EdgeInsets.all(Ds.s8),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 52,
                height: 52,
                decoration: BoxDecoration(
                  color: Ds.surfaceSunken,
                  borderRadius: BorderRadius.circular(Ds.rMd),
                ),
                child: Icon(icon, color: Ds.inkFaint, size: 24),
              ),
              const SizedBox(height: Ds.s4),
              Text(title, style: AppTheme.display(size: 16)),
              const SizedBox(height: Ds.s2),
              Text(body,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                      fontSize: 13, color: Ds.inkMuted, height: 1.5)),
              if (actionLabel != null) ...[
                const SizedBox(height: Ds.s5),
                SizedBox(
                  width: 220,
                  child: OutlinedButton(
                      onPressed: onAction, child: Text(actionLabel!)),
                ),
              ],
            ],
          ),
        ),
      );
}

/// Full-bleed working state used while a model is running. Names the model, so
/// a 90-second cold start doesn't read as a hang.
class WorkingOverlay extends StatelessWidget {
  final bool active;
  final String message;
  final Widget child;

  const WorkingOverlay({
    super.key,
    required this.active,
    required this.child,
    this.message = 'Working',
  });

  @override
  Widget build(BuildContext context) => Stack(
        children: [
          child,
          if (active)
            Positioned.fill(
              child: AnimatedOpacity(
                opacity: active ? 1 : 0,
                duration: Ds.fast,
                child: ColoredBox(
                  color: Ds.canvas.withValues(alpha: 0.86),
                  child: Center(
                    child: Panel(
                      padding: const EdgeInsets.symmetric(
                          horizontal: Ds.s6, vertical: Ds.s5),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const SizedBox(
                            width: 22,
                            height: 22,
                            child: CircularProgressIndicator(
                                strokeWidth: 2.4, color: Ds.brand),
                          ),
                          const SizedBox(height: Ds.s4),
                          Text(message,
                              textAlign: TextAlign.center,
                              style: const TextStyle(
                                  fontSize: 13, color: Ds.inkMuted)),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
        ],
      );
}

/// Standing clinical-decision-support notice. Appears wherever a model output is
/// the primary content on screen.
class DecisionSupportNotice extends StatelessWidget {
  const DecisionSupportNotice({super.key});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(Ds.s4),
        decoration: BoxDecoration(
          color: Ds.surfaceSunken,
          borderRadius: BorderRadius.circular(Ds.rMd),
          border: Border.all(color: Ds.hairline),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.gavel_rounded, size: 15, color: Ds.inkMuted),
            const SizedBox(width: Ds.s3),
            Expanded(
              child: RichText(
                text: TextSpan(
                  style: const TextStyle(
                      fontSize: 12, color: Ds.inkMuted, height: 1.5),
                  children: const [
                    TextSpan(
                      text: 'Clinical decision support. ',
                      style:
                          TextStyle(fontWeight: FontWeight.w700, color: Ds.ink),
                    ),
                    TextSpan(
                      text:
                          'These outputs do not constitute a diagnosis. Diagnosis, '
                          'treatment, and referral remain the responsibility of the '
                          'treating psychiatrist.',
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      );
}
