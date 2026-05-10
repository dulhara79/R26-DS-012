import 'package:flutter/material.dart';
import '../models/models.dart';
import '../theme/app_theme.dart';

class RiskBadge extends StatelessWidget {
  final RiskLevel risk;
  final bool large;

  const RiskBadge({super.key, required this.risk, this.large = false});

  Color get _bg {
    switch (risk) {
      case RiskLevel.low:      return AppColors.riskLowBg;
      case RiskLevel.moderate: return AppColors.riskModerateBg;
      case RiskLevel.high:     return AppColors.riskHighBg;
      case RiskLevel.veryHigh: return AppColors.riskVeryHighBg;
    }
  }

  Color get _fg {
    switch (risk) {
      case RiskLevel.low:      return AppColors.riskLow;
      case RiskLevel.moderate: return AppColors.riskModerate;
      case RiskLevel.high:     return AppColors.riskHigh;
      case RiskLevel.veryHigh: return AppColors.riskVeryHigh;
    }
  }

  @override
  Widget build(BuildContext context) {
    final fs = large ? 13.0 : 11.0;
    final px = large ? 12.0 : 8.0;
    final py = large ? 6.0  : 3.0;

    return Container(
      padding: EdgeInsets.symmetric(horizontal: px, vertical: py),
      decoration: BoxDecoration(
        color: _bg,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: _fg.withOpacity(0.3), width: 0.8),
      ),
      child: Text(
        risk.shortLabel,
        style: TextStyle(
          fontSize: fs,
          fontWeight: FontWeight.w600,
          color: _fg,
          letterSpacing: 0.3,
        ),
      ),
    );
  }
}

class RiskScoreBar extends StatelessWidget {
  final double score;
  final double threshold;

  const RiskScoreBar({
    super.key,
    required this.score,
    this.threshold = 0.4036,
  });

  Color get _color {
    if (score >= 0.85) return AppColors.riskVeryHigh;
    if (score >= 0.70) return AppColors.riskHigh;
    if (score >= threshold) return AppColors.riskModerate;
    return AppColors.riskLow;
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Risk score',
              style: Theme.of(context).textTheme.labelSmall,
            ),
            Text(
              score.toStringAsFixed(4),
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                color: _color, fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Stack(
          children: [
            Container(
              height: 10,
              decoration: BoxDecoration(
                color: AppColors.border,
                borderRadius: BorderRadius.circular(5),
              ),
            ),
            FractionallySizedBox(
              widthFactor: score.clamp(0.0, 1.0),
              child: Container(
                height: 10,
                decoration: BoxDecoration(
                  color: _color,
                  borderRadius: BorderRadius.circular(5),
                ),
              ),
            ),
            // Threshold marker
            FractionallySizedBox(
              widthFactor: threshold,
              child: Align(
                alignment: Alignment.centerRight,
                child: Container(
                  width: 2,
                  height: 10,
                  color: AppColors.textSecondary,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text('0.0', style: Theme.of(context).textTheme.bodySmall),
            Text(
              'threshold ${threshold.toStringAsFixed(2)}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            Text('1.0', style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
      ],
    );
  }
}
