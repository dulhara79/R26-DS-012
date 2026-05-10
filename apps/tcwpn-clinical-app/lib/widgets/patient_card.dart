import 'package:flutter/material.dart';
import '../models/models.dart';
import '../theme/app_theme.dart';
import 'risk_badge.dart';

class PatientCard extends StatelessWidget {
  final Patient patient;
  final VoidCallback onTap;

  const PatientCard({super.key, required this.patient, required this.onTap});

  Color get _avatarColor {
    switch (patient.latestRisk) {
      case RiskLevel.veryHigh: return AppColors.riskVeryHighBg;
      case RiskLevel.high:     return AppColors.riskHighBg;
      case RiskLevel.moderate: return AppColors.riskModerateBg;
      case RiskLevel.low:      return AppColors.primarySurface;
    }
  }

  Color get _avatarText {
    switch (patient.latestRisk) {
      case RiskLevel.veryHigh: return AppColors.riskVeryHigh;
      case RiskLevel.high:     return AppColors.riskHigh;
      case RiskLevel.moderate: return AppColors.riskModerate;
      case RiskLevel.low:      return AppColors.primary;
    }
  }

  @override
  Widget build(BuildContext context) {
    final last = patient.latestAssessment;
    return Material(
      color: AppColors.surface,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: patient.hasAlert
                  ? AppColors.riskHigh.withOpacity(0.4)
                  : AppColors.border,
              width: patient.hasAlert ? 1.2 : 0.8,
            ),
          ),
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              // Avatar
              Stack(
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: _avatarColor,
                      shape: BoxShape.circle,
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      patient.initials,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: _avatarText,
                      ),
                    ),
                  ),
                  if (patient.hasAlert)
                    Positioned(
                      right: 0,
                      top: 0,
                      child: Container(
                        width: 12,
                        height: 12,
                        decoration: BoxDecoration(
                          color: AppColors.riskHigh,
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 2),
                        ),
                      ),
                    ),
                ],
              ),
              const SizedBox(width: 14),
              // Details
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            patient.name,
                            style: Theme.of(context).textTheme.titleMedium,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        const SizedBox(width: 8),
                        RiskBadge(risk: patient.latestRisk),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${patient.age}y · ${patient.gender} · ${patient.ward}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        Icon(Icons.history_rounded,
                            size: 13, color: AppColors.textHint),
                        const SizedBox(width: 4),
                        Text(
                          '${patient.totalVisits} visits',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        if (last != null) ...[
                          const SizedBox(width: 10),
                          Icon(Icons.access_time_rounded,
                              size: 13, color: AppColors.textHint),
                          const SizedBox(width: 4),
                          Text(
                            _formatDate(last.timestamp),
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Icon(Icons.chevron_right_rounded,
                  size: 20, color: AppColors.textHint),
            ],
          ),
        ),
      ),
    );
  }

  String _formatDate(DateTime dt) {
    final diff = DateTime.now().difference(dt);
    if (diff.inDays == 0) return 'Today';
    if (diff.inDays == 1) return 'Yesterday';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return '${dt.day}/${dt.month}/${dt.year}';
  }
}
