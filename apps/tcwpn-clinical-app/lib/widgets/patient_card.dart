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
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 2, vertical: 4),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
        border: Border.all(
          color: patient.hasAlert
              ? AppColors.riskHigh.withOpacity(0.3)
              : AppColors.border,
          width: patient.hasAlert ? 1.5 : 1,
        ),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              // Avatar
              Stack(
                children: [
                  Container(
                    width: 54,
                    height: 54,
                    decoration: BoxDecoration(
                      color: _avatarColor,
                      borderRadius: BorderRadius.circular(16),
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      patient.initials,
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: _avatarText,
                      ),
                    ),
                  ),
                  if (patient.hasAlert)
                    Positioned(
                      right: -2,
                      top: -2,
                      child: Container(
                        width: 14,
                        height: 14,
                        decoration: BoxDecoration(
                          color: AppColors.riskHigh,
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 2.5),
                        ),
                      ),
                    ),
                ],
              ),
              const SizedBox(width: 16),
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
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: AppColors.textPrimary,
                            ),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        const SizedBox(width: 8),
                        RiskBadge(risk: patient.latestRisk),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      '${patient.age}y · ${patient.gender} · ${patient.ward}',
                      style: const TextStyle(
                        fontSize: 12,
                        color: AppColors.textSecondary,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        _buildSmallChip(
                          Icons.history_rounded,
                          '${patient.totalVisits} visits',
                        ),
                        if (last != null) ...[
                          const SizedBox(width: 12),
                          _buildSmallChip(
                            Icons.calendar_today_rounded,
                            _formatDate(last.timestamp),
                          ),
                        ],
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              const Icon(Icons.chevron_right_rounded,
                  size: 22, color: AppColors.textHint),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSmallChip(IconData icon, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 12, color: AppColors.textHint),
        const SizedBox(width: 4),
        Text(
          label,
          style: const TextStyle(
            fontSize: 11,
            color: AppColors.textHint,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
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
