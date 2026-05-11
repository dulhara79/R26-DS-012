import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../models/models.dart';
import '../services/patient_provider.dart';
import '../theme/app_theme.dart';
import '../widgets/risk_badge.dart';

class NotificationScreen extends StatelessWidget {
  const NotificationScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<PatientProvider>();
    final notifications = provider.notifications;

    return Scaffold(
      backgroundColor: AppColors.surfaceSecond,
      appBar: AppBar(
        title: const Text('Notifications'),
        actions: [
          if (notifications.isNotEmpty)
            TextButton(
              onPressed: () => provider.clearNotifications(),
              child: const Text('Clear all'),
            ),
        ],
      ),
      body: notifications.isEmpty
          ? _buildEmptyState()
          : ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: notifications.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, index) {
                final notification = notifications[notifications.length - 1 - index];
                return _NotificationItem(notification: notification);
              },
            ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.notifications_none_rounded, size: 64, color: AppColors.textHint.withOpacity(0.5)),
          const SizedBox(height: 16),
          Text(
            'All caught up!',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: AppColors.textSecondary,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'No new notifications to show.',
            style: TextStyle(color: AppColors.textHint),
          ),
        ],
      ).animate().fadeIn(),
    );
  }
}

class _NotificationItem extends StatelessWidget {
  final AppNotification notification;

  const _NotificationItem({required this.notification});

  @override
  Widget build(BuildContext context) {
    final bool isAlert = notification.type == NotificationType.riskAlert;
    final bool isUnread = !notification.isRead;

    return InkWell(
      onTap: () {
        context.read<PatientProvider>().markNotificationAsRead(notification.id);
        // Navigate if needed
      },
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isUnread ? Colors.white : Colors.white.withOpacity(0.7),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isUnread ? AppColors.primary.withOpacity(0.1) : AppColors.border,
            width: isUnread ? 1.5 : 1,
          ),
          boxShadow: isUnread ? [
            BoxShadow(
              color: AppColors.primary.withOpacity(0.05),
              blurRadius: 10,
              offset: const Offset(0, 4),
            )
          ] : [],
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildIcon(isAlert),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        notification.title,
                        style: TextStyle(
                          fontWeight: isUnread ? FontWeight.bold : FontWeight.w600,
                          fontSize: 15,
                          color: isAlert ? AppColors.riskHigh : AppColors.textPrimary,
                        ),
                      ),
                      Text(
                        _formatTime(notification.timestamp),
                        style: const TextStyle(fontSize: 11, color: AppColors.textHint),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    notification.body,
                    style: TextStyle(
                      fontSize: 13,
                      color: isUnread ? AppColors.textSecondary : AppColors.textHint,
                      height: 1.4,
                    ),
                  ),
                  if (notification.riskLevel != null) ...[
                    const SizedBox(height: 10),
                    RiskBadge(risk: notification.riskLevel!),
                  ],
                ],
              ),
            ),
            if (isUnread)
              Container(
                width: 8, height: 8,
                decoration: const BoxDecoration(
                  color: AppColors.primary,
                  shape: BoxShape.circle,
                ),
              ),
          ],
        ),
      ),
    ).animate().fadeIn().slideX(begin: 0.05);
  }

  Widget _buildIcon(bool isAlert) {
    return Container(
      width: 40, height: 40,
      decoration: BoxDecoration(
        color: isAlert ? AppColors.riskHighBg : AppColors.primarySurface,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Icon(
        isAlert ? Icons.warning_amber_rounded : Icons.info_outline_rounded,
        color: isAlert ? AppColors.riskHigh : AppColors.primary,
        size: 20,
      ),
    );
  }

  String _formatTime(DateTime dt) {
    final now = DateTime.now();
    final diff = now.difference(dt);
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    return '${dt.day}/${dt.month}';
  }
}
