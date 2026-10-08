import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../theme/c2_palette.dart';

/// Static crisis-resource banner shown on both Component 2 pages.
///
/// It is always visible and never tied to any model output or flagged state.
/// NOTE: the contacts must match the resource list approved in the ethics
/// protocol before recruitment.
class C2CrisisBanner extends StatelessWidget {
  /// Compact mode shows a single line with the helpline chips, for the
  /// Activity tab summary. The full mode adds the explanatory sentence.
  final bool compact;

  const C2CrisisBanner({super.key, this.compact = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: EdgeInsets.all(compact ? 12 : 14),
      decoration: BoxDecoration(
        color: C2Palette.roseBg,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: C2Palette.rose.withValues(alpha: 0.35)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.favorite_rounded, size: 17, color: C2Palette.rose),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  compact
                      ? 'Need to talk to someone now?'
                      : 'If you need to talk to someone right now',
                  style: GoogleFonts.poppins(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w700,
                    color: C2Palette.textPrimary,
                  ),
                ),
                if (!compact) ...[
                  const SizedBox(height: 4),
                  Text(
                    'This app does not monitor you for crisis. These lines '
                    'are staffed by people, any time.',
                    style: GoogleFonts.poppins(
                      fontSize: 11,
                      color: C2Palette.textSecondary,
                      height: 1.4,
                    ),
                  ),
                ],
                const SizedBox(height: 8),
                const Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    _CrisisChip(
                      label: 'National Mental Health Helpline',
                      number: '1926',
                    ),
                    _CrisisChip(
                      label: 'Sri Lanka Sumithrayo',
                      number: '011 2 696 666',
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _CrisisChip extends StatelessWidget {
  final String label;
  final String number;

  const _CrisisChip({required this.label, required this.number});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(20),
      onTap: () async {
        final messenger = ScaffoldMessenger.of(context);
        await Clipboard.setData(ClipboardData(text: number));
        messenger.showSnackBar(
          SnackBar(content: Text('Copied $number to clipboard')),
        );
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
        decoration: BoxDecoration(
          color: C2Palette.cardBase,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: C2Palette.rose.withValues(alpha: 0.4)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.call_rounded, size: 13, color: C2Palette.rose),
            const SizedBox(width: 6),
            Flexible(
              child: Text(
                '$label · $number',
                style: GoogleFonts.poppins(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: C2Palette.textPrimary,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
