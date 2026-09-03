import 'package:anxiety_mobile_app/pages/share_participant_id_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:qr_flutter/qr_flutter.dart';

void main() {
  testWidgets('renders QR and raw Aura participant ID', (tester) async {
    const participantId = 'P_0123456789ABCDEF';

    await tester.pumpWidget(
      const MaterialApp(
        home: ShareParticipantIdPage(participantId: participantId),
      ),
    );

    expect(find.byType(QrImageView), findsOneWidget);
    expect(
      find.byWidgetPredicate(
        (widget) => widget is SelectableText && widget.data == participantId,
      ),
      findsOneWidget,
    );
  });
}
