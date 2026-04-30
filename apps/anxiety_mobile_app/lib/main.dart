import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:connectivity_plus/connectivity_plus.dart';

import 'theme/app_theme.dart';
import 'pages/login_page.dart';
import 'pages/dashboard_page.dart';
import 'profile_page.dart';
import 'background_service_helper.dart';
import 'services/notification_helper.dart';
import 'services/background/background_service.dart' as bg;
import 'package:shared_preferences/shared_preferences.dart';

final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    await NotificationHelper.init();
  } catch (e, st) {
    debugPrint('Notification init error: $e');
    debugPrint('$st');
  }

  // 1. Queue Retry (Offline Architecture)
  try {
    await BackgroundServiceHelper.retryOfflineQueue();
  } catch (e) {
    debugPrint('Init Queue Retry Error: $e');
  }

  // 2. Connectivity Listener (Auto-Upload when internet returns)
  try {
    Connectivity().onConnectivityChanged.listen((result) async {
      if (result != ConnectivityResult.none) {
        await BackgroundServiceHelper.retryOfflineQueue();
      }
    });
  } catch (e) {
    debugPrint('Connectivity Listener Error: $e');
  }

  // 3. UI System Styling (Edge-to-edge)
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
    ),
  );

  // 4. Initialize Background Service
  try {
    await bg.initializeService();
  } catch (e) {
    debugPrint('Background Service Init Error: $e');
  }

  runApp(const ResearchApp());
}

class ResearchApp extends StatelessWidget {
  const ResearchApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      navigatorKey: navigatorKey,
      title: 'Mindful Tracker',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      home: const SplashRouter(),
    );
  }
}

class SplashRouter extends StatelessWidget {
  const SplashRouter({super.key});

  Future<Widget> _getHome() async {
    final prefs = await SharedPreferences.getInstance();
    final userId = prefs.getString('user_id');
    final profileComplete = prefs.getBool('profile_complete') ?? false;

    if (userId == null || userId.isEmpty) {
      return const LoginPage();
    } else if (!profileComplete) {
      return const ProfilePage();
    } else {
      return DashboardPage(userId: userId);
    }
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Widget>(
      future: _getHome(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }
        return snapshot.data ?? const LoginPage();
      },
    );
  }
}
