# TC-WPN Clinical: Anxiety Detection Dashboard

[![Flutter](https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white)](https://flutter.dev)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces)

TC-WPN Clinical is a professional-grade mobile application designed for clinicians and researchers to monitor, assess, and detect anxiety levels in patients using AI-driven clinical note analysis.

<p align="center">
  <img src="assets/images/app_icon.png" width="120" alt="TC-WPN Icon">
</p>

## 🌟 Key Features

-   **AI-Powered Assessment**: Real-time anxiety detection using machine learning models hosted on Hugging Face Spaces.
-   **Clinical Dashboard**: Comprehensive overview of patient lists, risk levels, and assessment histories.
-   **Risk Categorization**: Intelligent classification of patient states into four levels: Low, Moderate, High, and Very High.
-   **Dynamic Support Sets**: Manage and tune the "Support Notes" used by the underlying detection algorithm for improved accuracy.
-   **Rich Visualizations**: Interactive charts for tracking longitudinal patient data and risk trends.
-   **Premium UI/UX**: Soft light theme designed for clinical environments, featuring smooth animations and high readability.

## 🛠 Tech Stack

-   **Framework**: [Flutter](https://flutter.dev)
-   **State Management**: [Provider](https://pub.dev/packages/provider)
-   **Backend/ML**: Hugging Face Spaces (Python/Transformers)
-   **Networking**: [http](https://pub.dev/packages/http)
-   **Visualizations**: [fl_chart](https://pub.dev/packages/fl_chart)
-   **Animations**: [flutter_animate](https://pub.dev/packages/flutter_animate), [Lottie](https://pub.dev/packages/lottie)
-   **Theming**: Custom Clinical Teal Palette with [Google Fonts (Inter)](https://fonts.google.com/specimen/Inter)

## 🚀 Getting Started

### Prerequisites

-   Flutter SDK (v3.0.0 or higher)
-   Android Studio / Xcode (for mobile deployment)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/dulhara79/tcwpn_app.git
    cd tcwpn_app
    ```

2.  **Install dependencies**:
    ```bash
    flutter pub get
    ```

3.  **Generate App Icons** (if modified):
    ```bash
    dart run flutter_launcher_icons
    ```

4.  **Run the application**:
    ```bash
    flutter run
    ```

## 📂 Project Structure

```text
lib/
├── models/          # Data structures (Patient, Assessment, PredictionResult)
├── screens/         # UI Screens (Dashboard, Patient Details, New Assessment)
├── services/        # API integration and State Providers
├── theme/           # Clinical design system and color palettes
└── widgets/         # Reusable UI components and animated builders
```

## 🔐 Security & Privacy

This application is designed with clinical standards in mind. Telemetry is restricted to necessary clinical assessments, and data synchronization follows isolate-safe persistence logic to ensure reliability during research data collection.

---

Designed and developed for **Advanced Anxiety Research**.
