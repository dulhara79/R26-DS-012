# Aura – Anxiety Research Mobile Application

Aura is a Flutter-based mobile research application developed as part of the
multimodal anxiety research project.

The application supports passive smartphone sensing, questionnaire collection,
wearable integration, behavioural digital phenotyping, and multimodal research
data collection.

## Project Components

The wider research framework contains multiple components:

- Component 1 – Wearable / physiological sensing
- Component 2 – Digital phenotyping and behavioural sensing
- Component 4 – Clinical / NLP component
- Multimodal fusion – combines eligible component outputs

## Component 2 – Digital Phenotyping

Component 2 collects passive smartphone behavioural information and uses it to
construct behavioural features.

Current mobile-compatible sensing includes:

- Location patterns
- Screen and app-use behaviour
- Movement-related information
- Device and collection-status information
- Passive behavioural coverage
- Participant questionnaire/check-in information where applicable

Research data is associated with a pseudonymous participant ID rather than the
participant's display name.

## Component 2 Data Flow

```text
Android App
    ↓
Passive smartphone sensing
    ↓
Local offline queue
    ↓
Supabase
    ↓
sensor_events
    ↓
Daily Component 2 processing
    ↓
daily_behavior_features
    ↓
behavioral_observations