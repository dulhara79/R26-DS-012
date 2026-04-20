# ============================================================================
# ICD CODE DEFINITIONS
# ============================================================================

# ICD-10 codes for anxiety disorders
ANXIETY_ICD10 = [
    'F410',   # Panic disorder without agoraphobia
    'F411',   # Generalized anxiety disorder (GAD)
    'F412',   # Mixed anxiety and depressive disorder
    'F413',   # Other mixed anxiety disorders
    'F418',   # Other specified anxiety disorders
    'F419',   # Anxiety disorder, unspecified
    'F4000',  # Agoraphobia without panic disorder
    'F4001',  # Agoraphobia with panic disorder
    'F4010',  # Social phobia, unspecified
    'F4011',  # Social phobia, generalized
    'F40210', # Arachnophobia
    'F40218', # Other animal type phobia
    'F40220', # Fear of thunderstorms
    'F40228', # Other natural environment type phobia
    'F40230', # Fear of blood
    'F40231', # Fear of injections and transfusions
    'F40232', # Fear of other medical care
    'F40233', # Fear of injury
    'F40240', # Claustrophobia
    'F40241', # Acrophobia
    'F40242', # Fear of bridges
    'F40243', # Fear of flying
    'F40248', # Other situational type phobia
    'F40290', # Androphobia
    'F40291', # Gynephobia
    'F40298', # Other specified phobia
    'F408',   # Other phobic anxiety disorders
    'F409',   # Phobic anxiety disorder, unspecified
]

# ICD-9 codes for anxiety disorders
ANXIETY_ICD9 = [
    '30000',  # Anxiety state, unspecified
    '30001',  # Panic disorder without agoraphobia
    '30002',  # Generalized anxiety disorder
    '30009',  # Other anxiety states
    '30010',  # Hysteria, unspecified
    '30020',  # Phobia, unspecified
    '30021',  # Agoraphobia with panic attacks
    '30022',  # Agoraphobia without panic attacks
    '30023',  # Social phobia
    '30029',  # Other isolated or specific phobias
    '3003',   # Obsessive-compulsive disorders (often comorbid)
]

# Non-anxiety mental health codes (for control group)
NON_ANXIETY_MENTAL_HEALTH_ICD10 = [
    'F32',   # Depressive episode
    'F33',   # Recurrent depressive disorder
    'F341',  # Dysthymia
    'F31',   # Bipolar disorder
    'F20',   # Schizophrenia
    'F431',  # PTSD
]

NON_ANXIETY_MENTAL_HEALTH_ICD9 = [
    '296',   # Episodic mood disorders
    '311',   # Depressive disorder, not elsewhere classified
    '295',   # Schizophrenic disorders
    '3090',  # Adjustment disorder
]