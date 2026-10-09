class AssignedSubject {
  final String subjectId;

  const AssignedSubject(this.subjectId);
}

abstract interface class AssignmentInviteRepository {
  /// Redeems a one-use code issued by the patient for the authenticated clinician.
  Future<AssignedSubject> redeem(String inviteCode);
}
