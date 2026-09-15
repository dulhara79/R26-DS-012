enum AsyncDataStatus {
  loading,
  data,
  empty,
  partial,
  unavailable,
  offline,
  error,
  sessionExpired,
  forbidden,
  conflict,
}

class AsyncDataState<T> {
  final AsyncDataStatus status;
  final T? data;
  final String? message;

  const AsyncDataState._(this.status, {this.data, this.message});

  const AsyncDataState.loading()
      : this._(AsyncDataStatus.loading);

  const AsyncDataState.data(T value)
      : this._(AsyncDataStatus.data, data: value);

  const AsyncDataState.empty()
      : this._(AsyncDataStatus.empty);

  const AsyncDataState.partial(T value, {String? message})
      : this._(AsyncDataStatus.partial, data: value, message: message);

  const AsyncDataState.unavailable({String? message})
      : this._(AsyncDataStatus.unavailable, message: message);

  const AsyncDataState.offline({T? cached, String? message})
      : this._(AsyncDataStatus.offline, data: cached, message: message);

  const AsyncDataState.error({String? message})
      : this._(AsyncDataStatus.error, message: message);

  const AsyncDataState.sessionExpired({String? message})
      : this._(AsyncDataStatus.sessionExpired, message: message);

  const AsyncDataState.forbidden({String? message})
      : this._(AsyncDataStatus.forbidden, message: message);

  const AsyncDataState.conflict({String? message})
      : this._(AsyncDataStatus.conflict, message: message);

  bool get hasData => data != null;
}
