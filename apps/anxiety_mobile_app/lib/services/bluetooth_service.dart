// BleManager has been replaced by ChestStrapService.
// See: chest_strap_service.dart
//
// This file is kept to avoid breaking any remaining imports.
// All functionality has moved to ChestStrapService.

import 'chest_strap_service.dart';

@Deprecated('Use ChestStrapService instead')
typedef BleManager = ChestStrapService;
