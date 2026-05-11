import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../models/models.dart';
import '../services/patient_provider.dart';
import '../widgets/patient_card.dart';
import 'patient_detail_screen.dart';
import 'add_patient_screen.dart';

class PatientListScreen extends StatefulWidget {
  const PatientListScreen({super.key});

  @override
  State<PatientListScreen> createState() => _PatientListScreenState();
}

class _PatientListScreenState extends State<PatientListScreen> {
  final _searchCtrl = TextEditingController();
  String _query = '';
  RiskLevel? _riskFilter;
  String _wardFilter = 'All';

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final provider  = context.watch<PatientProvider>();
    final filtered  = provider.searchPatients(
      _query, 
      riskFilter: _riskFilter,
      wardFilter: _wardFilter,
    );

    return Scaffold(
      backgroundColor: AppColors.surfaceSecond,
      appBar: AppBar(
        title: const Text('Patients'),
      ),
      body: Column(
        children: [
          // Search bar
          Container(
            color: AppColors.surface,
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
            child: TextField(
              controller: _searchCtrl,
              onChanged: (v) => setState(() => _query = v),
              decoration: InputDecoration(
                hintText: 'Search by name or patient ID...',
                prefixIcon: const Icon(Icons.search_rounded, size: 20),
                suffixIcon: _query.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear_rounded, size: 18),
                        onPressed: () {
                          _searchCtrl.clear();
                          setState(() => _query = '');
                        },
                      )
                    : null,
              ),
            ),
          ),
          
          // Filters
          Container(
            height: 48,
            color: AppColors.surface,
            child: ListView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 16),
              children: [
                _FilterChip(
                  label: 'All', 
                  isSelected: _riskFilter == null && _wardFilter == 'All', 
                  onTap: () => setState(() {
                    _riskFilter = null;
                    _wardFilter = 'All';
                  }),
                ),
                _FilterChip(
                  label: 'High Risk', 
                  isSelected: _riskFilter == RiskLevel.high, 
                  onTap: () => setState(() => _riskFilter = RiskLevel.high), 
                  color: AppColors.riskHigh,
                ),
                _FilterChip(
                  label: 'Moderate', 
                  isSelected: _riskFilter == RiskLevel.moderate, 
                  onTap: () => setState(() => _riskFilter = RiskLevel.moderate), 
                  color: AppColors.riskModerate,
                ),
                _FilterChip(
                  label: 'Low Risk', 
                  isSelected: _riskFilter == RiskLevel.low, 
                  onTap: () => setState(() => _riskFilter = RiskLevel.low), 
                  color: AppColors.riskLow,
                ),
                _FilterChip(
                  label: 'Psychiatry OPD', 
                  isSelected: _wardFilter == 'Psychiatry OPD', 
                  onTap: () => setState(() => _wardFilter = 'Psychiatry OPD'),
                ),
              ],
            ),
          ),
          const Divider(height: 0),

          // Results count
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
            child: Row(
              children: [
                Text(
                  '${filtered.length} patient${filtered.length != 1 ? 's' : ''}',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                if (_riskFilter != null || _wardFilter != 'All' || _query.isNotEmpty) ...[
                  const SizedBox(width: 8),
                  GestureDetector(
                    onTap: () => setState(() {
                      _riskFilter = null;
                      _wardFilter = 'All';
                      _query = '';
                      _searchCtrl.clear();
                    }),
                    child: const Text(
                      'Clear all',
                      style: TextStyle(fontSize: 12, color: AppColors.primary, fontWeight: FontWeight.bold),
                    ),
                  ),
                ],
              ],
            ),
          ),

          // Patient list
          Expanded(
            child: filtered.isEmpty
                ? Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.search_off_rounded,
                            size: 48, color: AppColors.textHint),
                        const SizedBox(height: 12),
                        Text(
                          'No patients found',
                          style: Theme.of(context).textTheme.bodyLarge
                              ?.copyWith(color: AppColors.textSecondary),
                        ),
                      ],
                    ),
                  )
                : ListView.separated(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
                    itemCount: filtered.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 10),
                    itemBuilder: (context, i) => PatientCard(
                      patient: filtered[i],
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) =>
                              PatientDetailScreen(patient: filtered[i]),
                        ),
                      ),
                    ),
                  ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => const AddPatientScreen()),
        ),
        backgroundColor: AppColors.primary,
        child: const Icon(Icons.person_add_rounded, color: Colors.white),
      ),
    );
  }
}

class _FilterChip extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onTap;
  final Color? color;

  const _FilterChip({
    required this.label,
    required this.isSelected,
    required this.onTap,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: Center(
        child: ActionChip(
          label: Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
              color: isSelected ? Colors.white : (color ?? AppColors.textSecondary),
            ),
          ),
          backgroundColor: isSelected ? (color ?? AppColors.primary) : Colors.transparent,
          side: BorderSide(
            color: isSelected ? Colors.transparent : AppColors.border,
            width: 1,
          ),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          onPressed: onTap,
          padding: const EdgeInsets.symmetric(horizontal: 4),
        ),
      ),
    );
  }
}
