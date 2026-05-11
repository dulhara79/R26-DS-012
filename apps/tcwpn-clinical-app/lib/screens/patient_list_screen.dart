import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme/app_theme.dart';
import '../services/patient_provider.dart';
import '../widgets/patient_card.dart';
import 'patient_detail_screen.dart';

class PatientListScreen extends StatefulWidget {
  const PatientListScreen({super.key});

  @override
  State<PatientListScreen> createState() => _PatientListScreenState();
}

class _PatientListScreenState extends State<PatientListScreen> {
  final _searchCtrl = TextEditingController();
  String _query = '';

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final provider  = context.watch<PatientProvider>();
    final filtered  = provider.searchPatients(_query);

    return Scaffold(
      backgroundColor: AppColors.surfaceSecond,
      appBar: AppBar(
        title: const Text('Patients'),
        actions: [
          IconButton(
            icon: const Icon(Icons.filter_list_rounded),
            onPressed: () {},
          ),
          const SizedBox(width: 4),
        ],
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
            height: 40,
            color: AppColors.surface,
            child: ListView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 16),
              children: [
                _FilterChip(label: 'All', isSelected: true, onTap: () {}),
                _FilterChip(label: 'High Risk', isSelected: false, onTap: () {}, color: AppColors.riskHigh),
                _FilterChip(label: 'Moderate', isSelected: false, onTap: () {}, color: AppColors.riskModerate),
                _FilterChip(label: 'Low Risk', isSelected: false, onTap: () {}, color: AppColors.riskLow),
                _FilterChip(label: 'Ward 04', isSelected: false, onTap: () {}),
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
    );
  }
}
