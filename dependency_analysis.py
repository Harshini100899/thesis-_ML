import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt

def safe_parse_list(value):
    """Safely parse string representation of list."""
    if pd.isna(value) or value == '':
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

def main():
    # Load the data
    df = pd.read_csv('joss_all_with_dependency_labels1.csv')
    
    print("=" * 80)
    print("DEPENDENCY ANNOTATION ANALYSIS REPORT")
    print("=" * 80)
    
    # Basic dataset info
    total_projects = len(df)
    projects_with_deps = df[df['dependecies_found'].notna() & (df['dependecies_found'] != '[]')].shape[0]
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"   Total projects analyzed: {total_projects}")
    print(f"   Projects with dependencies found: {projects_with_deps}")
    
    # Parse all dependencies
    all_dependencies = []
    dependency_counts_per_project = []
    
    for idx, row in df.iterrows():
        deps = safe_parse_list(row['dependecies_found'])
        all_dependencies.extend(deps)
        dependency_counts_per_project.append(len(deps))
    
    # Count unique and total dependencies
    total_dependency_occurrences = len(all_dependencies)
    unique_dependencies = set(all_dependencies)
    num_unique_dependencies = len(unique_dependencies)
    
    print(f"\n📦 DEPENDENCY STATISTICS")
    print(f"   Total dependency occurrences across all projects: {total_dependency_occurrences:,}")
    print(f"   Unique dependencies found: {num_unique_dependencies:,}")
    print(f"   Average dependencies per project: {total_dependency_occurrences/projects_with_deps:.1f}")
    print(f"   Max dependencies in a single project: {max(dependency_counts_per_project)}")
    print(f"   Min dependencies in a single project: {min([d for d in dependency_counts_per_project if d > 0])}")
    
    # Frequency analysis
    dep_counter = Counter(all_dependencies)
    most_common = dep_counter.most_common(50)
    
    print(f"\n🔝 TOP 30 MOST FREQUENTLY USED DEPENDENCIES")
    print("-" * 50)
    for i, (dep, count) in enumerate(most_common[:30], 1):
        usage_percentage = (count / projects_with_deps) * 100
        print(f"   {i:2}. {dep:<40} {count:4} projects ({usage_percentage:5.1f}%)")
    
    # Annotation effort analysis
    print(f"\n⏱️  ANNOTATION EFFORT ANALYSIS")
    print("-" * 50)
    print(f"   If manually annotating each unique dependency:")
    print(f"   → {num_unique_dependencies:,} unique dependencies would need individual annotation")
    print(f"   ")
    print(f"   Time estimates (assuming 30 seconds per dependency):")
    print(f"   → Total time: {num_unique_dependencies * 0.5:.1f} minutes ({num_unique_dependencies * 0.5 / 60:.1f} hours)")
    print(f"   ")
    print(f"   Optimized approach (annotating most common first):")
    
    # Calculate coverage by annotating top N dependencies
    cumulative_coverage = 0
    thresholds = [10, 25, 50, 100, 200, 500]
    
    for threshold in thresholds:
        top_n = most_common[:threshold]
        coverage = sum(count for _, count in top_n)
        coverage_pct = (coverage / total_dependency_occurrences) * 100
        print(f"   → Annotating top {threshold:3} dependencies covers {coverage_pct:5.1f}% of all occurrences")
    
    # Parse and analyze dependency labels
    print(f"\n🏷️  DEPENDENCY LABEL CATEGORIES")
    print("-" * 50)
    
    all_labels = []
    for idx, row in df.iterrows():
        labels = safe_parse_list(row['dependency_labels'])
        all_labels.extend(labels)
    
    label_counter = Counter(all_labels)
    print(f"   Unique label categories: {len(label_counter)}")
    print(f"\n   Label distribution:")
    for label, count in label_counter.most_common():
        print(f"   → {label:<30} {count:4} projects")
    
    # Summary statement for paper
    print(f"\n" + "=" * 80)
    print("📝 SUMMARY STATEMENT FOR PAPER")
    print("=" * 80)
    
    top_50_coverage = sum(count for _, count in most_common[:50])
    top_50_coverage_pct = (top_50_coverage / total_dependency_occurrences) * 100
    
    top_100_coverage = sum(count for _, count in most_common[:100])
    top_100_coverage_pct = (top_100_coverage / total_dependency_occurrences) * 100
    
    summary = f"""
Manual annotation of dependencies was tedious and time-consuming. Our analysis 
of {total_projects} JOSS projects revealed {num_unique_dependencies:,} unique dependencies 
across {total_dependency_occurrences:,} total dependency occurrences. 

To reduce annotation effort, we prioritized the most frequently used dependencies. 
The top 50 most common dependencies (representing only {50/num_unique_dependencies*100:.1f}% of unique 
dependencies) covered {top_50_coverage_pct:.1f}% of all dependency occurrences. Similarly, 
annotating the top 100 dependencies ({100/num_unique_dependencies*100:.1f}% of unique) covered 
{top_100_coverage_pct:.1f}% of all occurrences, significantly reducing the manual effort required.

Key findings:
- Average of {total_dependency_occurrences/projects_with_deps:.0f} dependencies per project
- Most used dependency: '{most_common[0][0]}' (used in {most_common[0][1]} projects, {most_common[0][1]/projects_with_deps*100:.1f}%)
- Long-tail distribution: {len([d for d, c in dep_counter.items() if c == 1]):,} dependencies appear in only one project
    """
    print(summary)
    
    # Create visualization
    create_visualizations(dep_counter, most_common, num_unique_dependencies, 
                         total_dependency_occurrences, projects_with_deps)
    
    # Export detailed data
    export_detailed_report(dep_counter, df)

def create_visualizations(dep_counter, most_common, num_unique, total_occurrences, num_projects):
    """Create visualizations for the analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Top 20 dependencies bar chart
    ax1 = axes[0, 0]
    top_20 = most_common[:20]
    deps = [d[0] for d in top_20]
    counts = [d[1] for d in top_20]
    ax1.barh(deps[::-1], counts[::-1], color='steelblue')
    ax1.set_xlabel('Number of Projects')
    ax1.set_title('Top 20 Most Used Dependencies')
    
    # 2. Cumulative coverage chart
    ax2 = axes[0, 1]
    cumulative = []
    running_total = 0
    for i, (_, count) in enumerate(most_common[:200], 1):
        running_total += count
        cumulative.append((running_total / total_occurrences) * 100)
    ax2.plot(range(1, len(cumulative) + 1), cumulative, 'b-', linewidth=2)
    ax2.axhline(y=80, color='r', linestyle='--', label='80% coverage')
    ax2.axhline(y=90, color='g', linestyle='--', label='90% coverage')
    ax2.set_xlabel('Number of Top Dependencies Annotated')
    ax2.set_ylabel('Coverage (%)')
    ax2.set_title('Annotation Effort: Coverage by Top Dependencies')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Frequency distribution
    ax3 = axes[1, 0]
    freq_values = list(dep_counter.values())
    ax3.hist(freq_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Number of Projects Using Dependency')
    ax3.set_ylabel('Number of Dependencies')
    ax3.set_title('Distribution of Dependency Usage Frequency')
    ax3.set_yscale('log')
    
    # 4. Summary statistics pie chart
    ax4 = axes[1, 1]
    single_use = len([d for d, c in dep_counter.items() if c == 1])
    low_use = len([d for d, c in dep_counter.items() if 2 <= c <= 5])
    medium_use = len([d for d, c in dep_counter.items() if 6 <= c <= 20])
    high_use = len([d for d, c in dep_counter.items() if c > 20])
    
    sizes = [single_use, low_use, medium_use, high_use]
    labels = [f'Single use\n({single_use})', f'2-5 projects\n({low_use})', 
              f'6-20 projects\n({medium_use})', f'>20 projects\n({high_use})']
    colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Dependency Usage Distribution')
    
    plt.tight_layout()
    plt.savefig('dependency_analysis_report.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Visualization saved to: dependency_analysis_report.png")

def export_detailed_report(dep_counter, df):
    """Export detailed dependency data to CSV."""
    # Create dependency frequency dataframe
    dep_df = pd.DataFrame([
        {'dependency': dep, 'usage_count': count, 'usage_percentage': (count/len(df))*100}
        for dep, count in dep_counter.most_common()
    ])
    dep_df.to_csv('dependency_frequency_report.csv', index=False)
    print(f"📄 Detailed report saved to: dependency_frequency_report.csv")

if __name__ == "__main__":
    main()
