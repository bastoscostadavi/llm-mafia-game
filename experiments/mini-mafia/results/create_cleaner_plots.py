#!/usr/bin/env python3
"""
Create cleaner, less cluttered benchmark plots

Suggestions for reducing clutter:
1. Remove text labels (bar length shows value anyway)
2. Keep company logos but make them smaller
3. Reduce font sizes slightly
4. Simplify error bars (no caps)
5. Remove hatched overlay bars or make them subtle
"""

def create_clean_benchmark_plot(benchmark_data, title, filename, background_key="", use_good_wins=False):
    """Create a cleaner horizontal bar plot with minimal clutter"""
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import json
    
    # Export data (same as original)
    export_data = {
        'models': benchmark_data['models'],
        'win_rates': benchmark_data['values'],
        'tie_win_rates': benchmark_data.get('tie_values', [0] * len(benchmark_data['values'])),
        'errors': benchmark_data['errors'],
        'companies': benchmark_data['companies'],
        'background': background_key,
        'use_good_wins': use_good_wins
    }
    
    data_filename = filename.replace('.png', '_data.json')
    with open(data_filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    plt.ioff()
    
    # SUGGESTION 1: Smaller, more reasonable font sizes
    plt.rcParams.update({
        'font.size': 12,           # Reduced from 24
        'axes.labelsize': 14,      # Reduced from 24
        'axes.titlesize': 16,      # Reduced from 24
        'xtick.labelsize': 12,     # Reduced from 24
        'ytick.labelsize': 10,     # Reduced from 24
        'legend.fontsize': 12,     # Reduced from 24
        'figure.titlesize': 16     # Reduced from 24
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted size
    
    models = benchmark_data['models']
    values = benchmark_data['values'] 
    tie_values = benchmark_data.get('tie_values', [0] * len(values))
    errors = benchmark_data['errors']
    companies = benchmark_data['companies']
    
    y_positions = range(len(models))
    bar_color = get_background_color(background_key)
    
    # SUGGESTION 2: Simplified error bars (no caps, thinner)
    bars = ax.barh(y_positions, values, xerr=errors, 
                   color=bar_color, alpha=0.8, height=0.7,
                   error_kw={'capsize': 0, 'capthick': 1, 'elinewidth': 1})
    
    # SUGGESTION 3: More subtle tie bars (lighter, no hatching)
    if any(tie_val > 0 for tie_val in tie_values):
        tie_bars = ax.barh(y_positions, tie_values, 
                          color=bar_color, alpha=0.4, height=0.7,
                          edgecolor='none')
    
    # SUGGESTION 4: Model names on y-axis instead of text labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{model}" for model in models])
    
    # SUGGESTION 5: Optional - smaller company logos on far left
    for i, company in enumerate(companies):
        logo_img = load_company_logo(company, size=(25, 25))  # Smaller
        if logo_img is not None:
            try:
                imagebox = OffsetImage(logo_img, zoom=0.6)  # Smaller zoom
                ab = AnnotationBbox(imagebox, (-8, i), frameon=False, 
                                  xycoords='data', boxcoords="data")
                ax.add_artist(ab)
            except Exception as e:
                # Smaller fallback circles
                ax.text(-8, i, company[0], ha='center', va='center', 
                        fontweight='bold', fontsize=10, color='black',
                        bbox=dict(boxstyle="circle,pad=0.2", facecolor='lightgray'))
    
    # SUGGESTION 6: Show values only on hover or in a separate table
    # (Remove the cluttering text labels entirely)
    
    # Set labels
    xlabel = 'Town Win Rate (%)' if use_good_wins else 'Mafia Win Rate (%)'
    ax.set_xlabel(xlabel, fontweight='bold')
    
    # Cleaner axis formatting
    ax.set_xlim(-12, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    
    # Subtle grid
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Clean spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Clean plot saved as {filename}")

# ALTERNATIVE SUGGESTIONS:

def create_minimal_plot_option_a():
    """Option A: Keep logos, remove all text labels"""
    # - Company logos (smaller)
    # - No text labels on bars
    # - Model names on y-axis
    # - Cleaner error bars
    pass

def create_minimal_plot_option_b():
    """Option B: Remove logos, keep essential text"""
    # - No company logos
    # - Model names on y-axis  
    # - Optional: Show only top values as text
    # - More space for model names
    pass

def create_minimal_plot_option_c():
    """Option C: Color-coded by company instead of logos"""
    # - Different colors for each company
    # - Legend showing company colors
    # - No logos, no text labels
    # - Very clean appearance
    pass

if __name__ == "__main__":
    print("Cleaner plot suggestions:")
    print("1. Remove text labels - bar length shows the value")
    print("2. Smaller company logos (or remove entirely)")  
    print("3. Reduce font sizes (24pt is very large)")
    print("4. Simplify error bars (remove caps)")
    print("5. Make tie bars more subtle")
    print("6. Use model names on y-axis instead of floating labels")