#!/usr/bin/env python3
"""
Create horizontal bar plots for LLM Mafia benchmark results
with company logos, model names in bars, and values on the right.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import requests
from PIL import Image
import io
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Data for the three benchmarks
mistral_benchmark = {
    "models": ["GPT-4o", "Mistral Instruct 7B", "Llama 3.1 8B", "Qwen2.5 7B"],
    "values": [50.6, 53.7, 43.1, 41.6],
    "errors": [2.3, 1.6, 1.6, 1.6],
    "companies": ["OpenAI", "Mistral AI", "Meta", "Alibaba"],
    "colors": ["#10A37F", "#FF6B35", "#1877F2", "#FF6A00"]
}

qwen_benchmark = {
    "models": ["Llama 3.1 8B", "Qwen2.5 7B", "Mistral Instruct 7B"],
    "values": [48.9, 44.9, 43.8],
    "errors": [1.6, 1.6, 1.6],
    "companies": ["Meta", "Alibaba", "Mistral AI"],
    "colors": ["#1877F2", "#FF6A00", "#FF6B35"]
}

llama_benchmark = {
    "models": ["Mistral Instruct 7B", "Llama 3.1 8B"],
    "values": [67.9, 64.3],
    "errors": [1.5, 1.5],
    "companies": ["Mistral AI", "Meta"],
    "colors": ["#FF6B35", "#1877F2"]
}

def load_company_logo(company, size=(40, 40)):
    """Load actual company logos from the logos folder"""
    # Map company names to logo filenames
    logo_files = {
        "OpenAI": "openai.png",
        "Mistral AI": "mistral-color.png", 
        "Meta": "meta-logo-6760788.png",
        "Alibaba": "BABA.png"
    }
    
    try:
        logo_path = f"logos/{logo_files[company]}"
        if os.path.exists(logo_path):
            # Load and resize the actual logo
            img = Image.open(logo_path)
            
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Resize maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Create new image with transparent background
            new_img = Image.new('RGBA', size, (255, 255, 255, 0))
            
            # Center the logo
            x = (size[0] - img.width) // 2
            y = (size[1] - img.height) // 2
            new_img.paste(img, (x, y), img if img.mode == 'RGBA' else None)
            
            return np.array(new_img)
        else:
            print(f"Logo file not found: {logo_path}")
            return None
            
    except Exception as e:
        print(f"Error loading logo for {company}: {e}")
        return None

def create_benchmark_plot(data, title, filename):
    """Create a horizontal bar plot with logos and formatting"""
    # Use non-interactive backend
    plt.ioff()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Sort data by values (descending)
    sorted_indices = np.argsort(data["values"])[::-1]
    models = [data["models"][i] for i in sorted_indices]
    values = [data["values"][i] for i in sorted_indices]
    errors = [data["errors"][i] for i in sorted_indices]
    companies = [data["companies"][i] for i in sorted_indices]
    colors = [data["colors"][i] for i in sorted_indices]
    
    y_positions = range(len(models))
    
    # Create horizontal bars (all same color)
    bar_color = '#4A90E2'  # Nice blue color for all bars
    bars = ax.barh(y_positions, values, xerr=errors, 
                   color=bar_color, alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2})
    
    # Add model names inside the bars (white text)
    for i, (model, value) in enumerate(zip(models, values)):
        ax.text(value/2, i, model, ha='center', va='center', 
                fontweight='bold', fontsize=11, color='white')
    
    # Add values on the right side of bars
    for i, (value, error) in enumerate(zip(values, errors)):
        ax.text(value + error + 1.5, i, f'{value}% ± {error}%', 
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Add company logos on the left
    for i, company in enumerate(companies):
        logo_img = load_company_logo(company, size=(40, 40))
        if logo_img is not None:
            try:
                imagebox = OffsetImage(logo_img, zoom=0.8)
                ab = AnnotationBbox(imagebox, (-6, i), frameon=False, 
                                  xycoords='data', boxcoords="data")
                ax.add_artist(ab)
            except Exception as e:
                print(f"Failed to add logo for {company}: {e}")
                # Fallback to company initial
                ax.text(-6, i, company[0], ha='center', va='center', 
                        fontweight='bold', fontsize=12, color='black',
                        bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgray'))
        else:
            # Fallback to company initial
            ax.text(-6, i, company[0], ha='center', va='center', 
                    fontweight='bold', fontsize=12, color='black',
                    bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgray'))
    
    # Add no-information exchange protocol reference line (5/12 ≈ 41.67%)
    baseline = 5/12 * 100  # Convert to percentage
    ax.axvline(x=baseline, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add legend with dotted line (Mathematica style) - moved closer along diagonal
    legend_x = 65  # Move closer to the diagonal
    legend_y = len(models) - 0.5  # Move slightly down along diagonal
    
    # Add the dotted red line in front of the text (like Mathematica legends)
    ax.plot([legend_x - 8, legend_x - 2], [legend_y, legend_y], 
            color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add the text with background box
    ax.text(legend_x, legend_y, 
            f'No-information exchange benchline ({baseline:.1f}%)', 
            ha='left', va='center', color='red', fontweight='bold', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='red'))
    
    # Formatting
    ax.set_xlabel('Evil Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks([])  # Remove y-axis labels
    ax.set_xlim(-10, 100)  # Set range from -10 (for logos) to 100 (full domain)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Plot saved as {filename}")

def main():
    """Create all three benchmark plots"""
    
    # Create plots
    create_benchmark_plot(
        mistral_benchmark,
        "Models as Mafioso vs Mistral Background (Detective + Villager)",
        "mistral_benchmark_final.png"
    )
    
    create_benchmark_plot(
        qwen_benchmark, 
        "Models as Mafioso vs Qwen2.5 Background (Detective + Villager)",
        "qwen_benchmark_final.png"
    )
    
    create_benchmark_plot(
        llama_benchmark,
        "Models as Mafioso vs Llama3.1 Background (Detective + Villager)", 
        "llama_benchmark_final.png"
    )
    
    print("\nAll benchmark plots created successfully!")
    print("Files saved:")
    print("- mistral_benchmark_final.png")
    print("- qwen_benchmark_final.png") 
    print("- llama_benchmark_final.png")

if __name__ == "__main__":
    main()