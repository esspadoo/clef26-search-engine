import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_final_schema():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Styles
    box_style = dict(boxstyle='round,pad=0.5', facecolor='#fdfdfd', edgecolor='black', linewidth=2)
    arrow_style = dict(arrowstyle='-|>', color='black', lw=2, mutation_scale=20)
    
    # Common Y-axis for the main flow arrow
    arrow_y = 4.5

    # 1. Main Stages Blocks
    stages = [
        (3.5, arrow_y, "Query\nExpansion"),
        (8.0, arrow_y, "Candidate\nRetrieval"),
        (12.5, arrow_y, "Document\nRe-ranking")
    ]

    for x, y, label in stages:
        ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold', bbox=box_style, zorder=5)

    # --- STAGE 0: INPUT ---
    # Label Above
    ax.text(1.15, arrow_y + 0.8, "User Query", ha='center', fontsize=12, fontweight='bold')
    # Arrow
    ax.annotate('', xy=(2.3, arrow_y), xytext=(0.2, arrow_y), arrowprops=arrow_style)
    # Icon Below
    bubble = patches.FancyBboxPatch((0.4, arrow_y - 1.8), 1.5, 1.0, boxstyle="round,pad=0.1,rounding_size=0.2",
                                   linewidth=1.5, edgecolor='black', facecolor='#e3f2fd')
    ax.add_patch(bubble)
    ax.text(1.15, arrow_y - 1.3, '"Vitamin D and\nCOVID-19 risk"', ha='center', va='center', fontsize=10, fontweight='bold')

    # --- STAGE 1: EXPANSION ---
    # Label Above
    ax.text(5.75, arrow_y + 0.8, "Expanded query", ha='center', fontsize=12, fontweight='bold')
    # Arrow
    ax.annotate('', xy=(6.8, arrow_y), xytext=(4.7, arrow_y), arrowprops=arrow_style)
    # Icons Below
    ax.text(5.75, arrow_y - 0.4, '"Vitamin D and COVID-19 risk"', ha='center', fontsize=8, style='italic')
    # Sparse Card
    c1 = patches.Rectangle((4.8, arrow_y - 1.2), 1.9, 0.6, linewidth=1, edgecolor='black', facecolor='#fff9c4')
    ax.add_patch(c1)
    ax.text(4.9, arrow_y - 0.9, "Sparse: Cholecalciferol\nseverity deficiency", fontsize=7)
    # Dense Card
    c2 = patches.Rectangle((4.8, arrow_y - 2.0), 1.9, 0.6, linewidth=1, edgecolor='black', facecolor='#c8e6c9')
    ax.add_patch(c2)
    ax.text(4.9, arrow_y - 1.7, "Dense: Immune response\nrespiratory viral", fontsize=7)
    # ColBERT Card
    c3 = patches.Rectangle((4.8, arrow_y - 2.8), 1.9, 0.6, linewidth=1, edgecolor='black', facecolor='#bbdefb')
    ax.add_patch(c3)
    ax.text(4.9, arrow_y - 2.5, "ColBERT: Serum levels and\nICU admission", fontsize=7)

    # --- STAGE 2: CANDIDATES ---
    # Label Above
    ax.text(10.25, arrow_y + 0.8, "Top-K Candidates", ha='center', fontsize=12, fontweight='bold')
    # Arrow
    ax.annotate('', xy=(11.3, arrow_y), xytext=(9.2, arrow_y), arrowprops=arrow_style)
    # Icon Below
    paper1 = patches.Rectangle((9.65, arrow_y - 2.2), 1.2, 1.6, linewidth=1, edgecolor='gray', facecolor='white')
    ax.add_patch(paper1)
    for i in range(1, 4):
        ax.text(9.85, arrow_y - 0.5 - i*0.4, f"{i}.", fontsize=9, fontweight='bold')
        ax.plot([10.15, 10.65], [arrow_y - 0.55 - i*0.4, arrow_y - 0.55 - i*0.4], color='black', lw=1)

    # --- STAGE 3: FINAL ---
    # Label Above
    ax.text(14.75, arrow_y + 0.8, "Final Ranking", ha='center', fontsize=12, fontweight='bold', color='#2e7d32')
    # Arrow
    ax.annotate('', xy=(15.8, arrow_y), xytext=(13.7, arrow_y), arrowprops=arrow_style)
    # Icon Below
    paper2 = patches.Rectangle((14.15, arrow_y - 2.2), 1.2, 1.6, linewidth=2, edgecolor='#2e7d32', facecolor='white')
    ax.add_patch(paper2)
    for i in range(1, 4):
        ax.text(14.35, arrow_y - 0.5 - i*0.4, f"{i}.", fontsize=9, fontweight='bold', color='#1b5e20')
        ax.plot([14.65, 15.15], [arrow_y - 0.55 - i*0.4, arrow_y - 0.55 - i*0.4], color='black', lw=1.5)

    plt.title("Scientific Source Retrieval Pipeline Architecture", fontsize=18, fontweight='bold', pad=40)
    plt.tight_layout()
    plt.savefig('pipeline_final.png', dpi=300, bbox_inches='tight')
    plt.show()

create_final_schema()