import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_architecture_diagram():
    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define styles
    box_style = dict(boxstyle='round,pad=0.5', facecolor='#e6f2ff', edgecolor='#004c99', linewidth=2)
    layer_style = dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#cccccc', linewidth=1, alpha=0.5)
    arrow_props = dict(arrowstyle='->', lw=2, color='#555555')
    text_props = dict(ha='center', va='center', fontsize=10, fontfamily='sans-serif')
    title_props = dict(ha='center', va='center', fontsize=12, fontweight='bold', color='#333333')

    # --- LAYERS (Backgrounds) ---
    # Layer 1: Data Ingestion
    ax.add_patch(patches.FancyBboxPatch((0.5, 7.5), 13, 2.0, **layer_style))
    ax.text(7, 9.3, "1. Data Ingestion & Splitting Layer", **title_props)

    # Layer 2: Profiling
    ax.add_patch(patches.FancyBboxPatch((0.5, 5.0), 13, 2.0, **layer_style))
    ax.text(7, 6.8, "2. Profiling Layer", **title_props)

    # Layer 3: Model Layer
    ax.add_patch(patches.FancyBboxPatch((0.5, 2.5), 13, 2.0, **layer_style))
    ax.text(7, 4.3, "3. Model Layer", **title_props)

    # Layer 4: Serving Layer
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.5), 13, 1.5, **layer_style))
    ax.text(7, 1.8, "4. Serving Layer", **title_props)

    # --- NODES ---
    
    # Layer 1
    ax.text(2, 8.4, "PAN 2015\nDataset\n(XML)", bbox=dict(boxstyle='round,pad=0.5', fc='#fff2cc', ec='#d6b656'), **text_props)
    ax.text(5, 8.4, "Preprocessing\n(Cleaning, Concat)", bbox=box_style, **text_props)
    ax.text(8, 8.4, "User-based\nSplit\n(70/10/20)", bbox=box_style, **text_props)
    
    # Layer 2
    ax.text(3, 5.9, "User Profiling\n(Big Five Traits)", bbox=dict(boxstyle='round,pad=0.5', fc='#d9ead3', ec='#6aa84f'), **text_props)
    ax.text(11, 5.9, "Item Profiling\n(Hashtag Personality)", bbox=dict(boxstyle='round,pad=0.5', fc='#d9ead3', ec='#6aa84f'), **text_props)
    ax.text(7, 5.9, "Evidence\nRetrieval\n(BM25)", bbox=dict(boxstyle='round,pad=0.5', fc='#fce5cd', ec='#e69138'), **text_props)

    # Layer 3
    ax.text(3, 3.4, "Content-based\nFilter", bbox=box_style, **text_props)
    ax.text(5.5, 3.4, "Co-occurrence\nRules", bbox=box_style, **text_props)
    ax.text(8.5, 3.4, "Personality-Enhanced\nLightGCN", bbox=dict(boxstyle='round,pad=0.5', fc='#d0e0e3', ec='#45818e', lw=2), **text_props)
    ax.text(11, 3.4, "Hybrid Re-ranking\nScore(u, h)", bbox=dict(boxstyle='round,pad=0.5', fc='#e6b8af', ec='#a61c00', lw=2), **text_props)

    # Layer 4
    ax.text(7, 1.1, "Top-K Recommendation List\n(with Explanations)", bbox=dict(boxstyle='round,pad=0.5', fc='#d9d2e9', ec='#674ea7'), **text_props)

    # --- EDGES ---
    
    # L1
    ax.annotate("", xy=(3.5, 8.4), xytext=(2.8, 8.4), arrowprops=arrow_props)
    ax.annotate("", xy=(6.8, 8.4), xytext=(6.2, 8.4), arrowprops=arrow_props)
    
    # L1 -> L2
    ax.annotate("", xy=(3, 6.4), xytext=(8, 7.9), arrowprops=dict(arrowstyle='->', lw=2, color='#555555', connectionstyle="arc3,rad=-0.1")) # To User Prof
    ax.annotate("", xy=(11, 6.4), xytext=(8, 7.9), arrowprops=dict(arrowstyle='->', lw=2, color='#555555', connectionstyle="arc3,rad=0.1")) # To Item Prof
    ax.annotate("", xy=(7, 6.4), xytext=(8, 7.9), arrowprops=dict(arrowstyle='->', lw=2, color='#555555', connectionstyle="arc3,rad=0.0")) # To IR
    
    # L2 -> L3
    ax.annotate("", xy=(3, 3.9), xytext=(3, 5.4), arrowprops=arrow_props) # User -> Content
    ax.annotate("", xy=(8.5, 3.9), xytext=(3, 5.4), arrowprops=dict(arrowstyle='->', lw=2, color='#555555', connectionstyle="arc3,rad=0.1")) # User -> GCN
    ax.annotate("", xy=(11, 3.9), xytext=(11, 5.4), arrowprops=arrow_props) # Item -> Hybrid (Actually Item Prof is used in GCN too conceptually, but simplifying)
    ax.annotate("", xy=(8.5, 3.9), xytext=(11, 5.4), arrowprops=dict(arrowstyle='->', lw=2, color='#555555', connectionstyle="arc3,rad=-0.1")) # Item -> GCN
    
    # L3 Internal
    ax.annotate("", xy=(11, 3.9), xytext=(4.2, 3.4), arrowprops=dict(arrowstyle='->', lw=2, color='#555555', connectionstyle="arc3,rad=-0.2")) # Content -> Hybrid
    ax.annotate("", xy=(11, 3.9), xytext=(6.7, 3.4), arrowprops=dict(arrowstyle='->', lw=2, color='#555555', connectionstyle="arc3,rad=-0.1")) # Rules -> Hybrid
    ax.annotate("", xy=(11, 3.9), xytext=(9.7, 3.4), arrowprops=arrow_props) # GCN -> Hybrid

    # L3 -> L4
    ax.annotate("", xy=(7, 1.6), xytext=(11, 2.9), arrowprops=dict(arrowstyle='->', lw=2, color='#555555', connectionstyle="arc3,rad=0.1"))
    
    # Extra: Explanations
    ax.annotate("", xy=(8.3, 1.1), xytext=(7, 5.4), arrowprops=dict(arrowstyle='->', lw=1, color='#e69138', linestyle='dashed', connectionstyle="arc3,rad=-0.4")) # Evidence -> Output
    ax.text(8.5, 2.2, "Evidence\nFlow", fontsize=8, color='#e69138', ha='center')

    # Save
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/system_architecture.png', dpi=300, bbox_inches='tight')
    print("Architecture diagram saved to figs/system_architecture.png")

if __name__ == "__main__":
    create_architecture_diagram()
