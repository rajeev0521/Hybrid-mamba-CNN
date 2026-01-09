import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

def create_figure(figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    return fig, ax

def draw_box(ax, x, y, w, h, text, color='lightblue', fontsize=10):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", 
                                  linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)
    return rect

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=15, color='black')
    ax.add_patch(arrow)
    if text:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 2, text, ha='center', va='bottom', fontsize=8)

def draw_hld():
    fig, ax = create_figure(figsize=(14, 8))
    plt.title("Hybrid Mamba-CNN: High-Level Architecture", fontsize=16, pad=20)

    # Input
    draw_box(ax, 5, 45, 10, 10, "Input Image\n(512x512)", color='#FFD1DC')

    # Encoders Container
    rect = patches.Rectangle((20, 10), 30, 80, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(35, 92, "Parallel Encoders", ha='center', fontsize=12, fontweight='bold')

    # CNN Branch
    draw_box(ax, 25, 65, 20, 15, "CNN Encoder\n(ResNet50)", color='#ADD8E6')
    
    # Mamba Branch
    draw_box(ax, 25, 20, 20, 15, "Vision Mamba\nEncoder", color='#90EE90')

    # Fusion
    draw_box(ax, 60, 40, 15, 20, "Feature Fusion\nModule", color='#FFB6C1')

    # Classifier
    draw_box(ax, 85, 45, 10, 10, "Classifier\nHead", color='#E0E0E0')

    # Arrows
    # Input to CNN
    draw_arrow(ax, 15, 50, 25, 72.5)
    # Input to Mamba
    draw_arrow(ax, 15, 50, 25, 27.5)
    
    # CNN to Fusion
    draw_arrow(ax, 45, 72.5, 60, 55, "Local Feat.")
    # Mamba to Fusion
    draw_arrow(ax, 45, 27.5, 60, 45, "Global Ctx.")
    
    # Fusion to Classifier
    draw_arrow(ax, 75, 50, 85, 50, "Fused")
    
    # Output
    draw_arrow(ax, 95, 50, 100, 50, "Logits")

    plt.tight_layout()
    plt.savefig('hld_architecture.png', dpi=300, bbox_inches='tight')
    print("Generated hld_architecture.png")
    plt.close()

def draw_mamba_lld():
    fig, ax = create_figure(figsize=(12, 10))
    plt.title("Low-Level Design: Vision Mamba Encoder", fontsize=16, pad=20)

    # Input
    draw_box(ax, 5, 85, 15, 8, "Input Image", color='#FFD1DC')
    
    # Patch Embedding
    draw_box(ax, 5, 65, 15, 10, "Patch Embedding\n(Conv2d)", color='#FFE4B5')
    draw_arrow(ax, 12.5, 85, 12.5, 75)
    
    # Mamba Block Container
    rect = patches.Rectangle((30, 10), 40, 80, linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rect)
    ax.text(50, 92, "Mamba Block (Repeated L times)", ha='center', fontsize=12, fontweight='bold')
    
    # Inside Mamba Block
    draw_box(ax, 35, 75, 30, 8, "Layer Norm", color='#E0E0E0')
    
    # Split paths
    draw_arrow(ax, 50, 75, 40, 65)
    draw_arrow(ax, 50, 75, 60, 65)
    
    # Main path (SSM)
    draw_box(ax, 32, 55, 16, 8, "Linear Proj (x2)", color='#90EE90')
    draw_box(ax, 32, 40, 16, 8, "1D Conv + SiLU", color='#90EE90')
    draw_box(ax, 32, 25, 16, 8, "SSM (Discretize)", color='#90EE90')
    
    draw_arrow(ax, 40, 55, 40, 48)
    draw_arrow(ax, 40, 40, 40, 33)
    
    # Gating path
    draw_box(ax, 52, 55, 16, 8, "Linear Proj (Gate)", color='#90EE90')
    draw_box(ax, 52, 40, 16, 8, "SiLU Activation", color='#90EE90')
    
    draw_arrow(ax, 60, 55, 60, 48)
    
    # Element-wise Mult
    draw_box(ax, 42, 15, 16, 6, "Element-wise Mult", color='#FFB6C1')
    draw_arrow(ax, 40, 25, 45, 21) # From SSM
    draw_arrow(ax, 60, 40, 55, 21) # From Gate
    
    # Output Proj
    draw_box(ax, 80, 45, 15, 10, "Output Linear", color='#E0E0E0')
    draw_arrow(ax, 58, 18, 80, 48)
    
    # Residual
    draw_arrow(ax, 20, 70, 90, 70, "Residual Connection")
    # Make residual curve around? No, straight line is fine but needs to avoid boxes.
    # Let's just imply it.
    
    plt.tight_layout()
    plt.savefig('lld_mamba_encoder.png', dpi=300, bbox_inches='tight')
    print("Generated lld_mamba_encoder.png")
    plt.close()

def draw_fusion_lld():
    fig, ax = create_figure(figsize=(10, 8))
    plt.title("Low-Level Design: Feature Fusion Module", fontsize=16, pad=20)

    # Inputs
    draw_box(ax, 10, 70, 20, 10, "CNN Features\n(2048-dim)", color='#ADD8E6')
    draw_box(ax, 10, 20, 20, 10, "Mamba Features\n(768-dim)", color='#90EE90')

    # Projections
    draw_box(ax, 40, 70, 20, 10, "Linear Proj\n(to 512-dim)", color='#E0E0E0')
    draw_box(ax, 40, 20, 20, 10, "Linear Proj\n(to 512-dim)", color='#E0E0E0')
    
    draw_arrow(ax, 30, 75, 40, 75)
    draw_arrow(ax, 30, 25, 40, 25)

    # Attention/Gate
    draw_box(ax, 70, 45, 20, 15, "Multi-Head\nAttention / Gate", color='#FFB6C1')
    
    draw_arrow(ax, 60, 75, 70, 55)
    draw_arrow(ax, 60, 25, 70, 50)

    # Output
    draw_box(ax, 70, 10, 20, 10, "Fused Output\n(1024-dim)", color='#FFD700')
    draw_arrow(ax, 80, 45, 80, 20)

    plt.tight_layout()
    plt.savefig('lld_fusion_module.png', dpi=300, bbox_inches='tight')
    print("Generated lld_fusion_module.png")
    plt.close()

if __name__ == "__main__":
    try:
        draw_hld()
        draw_mamba_lld()
        draw_fusion_lld()
        print("All diagrams generated successfully.")
    except Exception as e:
        print(f"Error generating diagrams: {e}")
        import traceback
        traceback.print_exc()
