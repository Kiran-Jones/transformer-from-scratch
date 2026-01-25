import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

import gradio as gr
from tokenizer import Tokenizer
from generator import Generator

# Custom color scheme
CMAP = 'viridis'
ACCENT_COLOR = '#00693E' 
BG_COLOR = '#FAFAF9'


def load_model_and_tokenizer(model_path, vocab_encoder_path, vocab_merges_path):
    """Load the trained model and tokenizer."""
    tokenizer = Tokenizer.from_files(vocab_encoder_path, vocab_merges_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model, tokenizer


def plot_attention_with_summary(attention_matrix, src_tokens, tgt_tokens, title="Cross-Attention"):
    """Create heatmap with summary bar chart showing attention distribution."""
    fig = plt.figure(figsize=(12, 10), facecolor='white')

    # Create grid: heatmap on top, bar chart below
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[20, 1],
                          hspace=0.05, wspace=0.05)

    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_colorbar = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])

    # Heatmap
    im = ax_heatmap.imshow(attention_matrix, cmap=CMAP, aspect='auto', vmin=0, vmax=1)

    # Style heatmap
    ax_heatmap.set_xticks(np.arange(len(src_tokens)))
    ax_heatmap.set_yticks(np.arange(len(tgt_tokens)))
    ax_heatmap.set_xticklabels([])  # Hide x labels on heatmap (shown on bar chart)
    ax_heatmap.set_yticklabels(tgt_tokens, fontsize=10, fontweight='medium')
    ax_heatmap.set_ylabel('Generated Tokens', fontsize=12, fontweight='bold', labelpad=10)
    ax_heatmap.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Subtle grid
    ax_heatmap.set_xticks(np.arange(len(src_tokens) + 1) - 0.5, minor=True)
    ax_heatmap.set_yticks(np.arange(len(tgt_tokens) + 1) - 0.5, minor=True)
    ax_heatmap.grid(which='minor', color='white', linestyle='-', linewidth=1.5, alpha=0.8)
    ax_heatmap.tick_params(which='minor', size=0)

    # Remove spines
    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)

    # Colorbar
    cbar = plt.colorbar(im, cax=ax_colorbar)
    cbar.set_label('Attention', fontsize=10, fontweight='medium')
    cbar.ax.tick_params(labelsize=9)

    # Summary bar chart - total attention received by each source token
    attention_sum = attention_matrix.sum(axis=0)
    attention_normalized = attention_sum / attention_sum.max() if attention_sum.max() > 0 else attention_sum

    bars = ax_bar.bar(np.arange(len(src_tokens)), attention_normalized,
                      color=plt.cm.viridis(attention_normalized), edgecolor='white', linewidth=0.5)

    ax_bar.set_xlim(-0.5, len(src_tokens) - 0.5)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_xticks(np.arange(len(src_tokens)))
    ax_bar.set_xticklabels(src_tokens, rotation=45, ha='right', fontsize=10, fontweight='medium')
    ax_bar.set_ylabel('Total\nAttention', fontsize=10, fontweight='bold', rotation=0, ha='right', va='center')
    ax_bar.set_xlabel('Source Tokens', fontsize=12, fontweight='bold', labelpad=10)

    # Clean up bar chart
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_linewidth(0.5)
    ax_bar.spines['bottom'].set_linewidth(0.5)
    ax_bar.tick_params(axis='y', labelsize=8)
    ax_bar.set_yticks([0, 0.5, 1.0])

    plt.tight_layout()
    return fig


def plot_all_heads_grid(attention_weights, src_tokens, tgt_tokens, layer_idx, num_heads):
    """Create a grid showing all attention heads for a given layer."""
    cols = 4
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), facecolor='white')
    axes = axes.flatten() if num_heads > 1 else [axes]

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        attn = attention_weights[:, layer_idx, head_idx, :]

        im = ax.imshow(attn, cmap=CMAP, aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Head {head_idx + 1}', fontsize=12, fontweight='bold', pad=8)

        if len(src_tokens) <= 10:
            ax.set_xticks(np.arange(len(src_tokens)))
            ax.set_xticklabels(src_tokens, rotation=45, ha='right', fontsize=8)
        else:
            ax.set_xticks([])

        if len(tgt_tokens) <= 10:
            ax.set_yticks(np.arange(len(tgt_tokens)))
            ax.set_yticklabels(tgt_tokens, fontsize=8)
        else:
            ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#E5E5E5')

    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'All Attention Heads — Layer {layer_idx + 1}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_all_layers_grid(attention_weights, src_tokens, tgt_tokens, head_idx, num_layers):
    """Create a grid showing all layers for a given head."""
    cols = 4
    rows = (num_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), facecolor='white')
    axes = axes.flatten() if num_layers > 1 else [axes]

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        if head_idx is None:  # Average all heads
            attn = attention_weights[:, layer_idx, :, :].mean(axis=1)
        else:
            attn = attention_weights[:, layer_idx, head_idx, :]

        im = ax.imshow(attn, cmap=CMAP, aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=12, fontweight='bold', pad=8)

        if len(src_tokens) <= 10:
            ax.set_xticks(np.arange(len(src_tokens)))
            ax.set_xticklabels(src_tokens, rotation=45, ha='right', fontsize=8)
        else:
            ax.set_xticks([])

        if len(tgt_tokens) <= 10:
            ax.set_yticks(np.arange(len(tgt_tokens)))
            ax.set_yticklabels(tgt_tokens, fontsize=8)
        else:
            ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#E5E5E5')

    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')

    head_label = "Averaged" if head_idx is None else f"Head {head_idx + 1}"
    fig.suptitle(f'All Layers — {head_label}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_demo(generator):
    """Create the Gradio demo interface with enhanced visualization."""

    state = {}

    def translate(text, max_len, beam_width):
        if not text.strip():
            return "Please enter some text to translate.", gr.update(), gr.update(), None

        try:
            result = generator.generate_with_attention(
                text, max_len=int(max_len), beam_width=int(beam_width)
            )
            translation, attention_weights, src_tokens, tgt_tokens, num_layers, num_heads = result

            state['attention_weights'] = attention_weights
            state['src_tokens'] = src_tokens
            state['tgt_tokens'] = tgt_tokens[1:]  # Skip <SOS>
            state['num_layers'] = num_layers
            state['num_heads'] = num_heads

            layer_choices = [f"Layer {i+1}" for i in range(num_layers)]
            head_choices = ["Average (all heads)"] + [f"Head {i+1}" for i in range(num_heads)]

            avg_attn = attention_weights.mean(axis=(1, 2))
            fig = plot_attention_with_summary(avg_attn, src_tokens, state['tgt_tokens'], "Cross-Attention (Averaged)")

            return (
                translation,
                gr.update(choices=layer_choices, value="Layer 1"),
                gr.update(choices=head_choices, value="Average (all heads)"),
                fig
            )

        except Exception as e:
            return f"Error: {str(e)}", gr.update(), gr.update(), None

    def update_attention_plot(layer_select, head_select, view_mode):
        if 'attention_weights' not in state:
            return None

        attention_weights = state['attention_weights']
        src_tokens = state['src_tokens']
        tgt_tokens = state['tgt_tokens']
        num_heads = state['num_heads']

        layer_idx = int(layer_select.split()[-1]) - 1 if layer_select else 0

        if view_mode == "All Heads Grid":
            return plot_all_heads_grid(attention_weights, src_tokens, tgt_tokens, layer_idx, num_heads)

        if view_mode == "All Layers Grid":
            num_layers = state['num_layers']
            if head_select == "Average (all heads)":
                return plot_all_layers_grid(attention_weights, src_tokens, tgt_tokens, None, num_layers)
            else:
                head_idx = int(head_select.split()[-1]) - 1
                return plot_all_layers_grid(attention_weights, src_tokens, tgt_tokens, head_idx, num_layers)

        if head_select == "Average (all heads)":
            attn = attention_weights[:, layer_idx, :, :].mean(axis=1)
            title = f"Cross-Attention — Layer {layer_idx + 1} (Averaged)"
        else:
            head_idx = int(head_select.split()[-1]) - 1
            attn = attention_weights[:, layer_idx, head_idx, :]
            title = f"Cross-Attention — Layer {layer_idx + 1}, Head {head_idx + 1}"

        return plot_attention_with_summary(attn, src_tokens, tgt_tokens, title)

    # Custom CSS for Dartmouth Green theme
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: #FAFAF9 !important;
    }
    button.primary {
        background: #00693E !important;
        border: none !important;
    }
    button.primary:hover {
        background: #004D2C !important;
    }
    .prose h1 {
        color: #1C1C1C !important;
        font-weight: 700 !important;
    }
    .prose h3 {
        color: #6B6B6B !important;
    }
    """

    with gr.Blocks(title="Transformer Attention Visualizer", css=custom_css, theme=gr.themes.Base()) as demo:
        gr.Markdown("""
        # Transformer from Scratch
        ### English → Spanish Translation with Attention Visualization
        """)

        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="English",
                    placeholder="Enter text to translate...",
                    lines=2,
                    max_lines=4
                )
                with gr.Row():
                    max_len_slider = gr.Slider(
                        minimum=10, maximum=100, value=50, step=5,
                        label="Max Length", scale=1
                    )
                    beam_width_slider = gr.Slider(
                        minimum=1, maximum=10, value=4, step=1,
                        label="Beam Width", scale=1
                    )
                translate_btn = gr.Button("Translate", variant="primary", size="lg")
                output_text = gr.Textbox(label="Spanish", lines=2, max_lines=4, interactive=False)

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("#### Visualization Controls")
                    view_mode = gr.Radio(
                        choices=["Single View", "All Heads Grid", "All Layers Grid"],
                        value="Single View",
                        label="View Mode",
                        container=True
                    )
                    layer_dropdown = gr.Dropdown(
                        choices=["Layer 1"], value="Layer 1",
                        label="Layer"
                    )

                    head_dropdown = gr.Dropdown(
                        choices=["Average (all heads)"], value="Average (all heads)",
                        label="Head"
                    )

        attention_plot = gr.Plot(label="Attention Visualization", show_label=False)

        # Examples at the bottom, after visualization
        gr.Examples(
            examples=[
                ["Welcome to my translation demo!", 50, 4],
                ["The quick brown fox jumps over the lazy dog.", 50, 4],
                ["Could you please help me find the nearest hospital?", 50, 4],
                ["I dream of traveling to Spain next summer.", 50, 4],
                ["This sentence demonstrates attention patterns.", 50, 4],
            ],
            inputs=[input_text, max_len_slider, beam_width_slider],
        )

        # Event handlers with loading states
        translate_btn.click(
            fn=translate,
            inputs=[input_text, max_len_slider, beam_width_slider],
            outputs=[output_text, layer_dropdown, head_dropdown, attention_plot],
            api_name="translate"
        )

        input_text.submit(
            fn=translate,
            inputs=[input_text, max_len_slider, beam_width_slider],
            outputs=[output_text, layer_dropdown, head_dropdown, attention_plot]
        )

        # Update plot when layer or head changes
        for control in [layer_dropdown, head_dropdown]:
            control.change(
                fn=update_attention_plot,
                inputs=[layer_dropdown, head_dropdown, view_mode],
                outputs=[attention_plot]
            )

        # View mode change: update plot and toggle dropdown visibility
        def on_view_mode_change(layer_select, head_select, mode):
            plot = update_attention_plot(layer_select, head_select, mode)
            # Show/hide dropdowns based on view mode
            head_visible = mode != "All Heads Grid"    # Hide for All Heads Grid
            layer_visible = mode != "All Layers Grid"  # Hide for All Layers Grid
            return plot, gr.update(visible=head_visible), gr.update(visible=layer_visible)

        view_mode.change(
            fn=on_view_mode_change,
            inputs=[layer_dropdown, head_dropdown, view_mode],
            outputs=[attention_plot, head_dropdown, layer_dropdown]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio demo for attention visualization")
    parser.add_argument("-m", "--model", default="seq2seq_model.pkl",
                        help="Path to model weights file (default: seq2seq_model.pkl)")
    parser.add_argument("--vocab-encoder", default="src/vocab_encoder.json",
                        help="Path to vocab encoder JSON (default: src/vocab_encoder.json)")
    parser.add_argument("--vocab-merges", default="src/vocab_merges.json",
                        help="Path to vocab merges JSON (default: src/vocab_merges.json)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the server on (default: 7860)")
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.vocab_encoder, args.vocab_merges
    )

    generator = Generator(model, tokenizer)
    print("Model loaded successfully!")

    print("Launching Gradio demo...")
    demo = create_demo(generator)
    demo.launch(share=True, server_port=args.port)


if __name__ == "__main__":
    main()
