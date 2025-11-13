import os
import sys
import traceback
import torch
import numpy as np
from PIL import Image, ImageDraw

from experiment_modules.architecture_trainer import ArchitectureTrainer
from blueprint_modules.network import NeuralArchitecture


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> dict:
    """Load torch checkpoint safely onto given device"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def draw_architecture_diagram(architecture: NeuralArchitecture, out_dir: str, episode: int = 199, step: int = 0):
    """Draw architecture to JPEG using a fast PIL-based layout similar to trainer._draw_architecture_diagram

    Produces: {out_dir}/ep{episode:03d}/step{step:03d}.jpg
    """
    try:
        save_dir = os.path.join(out_dir, f'ep{episode:03d}')
        os.makedirs(save_dir, exist_ok=True)

        CANVAS_WIDTH = 1000
        CANVAS_HEIGHT = 800
        MARGIN = 50
        INPUT_OUTPUT_RADIUS = 3
        HIDDEN_RADIUS = 8

        img = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), color='white')
        draw = ImageDraw.Draw(img)

        input_neurons = []
        hidden_neurons = []
        output_neurons = []

        for neuron_id, neuron in architecture.neurons.items():
            if neuron.neuron_type.value == 'input':
                input_neurons.append(neuron_id)
            elif neuron.neuron_type.value == 'hidden':
                hidden_neurons.append(neuron_id)
            elif neuron.neuron_type.value == 'output':
                output_neurons.append(neuron_id)

        pos = {}

        # Inputs on left
        input_neurons.sort()
        if input_neurons:
            y_step = (CANVAS_HEIGHT - 2 * MARGIN) / max(len(input_neurons) - 1, 1)
            for i, neuron_id in enumerate(input_neurons):
                x = MARGIN
                y = MARGIN + int(i * y_step)
                pos[neuron_id] = (x, y)

        # Outputs on right
        output_neurons.sort()
        if output_neurons:
            y_step = (CANVAS_HEIGHT - 2 * MARGIN) / max(len(output_neurons) - 1, 1)
            for i, neuron_id in enumerate(output_neurons):
                x = CANVAS_WIDTH - MARGIN
                y = MARGIN + int(i * y_step)
                pos[neuron_id] = (x, y)

        # Hidden neurons by layer_position
        for neuron_id in hidden_neurons:
            neuron = architecture.neurons[neuron_id]
            layer_pos = neuron.layer_position
            x = MARGIN + int(layer_pos * (CANVAS_WIDTH - 2 * MARGIN))
            rng = np.random.RandomState(neuron_id)
            y = MARGIN + int(rng.uniform(0, 1) * (CANVAS_HEIGHT - 2 * MARGIN))
            pos[neuron_id] = (x, y)

        # Draw edges first
        for conn in architecture.connections:
            if getattr(conn, 'enabled', True) and conn.source_id in pos and conn.target_id in pos:
                x1, y1 = pos[conn.source_id]
                x2, y2 = pos[conn.target_id]
                edge_color = (200, 100, 100) if conn.weight < 0 else (100, 100, 200)
                draw.line([(x1, y1), (x2, y2)], fill=edge_color, width=1)

        color_map = {
            'input': (52, 152, 219),
            'hidden': (46, 204, 113),
            'output': (231, 76, 60)
        }

        for neuron_id, neuron in architecture.neurons.items():
            if neuron_id in pos:
                x, y = pos[neuron_id]
                color = color_map.get(neuron.neuron_type.value, (128, 128, 128))
                radius = HIDDEN_RADIUS if neuron.neuron_type.value == 'hidden' else INPUT_OUTPUT_RADIUS
                draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color, outline=(0, 0, 0))

        title = f'Episode {episode}, Step {step}: {len(architecture.neurons)} neurons, {len(architecture.connections)} connections'
        draw.text((20, 10), title, fill=(0, 0, 0))

        diagram_file = os.path.join(save_dir, f'step{step:03d}.jpg')
        img.save(diagram_file, quality=70, optimize=True)
        print(f"Saved diagram: {diagram_file}")

    except Exception as e:
        print(f"Failed to draw architecture diagram: {e}")
        traceback.print_exc()


def main(argv):
    # We will collect training_history from a set of checkpoints and write them to CSV
    episodes = [0, 50, 100, 150, 199]
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'analysis', 'results', 'checkpoints')
    out_csv = os.path.join(os.path.dirname(__file__), 'analysis', 'results', 'training_history_summary.csv')

    all_rows = []
    header_keys = set()

    for ep in episodes:
        path = os.path.join(ckpt_dir, f'checkpoint_ep{ep}.pth')
        if not os.path.exists(path):
            print(f"Checkpoint missing: {path} -- skipping")
            continue

        try:
            ckpt = torch.load(path, map_location='cpu')
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        th = ckpt.get('training_history', None)

        if th is None:
            # If training_history missing, try to salvage top-level metrics
            fallback = {}
            for k in ('best_accuracy', 'episode'):
                if k in ckpt:
                    fallback[k] = ckpt[k]
            row = {
                'checkpoint_episode': ep,
                'checkpoint_path': path,
                'history_index': 0,
                'note': 'no_training_history',
            }
            row.update(fallback)
            header_keys.update(row.keys())
            all_rows.append(row)
            continue

        # Expect training_history to be a list of dicts
        if isinstance(th, list):
            for idx, item in enumerate(th):
                row = {
                    'checkpoint_episode': ep,
                    'checkpoint_path': path,
                    'history_index': idx,
                }
                if isinstance(item, dict):
                    # merge item keys
                    for k, v in item.items():
                        # flatten lists/complex types to string for CSV safety
                        if isinstance(v, (list, dict)):
                            try:
                                row[k] = str(v)
                            except Exception:
                                row[k] = ''
                        else:
                            row[k] = v
                        header_keys.add(k)
                else:
                    row['value'] = item
                    header_keys.add('value')

                header_keys.update(['checkpoint_episode', 'checkpoint_path', 'history_index'])
                all_rows.append(row)
        else:
            # Unexpected type
            row = {
                'checkpoint_episode': ep,
                'checkpoint_path': path,
                'history_index': 0,
                'training_history_raw': str(th)
            }
            header_keys.update(row.keys())
            all_rows.append(row)

    # Write CSV
    import csv
    # Build ordered header: provenance first, then sorted metric keys
    prov_keys = ['checkpoint_episode', 'checkpoint_path', 'history_index']
    metric_keys = sorted(k for k in header_keys if k not in prov_keys)
    header = prov_keys + metric_keys

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in all_rows:
            # Ensure all keys present
            out = {k: r.get(k, '') for k in header}
            writer.writerow(out)

    print(f"Wrote training history summary to: {out_csv}")


if __name__ == '__main__':
    main(sys.argv)
