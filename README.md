# Conditional Diffusion Model for Sketch Generation

A PyTorch implementation of a conditional diffusion model that generates sketches from the Quick, Draw! dataset. The model learns to generate stroke sequences for specific categories (cat, bus, rabbit) through a denoising diffusion process that preserves temporal drawing order and semantic stroke information.

##  Project Overview

This project implements a novel approach to sketch generation by treating sketches as temporal sequences of strokes rather than static images. The model uses a specialized architecture that combines:

- **Temporal Modeling**: LSTM-based sequence processing that captures stroke order
- **Semantic Conditioning**: Category-specific generation control
- **Stroke-Level Understanding**: Pen state encoding for drawing semantics
- **Diffusion Process**: Progressive denoising for high-quality generation

##  Architecture

The model consists of four specialized components:

1. **Stroke Embedder**: Transforms `[x, y, pen_state]` coordinates into latent embeddings
2. **Category Embedder**: Provides semantic conditioning for specific sketch categories  
3. **Temporal Encoder**: 2-layer bidirectional LSTM for sequential stroke modeling
4. **Noise Predictor**: Maps hidden states back to coordinate space for denoising

**Key Design Insight**: Temporal order (LSTM) outperforms spatial relationships (attention) for sketch generation, validated through ablation studies showing FID improvements.

##  Project Structure

```
diffusion_model/
├── data/                           # Raw NDJSON sketch data
│   ├── bus.ndjson                 # Quick Draw bus sketches
│   ├── cat.ndjson                 # Quick Draw cat sketches  
│   └── rabbit.ndjson              # Quick Draw rabbit sketches
├── subset/                        # Train/test split indices
│   ├── bus/indices.json
│   ├── cat/indices.json
│   └── rabbit/indices.json
├── models/                        # Trained model checkpoints
│   ├── sketch_diffusion_bus_*.pth
│   ├── sketch_diffusion_cat_*.pth
│   └── sketch_diffusion_rabbit_*.pth
├── data_preprocessing.ipynb       # Data pipeline notebook
├── diffusion_model_traning.ipynb # Model training notebook
├── model_evaluation.ipynb        # Evaluation and metrics
├── processed_datasets.pkl        # Preprocessed training data
└── eval_results.json            # Performance metrics
```

##  Workflow and Usage

### 1. Data Preprocessing (`data_preprocessing.ipynb`)

**Purpose**: Convert raw Quick, Draw! NDJSON data into training-ready sequences

**Key Steps**:
- Load sketch data and train/test indices
- Extract stroke sequences and assign semantic pen states:
  - `-1`: Padding tokens
  - `0`: End stroke (pen lift)  
  - `1`: Start stroke (pen down)
  - `2`: Continue stroke (drawing)
  - `3`: End of drawing
- Normalize coordinates to [-1, 1] range
- Pad sequences to uniform length
- Save preprocessed data as `processed_datasets.pkl`

**Output**: Structured dataset with train/test splits for each category

### 2. Model Training (`diffusion_model_traning.ipynb`)

**Purpose**: Train conditional diffusion models for sketch generation

**Key Components**:
- **Model Architecture**: Define 4-component modular architecture (~6.2M parameters)
- **Training Loop**: Implement diffusion training with progressive noise addition
- **Loss Function**: MSE with specialized pen state weighting (0.3x multiplier)
- **Optimization**: AdamW with OneCycleLR scheduling
- **Checkpointing**: Save complete model states with training metadata

**Configuration**:
- **Epochs**: 175-200 (category-dependent)
- **Learning Rate**: 1e-4 with cosine annealing
- **Batch Size**: 128
- **Diffusion Steps**: 1000 training / 500-1000 inference

**Output**: Trained model checkpoints in `./models/` directory

### 3. Model Evaluation (`model_evaluation.ipynb`)

**Purpose**: Quantitative and qualitative assessment of generated sketches

**Evaluation Methods**:
- **FID Score**: Fréchet Inception Distance for generation quality
- **KID Score**: Kernel Inception Distance for distribution matching
- **Visual Inspection**: Stroke-level visualization and semantic validation

**Process**:
1. Load trained models and generate sample sketches
2. Convert stroke sequences to images for metric calculation
3. Compare generated vs real sketch distributions
4. Output comprehensive evaluation results

## 📊 Performance Results

| Category | FID Score | KID Score | Sequence Length | Training Epochs |
|----------|-----------|-----------|-----------------|-----------------|
| Rabbit   | 48.33     | 0.046     | 571             | 200             |
| Cat      | 113.2     | 0.132     | 425             | 200             |
| Bus      | 170.2     | 0.199     | 451             | 175             |

*Lower scores indicate better generation quality*

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib seaborn pandas pillow torch-fidelity
```

### Usage Example
```python
# Load preprocessed data
import pickle
with open("processed_datasets.pkl", "rb") as file:
    datasets = pickle.load(file)

# Load trained model
model_path = "./models/sketch_diffusion_rabbit_*.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path, map_location=device)

# Generate sketch
category_id = 2  # rabbit
seq_length = 571
sketch = generate_simple_sketch(trained_model, category_id, seq_length)
```

##  Key Features

- **Temporal Preservation**: Maintains natural drawing order through LSTM processing
- **Semantic Encoding**: Rich pen state representation for stroke semantics
- **Category Conditioning**: Controllable generation across different sketch types
- **Quality Validation**: Comprehensive evaluation with standard metrics
- **Modular Design**: Easily extensible architecture for new categories

##  Technical Details

- **Model Size**: ~6.2M parameters with embedding_dim=256, hidden_dim=512
- **Training Strategy**: Progressive noise scheduling with cosine beta schedule
- **Memory Efficiency**: Gradient clipping and optimized batch processing
- **Reproducibility**: Fixed random seeds and deterministic operations

## 🔮 Future Improvements

- **Multi-Category Models**: Unified architecture for cross-category generation
- **Hierarchical Generation**: Stroke-level then point-level modeling
- **Semantic Control**: Enhanced category conditioning mechanisms
- **Variable Length**: Dynamic sequence generation without padding