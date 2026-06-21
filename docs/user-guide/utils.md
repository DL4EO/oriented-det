# Utilities Module

The utils module provides low-level config loading (`load_config`, `FrozenConfig`) and visualization helpers. **Training experiments** use [`TrainingExperimentConfig`](../api/train.md) and JSON under `configs/` — see [Configuration](configuration.md).

## Configuration

### FrozenConfig

Immutable configuration object:

```python
from oriented_det.utils import FrozenConfig

config = FrozenConfig({
    "model": {"num_classes": 15},
    "train": {"epochs": 12}
})

# Access with dot notation
print(config.model.num_classes)  # 15

# Immutable - cannot modify
# config.model.num_classes = 20  # Raises error
```

### Loading Configs

```python
from oriented_det.utils import load_config

# From file
config = load_config("config.yaml")

# From dict
config = load_config({"model": {"num_classes": 15}})

# With overrides
config = load_config(
    "config.yaml",
    overrides=["model.num_classes=20", "train.epochs=24"]
)

# Supports nested config inheritance via _base_ field
# config.json:
# {
#   "_base_": ["../_base_/datasets/dota_le90.json"],
#   "model": {"num_classes": 15}
# }
config = load_config("config.json")  # Automatically loads and merges base configs
```

### Overrides

Apply dotted-key overrides:

```python
from oriented_det.utils import apply_overrides

config = {"model": {"num_classes": 15}}
overrides = ["model.num_classes=20", "train.epochs=24"]
updated = apply_overrides(config, overrides)
```

## Visualization

### Drawing Boxes

```python
from oriented_det.utils import viz
from oriented_det.geometry import RBox
from PIL import Image

image = Image.open("image.jpg")

# Draw RBoxes
rboxes = [RBox(100, 200, 50, 30, 0.5)]
result = viz.draw_boxes(image, rboxes)

# Save
result.save("output.jpg")
```

### Drawing Polygons

```python
from oriented_det.utils import viz
from oriented_det.geometry import Polygon

polygons = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
result = viz.draw_polygons(image, [p.points for p in polygons])
```

### Customization

```python
from oriented_det.utils import viz, DrawingSpec

# Custom drawing specs
specs = [
    DrawingSpec(outline=(255, 0, 0), width=3),  # Red, thick
    DrawingSpec(outline=(0, 255, 0), width=2),  # Green, medium
]

result = viz.draw_polygons(image, polygons, specs=specs)
```

### Color Palettes

```python
from oriented_det.utils import viz

# Cycle through default palette
colors = viz.cycle_palette(10)

# Random palette
colors = viz.random_palette(10, seed=42)
```

## Configuration Best Practices

1. **Use FrozenConfig** - Prevents accidental modifications
2. **Use overrides** - Apply command-line or environment-based overrides easily
3. **Nested access** - Use dot notation for cleaner code
4. **Type safety** - Configs maintain type information
5. **Use `_base_` inheritance** - Factorize common settings into base configs
   - Create base configs for datasets, schedules, and models
   - Inherit from multiple bases and override as needed
   - Paths are resolved relative to the config file's directory

## Visualization Best Practices

1. **Use PIL Images** - Works best with Pillow Image objects
2. **Custom specs** - Use `DrawingSpec` for per-box customization
3. **Color palettes** - Use `cycle_palette()` or `random_palette()` for consistent colors
4. **Save results** - Always save visualization outputs for debugging

## Logging and Debugging

The utils module provides logging and tracing utilities for debugging and performance analysis.

### TraceLogger

`TraceLogger` provides configurable logging for tracing function execution with timing information.

**Basic usage:**

```python
from oriented_det.utils import TraceLogger

# Enable tracing
TraceLogger.enable()

# Trace a message
TraceLogger.trace("Processing image: {}", "image_001.png")

# Trace a code block with timing
with TraceLogger.trace_block("Loading dataset"):
    dataset = load_dataset()
    # ... dataset loading code ...

# Trace a value
TraceLogger.trace_value("batch_size", 16)
TraceLogger.trace_value("learning_rate", 0.001, unit="lr")

# Disable tracing
TraceLogger.disable()
```

**Output example:**
```
Processing image: image_001.png
→ Loading dataset
  ... (indented code execution)
✅ Loading dataset (2.345s)
batch_size: 16
learning_rate: 0.001 lr
```

### Global Logger Instance

A global `logger` instance is available for convenience:

```python
from oriented_det.utils import logger

logger.enable()
logger.trace("Message")
logger.disable()
```

### Function Tracing Decorator

Use `trace_function()` to automatically trace function calls:

```python
from oriented_det.utils import trace_function, enable_tracing

# Enable tracing globally
enable_tracing()

@trace_function()
def process_image(image_path: str, threshold: float = 0.5):
    # Function body
    return result

# Custom function name in logs
@trace_function("custom_process")
def another_function():
    pass
```

**Output example:**
```
→ process_image(image_001.png, threshold=0.5)
  ... (function execution)
✅ process_image returned (0.123s)
```

**Features:**
- Automatically logs function entry and exit
- Shows function arguments (truncated for readability)
- Measures execution time
- Handles exceptions (logs error with timing)
- Respects indentation for nested calls

### Convenience Functions

```python
from oriented_det.utils import (
    enable_tracing,
    disable_tracing,
    is_tracing_enabled,
)

# Enable/disable tracing
enable_tracing()
disable_tracing()

# Check if tracing is enabled
if is_tracing_enabled():
    print("Tracing is active")
```

### Use Cases

**1. Debugging function calls:**

```python
from oriented_det.utils import trace_function, enable_tracing

enable_tracing()

@trace_function()
def forward_pass(model, images):
    return model(images)

# See which functions are called and how long they take
output = forward_pass(model, images)
```

**2. Performance analysis:**

```python
from oriented_det.utils import TraceLogger

TraceLogger.enable()

with TraceLogger.trace_block("Training epoch"):
    for batch in dataloader:
        with TraceLogger.trace_block("Processing batch"):
            loss = train_step(batch)
```

**3. Conditional tracing:**

```python
import os
from oriented_det.utils import enable_tracing, disable_tracing

# Enable tracing only in debug mode
if os.environ.get("DEBUG", "0") == "1":
    enable_tracing()
else:
    disable_tracing()
```

**4. Tracing model forward pass:**

```python
from oriented_det.utils import trace_function, enable_tracing
from oriented_det import OrientedRCNN

enable_tracing()

# Trace model forward pass
model = OrientedRCNN(num_classes=15)

@trace_function("model_forward")
def traced_forward(model, images, targets=None):
    return model(images, targets)

output = traced_forward(model, images, targets)
```

### Best Practices

1. **Enable/disable as needed**: Tracing adds overhead, disable in production
2. **Use environment variables**: Control tracing via `DEBUG` or `TRACE` env vars
3. **Trace key functions**: Focus on important functions, not every function
4. **Use trace blocks**: For code sections, use `trace_block` context manager
5. **Check before tracing**: Use `is_tracing_enabled()` to conditionally trace

**Example: Environment-based tracing:**

```python
import os
from oriented_det.utils import enable_tracing, trace_function

# Enable if DEBUG environment variable is set
if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
    enable_tracing()

@trace_function()
def my_function():
    # Only traced if DEBUG is set
    pass
```

## See Also

- [API Reference](../api/utils.md) - Complete API documentation

