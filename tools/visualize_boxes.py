"""Example: Visualizing oriented bounding boxes.

This script demonstrates how to:
1. Create and manipulate oriented boxes
2. Convert between different representations
3. Visualize boxes on images
4. Apply transformations
"""

import argparse
from pathlib import Path
import sys
import math

try:
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Required dependencies not installed: {e}")
    print("Please install: pip install Pillow numpy")
    sys.exit(1)

from oriented_det import Polygon, QBox, RBox, transforms
from oriented_det.utils import viz


def create_example_boxes():
    """Create example oriented boxes for visualization."""
    boxes = [
        # Axis-aligned box
        RBox(cx=100, cy=100, width=80, height=40, angle=0),
        
        # Rotated box
        RBox(cx=200, cy=150, width=60, height=30, angle=math.radians(45)),
        
        # Another rotated box
        RBox(cx=300, cy=200, width=100, height=50, angle=math.radians(-30)),
        
        # Box from quadrilateral
        QBox([(400, 100), (500, 120), (480, 200), (380, 180)]),
        
        # Box from polygon
        Polygon([(100, 250), (150, 240), (160, 290), (110, 300)]),
    ]
    return boxes


def demonstrate_conversions():
    """Demonstrate conversions between different box representations."""
    print("Demonstrating box conversions:")
    print("-" * 50)
    
    # Start with an RBox
    rbox = RBox(cx=256, cy=256, width=100, height=50, angle=math.radians(30))
    print(f"Original RBox: cx={rbox.cx:.1f}, cy={rbox.cy:.1f}, "
          f"w={rbox.width:.1f}, h={rbox.height:.1f}, angle={math.degrees(rbox.angle):.1f}°")
    
    # Convert to QBox
    qbox = transforms.rbox_to_qbox(rbox)
    print(f"\nConverted to QBox:")
    for i, point in enumerate(qbox.points):
        print(f"  Point {i+1}: ({point[0]:.1f}, {point[1]:.1f})")
    
    # Convert to Polygon
    poly = transforms.rbox_to_polygon(rbox)
    print(f"\nConverted to Polygon:")
    print(f"  Area: {poly.area:.1f}")
    print(f"  Centroid: ({poly.centroid[0]:.1f}, {poly.centroid[1]:.1f})")
    print(f"  Bounds: {poly.bounds}")
    
    # Round trip: RBox -> QBox -> RBox
    recovered = transforms.qbox_to_rbox(qbox)
    print(f"\nRound trip (RBox -> QBox -> RBox):")
    print(f"  Original angle: {math.degrees(rbox.angle):.1f}°")
    print(f"  Recovered angle: {math.degrees(recovered.angle):.1f}°")
    print(f"  Difference: {abs(rbox.angle - recovered.angle):.6f} rad")
    print()


def visualize_on_image(image_path: Path = None, output_path: Path = None):
    """Visualize boxes on an image.
    
    Args:
        image_path: Path to input image (creates blank if None)
        output_path: Path to save visualization
    """
    # Create or load image
    if image_path and image_path.exists():
        image = Image.open(image_path).convert("RGB")
        print(f"Loaded image: {image.size}")
    else:
        # Create a blank image for demonstration
        image = Image.new("RGB", (600, 400), color="white")
        print("Created blank image for demonstration")
    
    # Create example boxes
    boxes = create_example_boxes()
    print(f"\nVisualizing {len(boxes)} boxes:")
    
    # Convert all to polygons for visualization
    polygons = []
    for i, box in enumerate(boxes):
        if isinstance(box, RBox):
            poly = box.to_polygon()
            label = f"RBox {i+1}: {math.degrees(box.angle):.0f}°"
        elif isinstance(box, QBox):
            poly = box.to_polygon()
            label = f"QBox {i+1}"
        elif isinstance(box, Polygon):
            poly = box
            label = f"Polygon {i+1}"
        else:
            continue
        
        polygons.append(poly.points)
        print(f"  Box {i+1}: {type(box).__name__}, area={poly.area:.1f}")
    
    # Draw boxes
    result = viz.draw_polygons(
        image,
        polygons,
    )
    
    # Save or show
    if output_path:
        result.save(output_path)
        print(f"\nSaved visualization to: {output_path}")
    else:
        print("\nVisualization created (use --output to save)")
    
    return result


def demonstrate_transforms():
    """Demonstrate geometric transformations."""
    print("\nDemonstrating transformations:")
    print("-" * 50)
    
    from oriented_det.data import HorizontalFlip, Rotate
    
    rbox = RBox(cx=200, cy=150, width=80, height=40, angle=math.radians(30))
    print(f"Original: cx={rbox.cx:.1f}, cy={rbox.cy:.1f}, angle={math.degrees(rbox.angle):.1f}°")
    
    # Horizontal flip
    flip = HorizontalFlip(p=1.0)
    flipped = flip.apply_to_rbox(rbox, image_width=400, image_height=300)
    print(f"Flipped:  cx={flipped.cx:.1f}, cy={flipped.cy:.1f}, angle={math.degrees(flipped.angle):.1f}°")
    
    # Rotation
    rotate = Rotate(degrees=90, p=1.0)
    rotated = rotate.apply_to_rbox(rbox, image_width=400, image_height=300)
    print(f"Rotated:  cx={rotated.cx:.1f}, cy={rotated.cy:.1f}, angle={math.degrees(rotated.angle):.1f}°")
    print()


def main():
    parser = argparse.ArgumentParser(description="Visualize oriented bounding boxes")
    parser.add_argument("--image", type=Path, help="Input image path (creates blank if not provided)")
    parser.add_argument("--output", type=Path, default=Path("visualization.png"),
                       help="Output image path")
    parser.add_argument("--demo-conversions", action="store_true",
                       help="Demonstrate box conversions")
    parser.add_argument("--demo-transforms", action="store_true",
                       help="Demonstrate transformations")
    parser.add_argument("--all", action="store_true",
                       help="Run all demonstrations")
    
    args = parser.parse_args()
    
    if args.all or args.demo_conversions:
        demonstrate_conversions()
    
    if args.all or args.demo_transforms:
        demonstrate_transforms()
    
    # Always visualize
    visualize_on_image(args.image, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
