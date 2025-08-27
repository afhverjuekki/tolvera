"""
Comprehensive Taichi Language Essentials for Tölvera
====================================================
This module provides complete documentation of Taichi language features,
patterns, and critical rules for GPU-accelerated particle simulations.
"""

TAICHI_ESSENTIALS = """
# Taichi Language Essentials for Tölvera

## Type System

### Scalar Types
```python
ti.i32        # 32-bit integer
ti.f32        # 32-bit float (default for most calculations)
ti.u32        # 32-bit unsigned integer
```

### Vector Types
```python
ti.math.vec2  # 2D vector (x, y)
ti.math.vec3  # 3D vector (x, y, z)
ti.math.vec4  # 4D vector (r, g, b, a) for colors

# Creating vectors
pos = ti.math.vec2(100.0, 200.0)
color = ti.math.vec4(1.0, 0.0, 0.0, 1.0)

# Legacy style (still works)
vel = ti.Vector([vx, vy])
```

### Vector Operations
```python
# Arithmetic
v3 = v1 + v2         # Element-wise addition
v3 = v1 - v2         # Element-wise subtraction
v3 = v1 * scalar     # Scalar multiplication
v3 = v1 / scalar     # Scalar division

# Access components
x = vec.x  or  x = vec[0]
y = vec.y  or  y = vec[1]

# Methods (for ti.math.vec2/vec3/vec4)
length = vec.norm()           # Magnitude/length
direction = vec.normalized()  # Unit vector (CAREFUL: can fail if zero)
dot_product = v1.dot(v2)     # Dot product
```

## Math Functions

### Trigonometry (Use ti. prefix!)
```python
ti.sin(angle)     # Sine
ti.cos(angle)     # Cosine
ti.tan(angle)     # Tangent
ti.asin(x)        # Arcsine
ti.acos(x)        # Arccosine
ti.atan2(y, x)    # Arctangent (preferred for angles)
```

### Algebra
```python
ti.sqrt(x)        # Square root
ti.pow(x, n)      # Power
ti.exp(x)         # Exponential
ti.log(x)         # Natural logarithm
ti.abs(x)         # Absolute value
ti.sign(x)        # Sign (-1, 0, or 1)
```

### Utilities
```python
ti.random()                    # Random float [0, 1)
ti.random(ti.f32)             # Explicit type
ti.clamp(x, min, max)         # Constrain to range
ti.min(a, b)                  # Minimum
ti.max(a, b)                  # Maximum
ti.floor(x)                   # Round down
ti.ceil(x)                    # Round up
ti.cast(x, ti.i32)            # Type casting
```

### Constants
```python
# NO ti.pi or math.pi in Taichi scope!
PI = 3.14159265359
TWO_PI = 6.28318530718
HALF_PI = 1.57079632679
```

## CRITICAL: Variable Declaration Rules

### Rule 1: Declare ALL Variables Before Use
```python
# ❌ WRONG - Variable not defined in all paths
if condition:
    x = 5.0
else:
    x = 10.0
use(x)  # ERROR: x might not be defined

# ✅ CORRECT - Variable always defined
x = 10.0  # Default value
if condition:
    x = 5.0
use(x)  # OK: x is always defined
```

### Rule 2: Declare Before Conditional Blocks
```python
# ❌ WRONG - Variables declared inside conditional
if dist > 0:
    direction = diff / dist  # ERROR: direction not declared
    force = direction * 100
    
# ✅ CORRECT - Declare first, modify in conditional
direction = ti.math.vec2(0.0, 0.0)  # Declare with default
force = ti.math.vec2(0.0, 0.0)
if dist > 0:
    direction = diff / dist  # Modify existing variable
    force = direction * 100
```

## CRITICAL: No Returns in Conditionals/Loops

### Rule: Single Return at Function End
```python
# ❌ WRONG - Multiple returns, returns in conditionals
@ti.func
def bad_function(species: ti.i32) -> ti.math.vec2:
    if species == 0:
        return ti.math.vec2(100.0, 0.0)  # ERROR!
    elif species == 1:
        return ti.math.vec2(0.0, 100.0)  # ERROR!
    return ti.math.vec2(0.0, 0.0)

# ✅ CORRECT - Single return at end
@ti.func
def good_function(species: ti.i32) -> ti.math.vec2:
    result = ti.math.vec2(0.0, 0.0)  # Default
    if species == 0:
        result = ti.math.vec2(100.0, 0.0)
    elif species == 1:
        result = ti.math.vec2(0.0, 100.0)
    return result  # Single return
```

## Vector Normalization Patterns

### Safe Normalization (Avoid Division by Zero)
```python
# ❌ WRONG - Can crash if vec is zero
direction = vec.normalized()  # ERROR if vec.norm() == 0

# ✅ CORRECT - Check magnitude first
dist = vec.norm()
if dist > 0.001:  # Epsilon check
    direction = vec / dist  # Manual normalization
else:
    direction = ti.math.vec2(1.0, 0.0)  # Default direction
```

### Alternative Safe Normalization
```python
# Using ti.math.normalize with fallback
dist = vec.norm()
direction = ti.math.normalize(vec) if dist > 0.001 else ti.math.vec2(0.0, 0.0)
```

## Random Number Patterns

### Basic Random Values
```python
# Random float [0, 1)
r = ti.random()

# Random in range [min, max)
value = min + ti.random() * (max - min)

# Random integer
index = ti.cast(ti.random() * count, ti.i32)
```

### Random Directions
```python
# Random 2D direction (unit vector)
angle = ti.random() * 2.0 * 3.14159
direction = ti.math.vec2(ti.cos(angle), ti.sin(angle))

# Random velocity
speed = 50.0 + ti.random() * 100.0  # 50 to 150
velocity = direction * speed
```

### Random Position
```python
# Random position in screen
pos = ti.math.vec2(
    ti.random() * tv.x,
    ti.random() * tv.y
)

# Random in circle
angle = ti.random() * 2.0 * 3.14159
radius = ti.sqrt(ti.random()) * max_radius  # Uniform distribution
pos = center + ti.math.vec2(ti.cos(angle), ti.sin(angle)) * radius
```

## Loops and Iteration

### Static Range (Compile-time known)
```python
# Use ti.static for compile-time constants
for i in ti.static(range(3)):  # Unrolled at compile time
    process(i)
```

### Dynamic Range
```python
# Regular range for runtime values
for i in range(tv.pn):  # Runtime particle count
    if tv.p.field[i].active > 0:
        process(i)
```

### Nested Loops
```python
# 2D iteration
for x, y in ti.ndrange(tv.x, tv.y):
    tv.px.px.rgba[x, y] = color

# With bounds
for x, y in ti.ndrange((10, 20), (30, 40)):
    # x: 10 to 19, y: 30 to 39
    process(x, y)
```

## Function Decorators

### @ti.func - Taichi Function
```python
@ti.func
def calculate_force(...) -> ti.math.vec2:
    # Can be called from kernels or other funcs
    # Inlined during compilation
    return force
```

### @ti.kernel - Taichi Kernel
```python
@ti.kernel
def update_particles():
    # Entry point from Python
    # Launches GPU computation
    for i in range(tv.pn):
        # Parallel execution
```

## Common Pitfalls and Solutions

### Pitfall 1: Using Python Math in Taichi
```python
# ❌ WRONG
import math
angle = math.sin(x)  # ERROR in Taichi scope

# ✅ CORRECT
angle = ti.sin(x)  # Use ti. prefix
```

### Pitfall 2: Forgetting Type Casting
```python
# ❌ WRONG
tv.px.px.rgba[pos.x, pos.y] = color  # ERROR: float indices

# ✅ CORRECT
xi = ti.cast(pos.x, ti.i32)
yi = ti.cast(pos.y, ti.i32)
tv.px.px.rgba[xi, yi] = color
```

### Pitfall 3: Modifying Loop Variable
```python
# ❌ WRONG
for i in range(10):
    i = i * 2  # ERROR: Can't modify loop variable

# ✅ CORRECT
for i in range(10):
    j = i * 2  # Use different variable
```

## Performance Tips

1. **Minimize Branching**: GPUs prefer uniform execution
2. **Use Local Variables**: Reduce global memory access
3. **Avoid Complex Conditionals**: Simple if/else is better
4. **Vectorize Operations**: Use vector ops instead of component-wise
5. **Early Exit**: Skip unnecessary calculations

```python
# Early exit pattern
if dist > max_range:
    continue  # Skip this iteration

# Vectorized operation
new_pos = pos + vel * dt  # Better than component-wise
```
"""

TAICHI_CRASH_FIXES = """
# Common Taichi Crash Patterns and Fixes

## Return Statement Crashes

### Error: "Return inside non-static if"
```python
# ❌ CRASH - Return in conditional
if condition:
    return value  # CRASH!

# ✅ FIX - Single return at end
result = default_value
if condition:
    result = value
return result
```

## Variable Declaration Crashes

### Error: "Variable used before assignment"
```python
# ❌ CRASH - Variable not always defined
if x > 0:
    y = x * 2
z = y + 1  # CRASH: y might not exist

# ✅ FIX - Always declare first
y = 0  # Default
if x > 0:
    y = x * 2
z = y + 1  # OK: y always exists
```

## Division by Zero Crashes

### Error: NaN or Inf values
```python
# ❌ CRASH - Direct normalization
direction = vec.normalized()  # CRASH if vec is zero

# ✅ FIX - Check magnitude
if vec.norm() > 0.001:
    direction = vec.normalized()
else:
    direction = ti.math.vec2(0.0, 0.0)
```

## Array Access Crashes

### Error: "Index out of bounds"
```python
# ❌ CRASH - No bounds check
color = tv.px.px.rgba[x, y]  # CRASH if x,y outside screen

# ✅ FIX - Always check bounds
if 0 <= x < tv.x and 0 <= y < tv.y:
    color = tv.px.px.rgba[x, y]
```
"""

# Export all documentation
__all__ = ['TAICHI_ESSENTIALS', 'TAICHI_CRASH_FIXES']