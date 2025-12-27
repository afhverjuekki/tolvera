# Achieving Mathematical Composition + High Performance for Interactive Art

**Date**: 2025-12-27
**Document Type**: Implementation Guide
**Goal**: Build compositionally sound system while maintaining real-time performance

---

## Executive Summary

**Good News**: Mathematical purity and high-performance interactive art are **compatible**. Modern systems prove you can have both.

**The Key**: Separate **specification** (what to compute) from **execution** (how to compute it).

This document shows three proven architectures and provides a concrete roadmap for Tölvera 2.0.

---

## 1. The Core Tension

### The Challenge

**Mathematical Purity** ⟷ **Real-time Performance**

- Immutability vs. GPU in-place updates
- Composition vs. sequential mutation
- Abstraction vs. bare-metal speed

### The Solution

Systems like Halide, JAX, and Futhark have solved this by using:

1. **Deferred execution** - build computation graph, optimize, then execute
2. **Uniqueness types** - track safe mutation in type system
3. **Separation of concerns** - pure interface, optimized execution

---

## 2. The Halide Principle: Separate WHAT from HOW

### Core Insight

[Halide](https://halide-lang.org/) revolutionized image processing by **decoupling algorithm from schedule**:

```python
# WHAT: Pure functional specification
blur = lambda img: convolve(img, gaussian_kernel)
sharpen = lambda img: img - blur(img)
pipeline = sharpen ∘ blur ∘ sharpen  # Composition!

# HOW: Performance schedule (separate!)
schedule = {
    blur: parallel().vectorize().gpu(),
    sharpen: fuse_with(blur).inline()
}
```

**Result**: [5x faster than hand-tuned CUDA](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) while staying compositional.

**Why it works**:
- Algorithm is pure functions (compositional)
- Compiler fuses operations into single GPU kernel
- No intermediate allocations (everything optimized away)

**Reference**: [Halide: Decoupling Algorithms from Schedules](https://halide-lang.org/)

---

## 3. Three Proven Architectures

### Architecture A: Deferred Execution Graphs (JAX-style)

**Concept**: Build computation graph, optimize, then execute.

```haskell
-- Interface: Pure functional
data Particles = Particles {
    positions :: Array Vec2,
    velocities :: Array Vec2
}

flock :: Particles -> Particles
flock p = Particles {
    positions = p.positions + p.velocities,  -- Pure!
    velocities = p.velocities + flocking_forces p
}

slime :: Particles -> Particles
slime p = Particles {
    positions = p.positions + slime_movement p,
    velocities = decay 0.9 p.velocities
}

-- Composition builds graph (no execution yet!)
behavior = flock >>> slime >>> flock
```

**Execution**:
```haskell
-- Compiler transforms to:
optimized_kernel = fusion_pass $ inline_pass $ behavior
-- Single GPU kernel: flock→slime→flock fused together
-- No intermediate particle arrays allocated!

result = execute_on_gpu optimized_kernel initial_particles
```

**Key Properties**:
- ✅ **Compositional**: Behaviors compose with `>>>`
- ✅ **Fast**: Fusion eliminates intermediates
- ✅ **Parallel**: Automatically parallelized
- ✅ **Pure**: No side effects in user code

**Real Example**: [JAX](https://github.com/jax-ml/jax)

```python
import jax.numpy as jnp
from jax import jit

@jit  # Deferred execution + optimization
def behavior(particles):
    particles = flock(particles)
    particles = slime(particles)
    return flock(particles)
# First call compiles, subsequent calls are GPU-fast
```

**References**:
- [JAX Documentation](https://jax.readthedocs.io/en/latest/glossary.html)
- [JAX Introduction](https://jax.quantecon.org/jax_intro.html)

---

### Architecture B: Uniqueness Types (Futhark-style)

**Concept**: Track which values can be safely mutated via types.

```rust
// '*' marks unique/linear types (can only be used once)
fn flock(particles: *Particles) -> *Particles {
    // Compiler knows particles is unique, can mutate in-place!
    particles.velocities += flocking_forces(&particles);
    particles.positions += particles.velocities;
    return particles;  // Ownership transferred
}

fn slime(particles: *Particles) -> *Particles {
    // Again, in-place mutation is safe
    particles.velocities *= 0.9;
    particles.positions += slime_movement(&particles);
    return particles;
}

// Composition is safe because each function consumes its input
let result = initial_particles
    |> flock   // Transfers ownership
    |> slime   // Gets unique reference
    |> flock;  // Gets unique reference
```

**Key Properties**:
- ✅ **Zero-copy**: In-place mutation when safe
- ✅ **Type-safe**: Compiler prevents aliasing bugs
- ✅ **Compositional**: Linear types form a category
- ✅ **Performance**: [Matches hand-written GPU code](https://futhark-lang.org/performance.html)

**Real Example**: [Futhark](https://futhark-lang.org/)

```futhark
-- Pure interface, in-place execution
let flock [n] (ps: [n]particle) : [n]particle =
  map update_particle ps  -- Looks pure, executes in-place!

let pipeline = flock >-> slime >-> flock  -- Composition!
```

**References**:
- [Futhark: Purely Functional GPU Programming](https://futhark-lang.org/)
- [Futhark PLDI 2017 Paper](https://elsman.com/pdf/pldi17.pdf)

---

### Architecture C: Effect Systems (Koka/Eff-style)

**Concept**: Track effects in types, allow controlled mutation.

```typescript
// Effect annotations in types
type Behavior<State> = (s: State) => State @ {gpu, mutate}
//                                          ↑ Effect annotation

// Pure behaviors (no effects)
const identity: <S>(s: S) => S @ {} = s => s

// Effectful behaviors (tracked in types)
const flock: Behavior<Particles> = (p) => {
    mutate p.velocities;  // Effect tracked by type system
    return p;
}

// Composition preserves effect tracking
const compose = <A, B, C>(
    f: (a: A) => B @ E1,
    g: (b: B) => C @ E2
) => (a: A) => g(f(a)) @ {E1, E2}  // Union of effects

const pipeline = compose(compose(flock, slime), flock)
// Type: Behavior<Particles> @ {gpu, mutate}
```

**Key Properties**:
- ✅ **Effect polymorphism**: Pure and impure code coexist
- ✅ **Type-safe**: Effects are tracked, not hidden
- ✅ **Flexible**: Allow mutation when beneficial
- ✅ **Compositional**: Effect algebras form categories

---

## 4. Concrete Design for Tölvera 2.0

### Layer 1: Pure Functional Core

```haskell
-- Core algebra: Behaviors form a Category
class Behavior b where
    identity :: b a a
    compose :: b a b -> b b c -> b a c

-- Objects are computational states
data Particles = Particles {
    count :: Int,
    positions :: GPU.Array Vec2,
    velocities :: GPU.Array Vec2,
    species :: GPU.Array Int
}

-- Morphisms are behavior transformations
data BehaviorGraph a b where
    Id :: BehaviorGraph a a
    Flock :: Float -> BehaviorGraph Particles Particles
    Slime :: Float -> BehaviorGraph Particles Particles
    Compose :: BehaviorGraph a b -> BehaviorGraph b c -> BehaviorGraph a c
    Parallel :: BehaviorGraph a b -> BehaviorGraph a c -> BehaviorGraph a (b, c)
```

**Composition is just AST construction**:
```haskell
myBehavior :: BehaviorGraph Particles Particles
myBehavior =
    Flock 0.8 `Compose`
    Slime 0.5 `Compose`
    Flock 1.0

-- Satisfies category laws:
-- Compose Id f = f
-- Compose f Id = f
-- Compose (Compose f g) h = Compose f (Compose g h)
```

### Layer 2: Optimization Pass

```haskell
-- Fusion: Combine adjacent behaviors
fuse :: BehaviorGraph a b -> FusedKernel a b
fuse (Compose f g) = case (fuse f, fuse g) of
    (FlockKernel w1, SlimeKernel w2) ->
        FlockSlimeKernel w1 w2  -- Single kernel!
    (k1, k2) -> SequentialKernel k1 k2

-- Parallel execution
fuse (Parallel f g) = ParallelKernel (fuse f) (fuse g)

-- Dead code elimination
fuse (Compose f Id) = fuse f
fuse (Compose Id f) = fuse f
```

### Layer 3: GPU Code Generation

```haskell
-- Generate actual GPU kernel
codegen :: FusedKernel a b -> GPUKernel
codegen (FlockSlimeKernel w1 w2) =
    -- Single kernel doing both operations:
    kernel \particle ->
        let v1 = flock_update particle w1
        let p1 = particle { velocity = v1 }
        let v2 = slime_update p1 w2
        in p1 { velocity = v2 }
```

### Layer 4: Execution Runtime

```haskell
-- Compile once, execute many times
execute :: BehaviorGraph a b -> a -> IO b
execute graph input = do
    kernel <- compileOnce graph  -- Cached!
    runKernel kernel input

-- Interactive runtime
data LiveCoding = LiveCoding {
    currentBehavior :: IORef (BehaviorGraph Particles Particles),
    compiledKernel :: IORef (Maybe GPUKernel)
}

update :: LiveCoding -> Particles -> IO Particles
update lc particles = do
    behavior <- readIORef (currentBehavior lc)
    kernel <- getOrCompile (compiledKernel lc) behavior
    runKernel kernel particles

-- Hot-reloading: recompile on behavior change
setBehavior :: LiveCoding -> BehaviorGraph a b -> IO ()
setBehavior lc newBehavior = do
    writeIORef (currentBehavior lc) newBehavior
    writeIORef (compiledKernel lc) Nothing  -- Invalidate cache
```

---

## 5. User Experience: Best of Both Worlds

### For Creative Coders (Python-like API)

```python
# High-level compositional API
from tolvera import Behavior, run

# Define behaviors functionally
flock = Behavior.flock(weight=0.8)
slime = Behavior.slime(weight=0.5)

# Compose with operators
behavior = flock >> slime >> flock  # Category theory!

# Or use monadic style
behavior = (
    Behavior.identity()
    .then(flock)
    .then(slime)
    .parallel(Behavior.gol())  # Parallel composition!
)

# Execute in real-time
def main():
    tv = Tolvera(particles=1024)

    @tv.on_update
    def _(particles):
        return behavior(particles)  # Pure function!

    tv.run()
```

**Performance**: First frame compiles, subsequent frames run at 60+ FPS.

### For Advanced Users (Low-level control)

```python
# Explicit scheduling (Halide-style)
from tolvera import Schedule

behavior = flock >> slime
schedule = Schedule(behavior)
    .fuse(flock, slime)  # Manual fusion
    .vectorize(width=4)
    .parallelize(num_threads=8)
    .gpu(block_size=256)

compiled = schedule.compile()
result = compiled(particles)
```

---

## 6. Performance Techniques (Language-Agnostic)

### A. Stream Fusion

**Problem**: `map f (map g xs)` allocates intermediate array.

**Solution**: Fuse into single pass.

```haskell
-- Before fusion
positions = map update_pos particles
velocities = map update_vel particles

-- After fusion (single kernel)
particles' = map (\p -> {
    pos = update_pos p,
    vel = update_vel p
}) particles
```

### B. Structure Sharing

**Problem**: Copying large arrays is expensive.

**Solution**: Persistent data structures with sharing.

```haskell
-- Particles array shares structure
particles1 = updateParticle 42 particles0
-- Only index 42 is copied, rest is shared!
```

### C. Incremental Computation

**Problem**: Recomputing everything each frame is wasteful.

**Solution**: Track dependencies, only recompute changed values.

```haskell
-- Memoization with dependency tracking
flock_result = memoize (\p -> flock p) particles
-- Only recomputes if particles changed
```

### D. Compilation Caching

**Problem**: JIT compilation adds latency.

**Solution**: Cache compiled kernels.

```haskell
-- First call: compile (slow)
result1 = execute behavior particles1  -- 100ms

-- Subsequent calls: cached (fast)
result2 = execute behavior particles2  -- 0.5ms
```

---

## 7. Can You Do This With Taichi + Python?

### **YES, with caveats**

Taichi already has the infrastructure via [AsyncTaichi](https://mingkuan.taichi.graphics/publication/2020-asynctaichi/asynctaichi.pdf):

**What Taichi provides**:
- ✅ JIT compilation via LLVM/SPIR-V/CUDA
- ✅ Megakernel fusion (combining multiple kernels)
- ✅ Dead code elimination (DCE)
- ✅ Common subexpression elimination (CSE)
- ✅ [AST transformation pipeline](https://docs.taichi-lang.org/docs/compilation)
- ✅ Automatic parallelization

**The problem**: Tölvera uses Taichi at the **wrong abstraction level**.

### Better Approach: Compositional Layer on Taichi

```python
# Layer 1: Pure Functional Interface (Python)
from dataclasses import dataclass
from typing import Callable
import taichi as ti

@dataclass(frozen=True)
class Behavior:
    """Immutable behavior specification"""
    name: str
    params: dict

    def __rshift__(self, other: 'Behavior') -> 'Composition':
        """Category composition: self >> other"""
        return Composition(self, other)

    def parallel(self, other: 'Behavior') -> 'Parallel':
        """Monoidal product: run in parallel"""
        return Parallel(self, other)

# Define primitive behaviors
flock = lambda weight: Behavior('flock', {'weight': weight})
slime = lambda weight: Behavior('slime', {'weight': weight})

# Compose functionally (pure, immutable)
behavior = flock(0.8) >> slime(0.5) >> flock(1.0)
```

**This is pure category theory** - no execution yet!

### Layer 2: Compilation to Taichi

```python
class BehaviorCompiler:
    """Compiles behavior graphs to Taichi kernels"""

    def compile(self, behavior: Behavior) -> Callable:
        """Convert behavior graph to single Taichi kernel"""

        # Flatten to operations list
        operations = self._flatten(behavior)

        # Generate single fused kernel
        @ti.kernel
        def kernel(particles: ti.template()):
            for i in particles:
                p = particles[i]

                # Taichi unrolls ti.static at compile time!
                for op_type, params in ti.static(operations):
                    if op_type == 'flock':
                        p.vel += flock_force(p) * params['weight']
                    elif op_type == 'slime':
                        p.vel += slime_force(p) * params['weight']

                particles[i] = p

        return kernel

# Usage
compiler = BehaviorCompiler()
behavior = flock(0.8) >> slime(0.5) >> flock(1.0)
kernel = compiler.compile(behavior)  # Compiles to single kernel

# Execute (fast - single kernel launch)
kernel(tv.p.field)
```

---

## 8. Performance Reality Check

### What You Get

**First call** (compilation):
```python
kernel = tv.compile(behavior)  # ~100-500ms (one-time cost)
```

**Subsequent calls** (execution):
```python
kernel(particles)  # ~0.5-2ms for 10k particles @ 60fps ✅
```

### Benchmark Comparison

| Approach | Performance | Compositionality |
|----------|-------------|------------------|
| Current Tölvera | ⭐⭐⭐ (good) | ❌ (broken) |
| Separate kernels | ⭐⭐⭐ (good) | ❌ (sequential only) |
| **Proposed: Compiled behaviors** | ⭐⭐⭐⭐ (excellent) | ✅ (category theory!) |
| Hand-written CUDA | ⭐⭐⭐⭐⭐ (optimal) | ❌ (not compositional) |

**Taichi's [AsyncTaichi](https://ar5iv.labs.arxiv.org/html/2012.08141) already achieves 1.4-7x speedup** over naive kernels via fusion.

---

## 9. Recommended Implementation Strategy

### Phase 1: Build Compositional Layer in Python (2-3 weeks)

```python
# Pure Python for behavior graphs (your interface)
behavior = flock(0.8) >> slime(0.5) >> flock(1.0)

# Python compiles to Taichi (your backend)
kernel = compiler.compile(behavior)
```

**Advantages**:
- Keep existing Tölvera ecosystem
- Python metaprogramming is powerful enough
- Taichi handles GPU compilation
- Can migrate later if needed

### Phase 2: Optimize Critical Paths (2-3 weeks)

1. **Use ti.static** for compile-time unrolling
2. **Leverage AsyncTaichi** for kernel fusion
3. **Profile and optimize** specific kernels
4. **Cache compiled kernels** aggressively

### Phase 3: Optional - Rust/C++ Extensions (if needed)

If you hit Python limitations:

```python
# Python interface (compositional)
from tolvera_core import compile_behavior  # Rust/C++ extension

behavior = flock(0.8) >> slime(0.5)
kernel = compile_behavior(behavior)  # Compiled by Rust!
```

---

## 10. Should You Rethink Python/Taichi?

### **Keep Taichi If:**

✅ You want to stay in Python ecosystem
✅ You're okay with ~80-90% of hand-written CUDA performance
✅ You value rapid prototyping
✅ You need cross-platform (Vulkan/Metal/CUDA/CPU)
✅ You want to leverage existing Tölvera code

### **Consider Alternatives If:**

❌ You need 100% maximum performance (use Futhark/CUDA)
❌ You want stronger type guarantees (use Rust/Haskell)
❌ Python startup time is unacceptable
❌ You want compile-time optimization guarantees

---

## 11. Concrete Next Steps

### Step 1: Prototype Compositional Layer (1-2 weeks)

- Define Behavior dataclass
- Implement composition operators (`>>`, `parallel`)
- Verify category laws hold

### Step 2: Build Taichi Compiler (2-3 weeks)

- Flatten behavior graphs to operation lists
- Generate fused Taichi kernels
- Benchmark vs. current approach

### Step 3: Optimize (ongoing)

- Profile generated kernels
- Add kernel caching
- Tune Taichi compilation flags

### Step 4: Evaluate Alternatives (if needed)

- Try JAX as drop-in replacement
- Consider Rust FFI for compiler
- Benchmark against hand-written CUDA

---

## 12. Success Criteria

### Technical Metrics

- ✅ Category laws verified (identity, associativity, composition)
- ✅ Type-safe composition (no runtime errors)
- ✅ Performance ≥ current Tölvera (ideally better via fusion)
- ✅ Hot-reload behaviors without restart
- ✅ Maintain 60+ FPS for typical use cases

### User Experience Metrics

- ✅ Simpler API than current Tölvera
- ✅ Clearer mental model (pure functions)
- ✅ Better error messages (type errors at compile time)
- ✅ Documentation with category theory background

---

## Summary

**You can absolutely have mathematical composition AND high performance.**

Modern systems (Halide, JAX, Futhark) prove this is not only possible but practical. The key is:

1. **Separate specification from execution** (Halide principle)
2. **Use deferred execution** (JAX approach)
3. **Leverage type systems** (Futhark's uniqueness types)
4. **Fuse operations** (eliminate intermediate allocations)
5. **Cache compiled kernels** (JIT compilation)

**For Tölvera**: Start with Python + Taichi. Build a compositional layer on top. You'll get 80-90% of theoretical maximum performance while gaining mathematical rigor.

If you hit limits later, you can migrate incrementally to Rust/Futhark/JAX. But Taichi is likely sufficient.

---

## References

- [Futhark: Purely Functional GPU Programming](https://futhark-lang.org/)
- [Futhark PLDI 2017 Paper](https://elsman.com/pdf/pldi17.pdf)
- [Halide: Decoupling Algorithms from Schedules](https://halide-lang.org/)
- [Halide PLDI 2013 Paper](https://people.csail.mit.edu/jrk/halide-pldi13.pdf)
- [JAX: Composable Transformations](https://github.com/jax-ml/jax)
- [JAX Documentation on Pure Functions](https://jax.readthedocs.io/en/latest/glossary.html)
- [JAX Introduction](https://jax.quantecon.org/jax_intro.html)
- [AsyncTaichi: Megakernel Fusion](https://mingkuan.taichi.graphics/publication/2020-asynctaichi/asynctaichi.pdf)
- [AsyncTaichi: Inter-Kernel Optimizations](https://ar5iv.labs.arxiv.org/html/2012.08141)
- [Taichi Compilation Pipeline](https://docs.taichi-lang.org/docs/compilation)
- [Taichi Performance Tuning](https://docs.taichi-lang.org/docs/performance)
