# The Dao of Circuits

1. Computations should be represented as circuits. (aka, put everything possible in your circuit)
2. Manipulation functions should take circuits as inputs and return circuits
   (without themselves doing computation). Computation (rather than
   manipulation) should be explicit when possible (the evaluation function
   isn't called in many places).
3. Circuit manipulations should be done with extensional equality rewrites
   which take in and return circuits. If your target manipulation doesn't
   preserve extensional equality, then you
   should do as much as possible via equality rewrites before making small
   targeted changes which do break equality.
4. Corollary of above: there should be a large and flexible array of
   extensional equality rewrites which always preserve extensional equality.
   Then there should be a small number of well tested transformations which
   preserve some invariant, but don't preserve equality (batching, sampling,
   etc). And then users can sometimes do simple modifications which don't
   respect any such invariant (e.g. zeroing out).
5. Circuits don't require context. This implies that circuits can be freely
   serialized and de-serialized and operated on functionally more generally.
6. It should be possible to represent your computation in whatever layout is
   most convenient for manipulation, regardless of the naive computational or
   memory cost of that layout.
7. Circuits should have as few different node types as reasonably possible.
   This makes rewrites as general as possible
8. Circuits are trees and can be operated on as such, but operations are
   memoized whenever possible.
8. Getting, updating, and traversing circuit trees should be done in a unified
   way. (In practice, IterativeMatcher)

## Examples


For 2, see sampling which is a circuit to circuit operation.

For 7, we avoid having a 'stack' primitive in favor of unsqueeze (via
   rearrange) and then concat.
