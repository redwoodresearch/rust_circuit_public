Rough priority order:

1. name of inner item (e.g., if you remove noop rearrange, inner item should be preserved). This also applies for module substitution etc. Note that this is important for correctness! Renaming irreducible nodes (symbols and arrays) is illegal in rewrites!
2. name of outer item which is 'equivalent' (e.g., if you do rewrite and overall circuits is 'equivalent' to input, rename to input name)
3. generated name using rewrite specific generation
4. auto-gen name
