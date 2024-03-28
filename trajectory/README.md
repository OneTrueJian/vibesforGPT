Trajectory
===

## Indices and Labels

- atom labels: Capital letters starting from $`I`$: $`I, J, K, L, ...`$

- Coordinate labels: small letters starting form $`a`$: $`a, b, c, d, ...`$
  
  - instead of the greek letters $`\alpha, \beta, \gamma, \delta, ...`$

- Time label: just `time`

## Examples

| Observable                                           | dimensions                   |
| ---------------------------------------------------- | ---------------------------- |
| temperature, pressure, energy                        | [`time`]                     |
| stress                                               | [`time`, `a`, `b`]           |
| positions, velocities, forces, heat flux             | [`time`, `I`, `a`]           |
| stresses                                             | [`time`, `I`, `a`, `b`]      |
| heat flux autocorrelation function, cumulative kappa | [`time`, `I`, `J`, `a`, `b`] |
