# Battery Physics in Dendrite

Overview of the electrochemical physics implemented in Dendrite kernels.

## Fick's Diffusion

The fundamental equation for mass transport:

```
dc/dt = D * nabla^2(c)
```

### 2D Cartesian

```
dc/dt = D * (d2c/dx2 + d2c/dy2)
```

Discretized with central differences:

```
c_new = c + rx*(c_left - 2*c + c_right) + ry*(c_top - 2*c + c_bottom)

where rx = D*dt/dx^2, ry = D*dt/dy^2
```

CFL stability: `rx + ry <= 0.5`

### Spherical (for Particles)

For radially symmetric diffusion in a sphere:

```
dc/dt = (1/r^2) * d/dr(D * r^2 * dc/dr)
```

Discretized with special handling at r=0 (L'Hopital's rule) and surface (flux BC).

CFL stability: `D*dt/dr^2 <= 1/6`

## Butler-Volmer Kinetics

Relates current density to overpotential at an electrode:

```
j = i0 * [exp(alpha*F*eta/RT) - exp(-(1-alpha)*F*eta/RT)]
```

Parameters:
- `j`: Current density [A/m^2]
- `i0`: Exchange current density [A/m^2]
- `eta`: Overpotential (V - V_eq) [V]
- `alpha`: Transfer coefficient (typically 0.5)
- `F`: Faraday constant (96485 C/mol)
- `R`: Gas constant (8.314 J/mol/K)
- `T`: Temperature [K]

### Symmetric Case (alpha = 0.5)

When alpha = 0.5:
```
j = i0 * 2 * sinh(0.5*F*eta/RT)
```

Uses hyperbolic sine for better numerical stability.

### Linearized (Small eta)

For |eta| << RT/F (~26 mV at 298K):
```
j = i0 * F * eta / RT
```

Useful for EIS simulations.

## Thermal Transport

Heat equation with generation term:

```
rho*Cp*dT/dt = k*(d2T/dx2 + d2T/dy2) + Q
```

Parameters:
- `rho`: Density [kg/m^3]
- `Cp`: Heat capacity [J/kg/K]
- `k`: Thermal conductivity [W/m/K]
- `Q`: Heat generation [W/m^3]

Heat generation sources:
- Joule heating: `Q_j = I^2 * R`
- Reversible entropy: `Q_s = T * dU/dT * I`
- Reaction heat: `Q_r = I * eta`

## Single Particle Model (SPM)

Dendrite kernels can be combined to build an SPM:

1. Compute surface concentration from spherical diffusion
2. Calculate OCV from thermodynamic model
3. Compute overpotential: `eta = V - OCV - I*R_ohm`
4. Get current from Butler-Volmer
5. Update particle concentration with current as flux BC
6. Repeat

## Typical Parameter Values

### NMC Cathode
- D_s: 1e-14 to 1e-13 m^2/s
- R_p: 1-10 um
- c_max: 51000 mol/m^3
- i0: 1-50 A/m^2

### Graphite Anode
- D_s: 1e-14 to 1e-12 m^2/s
- R_p: 5-20 um
- c_max: 31000 mol/m^3
- i0: 10-100 A/m^2

### Electrolyte
- D: 1e-10 to 1e-9 m^2/s
- kappa: 0.1-1.0 S/m

## References

- Doyle, Fuller, Newman, "Modeling of Galvanostatic Charge and Discharge of the Lithium/Polymer/Insertion Cell" (1993)
- Newman & Thomas-Alyea, "Electrochemical Systems" (2004)
- Bard & Faulkner, "Electrochemical Methods" (2001)
