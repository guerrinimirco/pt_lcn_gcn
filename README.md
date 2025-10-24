# Finite-T Hadron-Quark Phase Transition Tables

This repository provides tabulated results for a **finite-temperature hadron-quark phase-transition framework** that continuously interpolates between local charge neutrality and global neutrality constructions via a mixing parameter **η** (η = 1 → fully local, η = 0 → fully global).  
The data have been produced to be used in the manuscript https://doi.org/10.1103/8l3m-tdlc arXiv:2506.20418.

---

- `Tables/CC_Construction_output_plots.dat`  
 A whitespace-separated table with no header. Each row corresponds to a point in the grid
$(\eta, Y_e, T\,[\mathrm{MeV}], n_B\,[\mathrm{fm}^{-3}])$.

**Grid sampling**

- **Temperatures:** $T=\{0.1,10,20,30,40,50,60\}\,\mathrm{MeV}$
- **Electron fractions:** $Y_e \in \{0.1,0.25,0.4\}$
- **Mixing parameter:** $\eta \in \{0,0.1,0.3,0.6,1\}$
- **Baryon density grid (uniform):**

  $$n_B \in [n_{B,i}, n_{B,f}] = [0.1\,n_0, 12\,n_0], \qquad
  N_{n_B}=300, \qquad
  \Delta n_B = \frac{n_{B,f}-n_{B,i}}{N_{n_B}-1}$$

  where $n_0 \simeq 0.16\,\mathrm{fm}^{-3}$; the range corresponds to $\sim 0.016$–$1.92\,\mathrm{fm}^{-3}$.


  
**Columns (1–16):**
1. `eta` — mixing parameter η (η=1 fully local charge neutrality, η=0 fully global charge neutrality)  
2. `Ye` — electron fraction \(Y_e\)  
3. `T_MeV` — temperature \(T\) in MeV  
4. `nB_fm3` — baryon number density \(n_B\) in fm\(^{-3}\)  
5. `1-chi` — quark volume fraction \( 1- \chi \) (hadron fraction is \(\chi\))  
6. `Yp` — proton net fraction  
7. `Yn` — neutron net fraction  
8. `Yu` — up-quark net fraction  
9. `Yd` — down-quark net fraction  
10. `Ys` — strange-quark net fraction  
11. `e_MeVfm3` — energy density \(varepsilon\) in MeV fm\(^{-3}\)  
12. `P_MeVfm3` — pressure \(P\) in MeV fm\(^{-3}\)  
13. `s_per_baryon` — specific entropy per baryon \(s/n_B\) 
14. `cv_per_baryon` — \(c_v/n_B\) (per baryon)  
15. `cp_per_baryon` — \(c_p/n_B\) (per baryon)  
16. `ca2` — adiabatic sound speed squared \(c_a^2\)

_Units:_ natural units with \(\hbar=c=k_B=1\); 

---
