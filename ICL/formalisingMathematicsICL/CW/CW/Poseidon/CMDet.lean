import Mathlib
import CW.Poseidon.CauchyMatrix

/-!
# Cauchy Matrix Determinant

Every Cauchy matrix has nonzero determinant. This is the core technical
lemma used to show that Cauchy matrices are MDS.

## Proof outline

We do a proof by contradiction.

Suppose `det(C) = 0`. Then there exists a nonzero vector `v` such that
`C · v = 0`. We show that `v = 0`, which gives a contradiction.

**Step 1 — Auxiliary polynomials**
Define the polynomial `Q(X) = ∑ⱼ vⱼ · ∏_{l ≠ j} (X - yₗ)`
and the rational sum `P(i) = ∑ⱼ vⱼ / (xᵢ - yⱼ)` (the i-th entry of `C·v`).
Prove the identity `Q(xᵢ) = (∏ₗ (xᵢ - yₗ)) · P(i)`, which relates
evaluations of `Q` at the row parameters to the matrix–vector product.

**Step 2 — Degree bound**
Each summand of `Q` has degree at most `k`, so `deg(Q) ≤ k`.

**Step 3 — `Q = 0`**
Since `C · v = 0`, every `P(i) = 0`, hence `Q(xᵢ) = 0` for all
`i ∈ {0, …, k}`. That gives `k + 1` distinct roots (the `xᵢ` are
pairwise distinct by the Cauchy matrix conditions). But a nonzero
polynomial of degree ≤ `k` can have at most `k` roots, so `Q = 0`.

**Step 4 — Evaluate `Q` at column parameters**
Evaluating the (now zero) polynomial at a column parameter `yₘ` gives
`0 = Q(yₘ) = vₘ · ∏_{l ≠ m} (yₘ - yₗ)`.
The product is nonzero because the `yⱼ` are pairwise distinct.

**Step 5 — Contradiction**
Since the product is nonzero, `vₘ = 0` for every `m`. This contradicts
`v ≠ 0`, so `det(C) ≠ 0`.
-/

namespace MDS

noncomputable section

open Matrix Polynomial Lagrange Finset

section CauchyAux

variable {F : Type*} [Field F] {k : ℕ}
variable (C : CauchyMatrix F (k + 1))
variable (v : Fin (k + 1) → F)

/-! ### Step 1: Auxiliary polynomials -/

/-- Auxiliary polynomial used in the determinant argument.
    `Q(X) = ∑ⱼ vⱼ · ∏_{l ≠ j} (X - yₗ)` -/
def Q : F[X] :=
  ∑ j : Fin (k + 1), Polynomial.C (v j) * nodal (univ.erase j) C.y

/-- Auxiliary rational sum used to rewrite evaluations of`Q`.
    `P(i) = ∑ⱼ vⱼ / (xᵢ - yⱼ)` -/
def P (i : Fin (k + 1)) : F :=
  ∑ j : Fin (k + 1), v j / (C.x i - C.y j)

/-- Reshaped evaluation identity for `Q` at row parameters.
    `Q(xᵢ) = (∏ₗ (xᵢ - yₗ)) · P(i)` -/
lemma Q_eval_reshape :
    ∀ i : Fin (k + 1), (Q C v).eval (C.x i) =
      (∏ l ∈ univ, (C.x i - C.y l)) * (P C v i) := by
  intro i
  -- First, let's evalaute the polynomial to get the algebraic expression.
  simp only [Q, P, eval_finset_sum, eval_mul, eval_C, eval_nodal]
  rw [Finset.mul_sum] -- Get the product inside the sum
  apply Finset.sum_congr rfl -- Sum is equal if each summand is equal
  intro j _
  -- Lets rewrite the second multiplicatiopns so both are over univ.erase j
  rw [← Finset.mul_prod_erase Finset.univ (fun l => C.x i - C.y l) (Finset.mem_univ j)]
  -- Now both sides have the same terms.
  -- Now the result should just consist of arithmetic manipulations
  have hne : C.x i - C.y j ≠ 0 := by exact sub_ne_zero.mpr (C.hxy i j)
  field_simp

/-- The `i`-th coordinate of `C.mulVec v` is exactly `P`. -/
lemma mulVec_eq_P :
    ∀ i : Fin (k + 1), (C.toMatrix.mulVec v) i = P C v i := by
  intro i
  -- Simplify all of the vector multiplications and dot products to get a simple arithmetic expr.
  simp only [CauchyMatrix.toMatrix, of_apply, P, mulVec, dotProduct]
  grind

/-- If `C.mulVec v = 0`, then all evaluations of `Q` at row parameters are zero. -/
lemma Q_eval_eq_zero_of_mulVec_eq_zero (hCv : C.toMatrix.mulVec v = 0) :
    ∀ i : Fin (k + 1), (Q C v).eval (C.x i) = 0 := by
  intro i
  have hp : P C v i = 0 := by
    rw [← mulVec_eq_P (C := C) (v := v) i]
    simp only [hCv, Pi.zero_apply]
  simp only [Q_eval_reshape (C := C) (v := v) i, hp, mul_zero]

/-! ### Step 2: Degree bound -/

/-- Degree bound for the auxiliary polynomial `Q`. -/
lemma Q_natDegree_le : (Q C v).natDegree ≤ k := by
  unfold Q
  -- `natDegree_sum_le` says deg of a sum of polynomials ≤ max of their individual degrees.
  -- `.trans` chains this so instead of proving `deg(Q) ≤ k` directly,
  -- we only need to show each summand has degree ≤ k.
  -- _ _ are inferred by Lean: `Finset.univ` and `fun j => C (v j) * nodal (univ.erase j) C.y`
  apply (natDegree_sum_le _ _).trans
  rw [Finset.fold_max_le] -- Here we rewrite the `fold` into couple inequallities
  constructor
  · grind -- 0 <= k
  · -- ∀ j, deg(C (v j) * nodal (univ.erase j) C.y) ≤ k
    intro j hJ
    simp only [Function.comp_apply] -- Evaluate the function at j
    by_cases hV : v j = 0
    · -- Zero scalar yields degree zero
      simp only [hV, map_zero, zero_mul, natDegree_zero, zero_le]
    · -- Non-zero scalar doesn't change the degree
      rw [natDegree_C_mul hV]
      -- Now we need to prove that the degree of the nodal is <= k
      simp only [natDegree_nodal, mem_univ, card_erase_of_mem, card_univ, Fintype.card_fin,
        add_tsub_cancel_right, le_refl]

/-! ### Step 3: `Q=0` -/

/-- If `Q` vanishes at all row parameters, then `Q = 0`. -/
lemma Q_eq_zero_of_eval_eq_zero
    (hQ_eval : ∀ i : Fin (k + 1), (Q C v).eval (C.x i) = 0) :
    Q C v = 0 := by
  classical
  by_contra hQ  -- Assume the poly is not zero
  -- Q has at least k+1 roots (the k+1 distinct values C.x i are all roots of Q),
  -- hence `k + 1 ≤ natDegree Q`.
  have h_root : ∀ i : Fin (k + 1), C.x i ∈ (Q C v).roots := by
    intro i
    exact (Polynomial.mem_roots hQ).mpr (hQ_eval i)
  -- C.x is a subset of roots of Q
  have h_sub : Finset.univ.image C.x ⊆ (Q C v).roots.toFinset := by
    intro a ha
    grind
  /- Chain of two inequalities via `.trans`:
     `(univ.image C.x).card ≤ (Q C v).roots.toFinset.card ≤ Q C v).roots.card`
     First one we get from inclusion
     Second one is just cosmetic transition from Fisnet to Multiset -/
  have h_natDegree_X_le: (Finset.univ.image C.x).card <= (Q C v).roots.card := by
    exact (Finset.card_le_card h_sub).trans (Multiset.toFinset_card_le (Q C v).roots)
  -- Cardinality of C.x = k + 1
  have h_natDegree_X_eq: k + 1 = (Finset.univ.image C.x).card  := by
    rw [Finset.card_image_of_injective univ C.hx]
    simp only [card_univ, Fintype.card_fin]
  -- A polynomial has at most natDegree roots (counting multiplicity)
  have h_roots_le_deg : (Q C v).roots.card ≤ (Q C v).natDegree := by
    exact Polynomial.card_roots' (Q C v)
  -- Combine the above 3 hypothesis
  have hQ_natDegree_1 : k + 1 ≤ (Q C v).natDegree := by
    rw [h_natDegree_X_eq]
    exact h_natDegree_X_le.trans h_roots_le_deg
  -- But we already know that the degree of Q is <= k
  have hQ_natDegree_2 : (Q C v).natDegree ≤ k := by
    exact Q_natDegree_le (C := C) (v := v)
  grind -- Contradiction: `k+1` ≤ `natDegree` ≤ `k`. Uses both `hQ_natDegree_1` and `hQ_natDegree_2`

/-! ### Step 4: Evaluate `Q` at column parameters -/

/-- Evaluation of `Q` at a column parameter. -/
lemma Q_eval_at_y (m : Fin (k + 1)) :
    (Q C v).eval (C.y m) = v m * ∏ l ∈ univ.erase m, (C.y m - C.y l) := by
  -- First expand the evaluation
  simp only [Q, eval_finset_sum, eval_mul, eval_C, eval_nodal]
  /- We get the sum  `∑ⱼ vⱼ · ∏_{l ≠ j} (yₘ - yₗ)` that collapses to the m-th term,
     because for j ≠ m the factor `(yₘ - yₘ) = 0` appears in the product. -/
  -- `f = fun j => v j * ∏ l ∈ univ.erase j, (C.y m - C.y l)` is infered by lean
  apply Finset.sum_eq_single m
  · -- ∀n ≠ m, vₙ · ∏_{l ≠ n} (yₘ - yₗ) = 0
    intro n hn hnm
    rw [mul_eq_zero]
    right
    have h_m_in_erase_n: m ∈ univ.erase n := by
      simp only [mem_erase, mem_univ]
      grind
    apply prod_eq_zero h_m_in_erase_n (by grind) -- The grind proves that yₘ - yₘ = 0
  · grind -- The case when m is not in univ. But m cannot not be in univ for a Finset.

end CauchyAux

/-- ### Main: Non-zero determinant
     Every square submatrix of a Cauchy matrix has nonzero determinant. -/
lemma CauchyMatrix.non_zero_determinant {F : Type*} [Field F] {t : ℕ}
    (C : CauchyMatrix F t) :
    (C.toMatrix).det ≠ 0 := by
  cases t with
  -- Solve the degenerate case separately
  | zero => simp only [det_fin_zero, ne_eq, one_ne_zero, not_false_eq_true]
  | succ k =>
    -- Lets prove it by contradiction
    by_contra hdet
    -- det = 0 implies ∃ nonzero v with C·v = 0
    rw [← Matrix.exists_mulVec_eq_zero_iff] at hdet
    obtain ⟨v, hv_ne, hCv⟩ := hdet
    apply hv_ne
    ext m -- We will show that m-th entry of v is zero for all m
    -- Lets create local hypothesis from all prev steps
    have hQ_eval : ∀ i : Fin (k + 1), (Q C v).eval (C.x i) = 0 := by
      exact Q_eval_eq_zero_of_mulVec_eq_zero (C := C) (v := v) hCv
    have hQ_zero : Q C v = 0 := by
      exact Q_eq_zero_of_eval_eq_zero (C := C) (v := v) hQ_eval
    have hQ_ym : (Q C v).eval (C.y m) = v m * ∏ l ∈ univ.erase m, (C.y m - C.y l) := by
      exact Q_eval_at_y (C := C) (v := v) (m := m)
    -- **Step 5**: Contradiction
    have hprod_ne : ∏ l ∈ Finset.univ.erase m, (C.y m - C.y l) ≠ 0 := by
      -- Each term of the product is non-zero
      rw [Finset.prod_ne_zero_iff]
      grind [C.hy]
    have hContradict : v m = 0 := by
      simp only [hQ_zero, eval_zero] at hQ_ym
      -- Prod. equals zero and one side is non-zero from hprod_ne implies the other is 0
      exact Or.resolve_right (mul_eq_zero.mp hQ_ym.symm) hprod_ne
    exact hContradict

end

end MDS
