import Mathlib
import CW.Poseidon.CauchyMatrix

/-!
# Cauchy Matrix Determinant

Every Cauchy matrix has nonzero determinant. This is the core technical
lemma used to show that Cauchy matrices are MDS.
-/

-- TODO: Add more comments about how univ automatically infers y

namespace MDS

noncomputable section

open Matrix Polynomial Lagrange Finset

section CauchyAux

variable {F : Type*} [Field F] {k : ℕ}
variable (C : CauchyMatrix F (k + 1))
variable (v : Fin (k + 1) → F)

/-! ### Step 1: Defining and relating auxiliary polynomials and matrices -/

/-- Auxiliary polynomial used in the determinant argument. -/
def Q : F[X] :=
  ∑ j : Fin (k + 1), Polynomial.C (v j) * nodal (univ.erase j) C.y

/-- Auxiliary rational sum used to rewrite evaluations of `Q`. -/
def P (i : Fin (k + 1)) : F :=
  ∑ j : Fin (k + 1), v j / (C.x i - C.y j)

/-- Reshaped evaluation identity for `Q` at row parameters. -/
lemma Q_eval_reshape :
    ∀ i : Fin (k + 1), (Q C v).eval (C.x i) =
      (∏ l ∈ univ, (C.x i - C.y l)) * (P C v i) := by
  sorry

/-- The `i`-th coordinate of `C.mulVec v` is exactly `P`. -/
lemma mulVec_eq_P :
    ∀ i : Fin (k + 1), (C.toMatrix.mulVec v) i = P C v i := by
  intro i
  simp only [mulVec, dotProduct, CauchyMatrix.toMatrix, of_apply, mul_comm, P, div_eq_mul_inv]

/-- If `C.mulVec v = 0`, then each coordinate of `P` is zero. -/
lemma P_eq_zero_of_mulVec_eq_zero (hCv : C.toMatrix.mulVec v = 0) :
    ∀ i : Fin (k + 1), P C v i = 0 := by
  intro i
  rw [← mulVec_eq_P (C := C) (v := v) i]
  simp [hCv]

/-- If `C.mulVec v = 0`, then all evaluations of `Q` at row parameters are zero. -/
lemma Q_eval_eq_zero_of_mulVec_eq_zero (hCv : C.toMatrix.mulVec v = 0) :
    ∀ i : Fin (k + 1), (Q C v).eval (C.x i) = 0 := by
  intro i
  have hp : P C v i = 0 := P_eq_zero_of_mulVec_eq_zero (C := C) (v := v) hCv i
  simp only [Q_eval_reshape (C := C) (v := v) i, hp, mul_zero]

/-! ### Step 2: Bounding the degree of `Q` -/

/-- Degree bound for each nodal factor in `Q`. -/
lemma nodal_natDegree_le : ∀ j : Fin (k + 1), (nodal (univ.erase j) C.y).natDegree ≤ k := by
  intro j
  simp only [natDegree_nodal, mem_univ, card_erase_of_mem, card_univ, Fintype.card_fin,
    add_tsub_cancel_right, le_refl]

/-- Degree bound for the auxiliary polynomial `Q`. -/
lemma Q_natDegree_le : (Q C v).natDegree ≤ k := by
  unfold Q
  -- `natDegree_sum_le` gives `deg(∑ fⱼ) ≤ fold max 0 (deg ∘ f) univ`.
  -- `.trans` then lets us chain it: instead of proving `deg(Q) ≤ k` directly,
  -- we now only need to show `fold max 0 (deg ∘ f) univ ≤ k`.
  -- _ _ are inferred by Lean: `Finset.univ` and `fun j => C (v j) * nodal (univ.erase j) C.y`
  apply (natDegree_sum_le _ _).trans
  rw [Finset.fold_max_le] -- Here we rewrite the `fold` into couple inequallities
  constructor
  · grind -- 0 <= k
  · -- ∀ j, deg(C (v j) * nodal (univ.erase j) C.y) ≤ k
    intro j hJ
    have hNodal_deg := nodal_natDegree_le (C := C) j
    simp only [Function.comp_apply] -- Evaluate the function at j
    by_cases hV : v j = 0
    · -- Zero scalar yields degree zero
      have h_zero : (Polynomial.C (v j) * nodal (univ.erase j) C.y).natDegree = 0 := by
        simp only [hV, map_zero, zero_mul, natDegree_zero]
      simp only [h_zero, zero_le]
    · -- Non-zero scalar doesn't change the degree
      rw [natDegree_C_mul hV]
      exact hNodal_deg

/-! ### Step 3: Showing `Q = 0` from many roots -/

/-- If `Q` vanishes at all row parameters, then `Q = 0`. -/
lemma Q_eq_zero_of_eval_eq_zero
    (hQ_eval : ∀ i : Fin (k + 1), (Q C v).eval (C.x i) = 0) :
    Q C v = 0 := by
  by_contra hQ  -- Assume the poly is not zero
  -- Q has at least k+1 roots (the k+1 distinct values C.x i are all roots of Q),
  -- hence `k + 1 ≤ natDegree Q`.
  have hQ_roots : k + 1 ≤ (Q C v).natDegree := by
    -- Each C.x i is a root of Q
    have h_root : ∀ i : Fin (k + 1), C.x i ∈ (Q C v).roots := by
      intro i
      exact (Polynomial.mem_roots hQ).mpr (hQ_eval i)
    have hQ_roots_card : k + 1 ≤ (Q C v).roots.card := by sorry
    exact le_trans hQ_roots_card (Polynomial.card_roots' (Q C v))
  -- But we already know that the degree of Q is <- k
  have hQ_natDegree : (Q C v).natDegree ≤ k := by
    exact Q_natDegree_le (C := C) (v := v)
  grind -- Contradiction: `k+1` ≤ `natDegree` ≤ `k`

/-! ### Step 4: Evaluating `Q` at column parameters -/

/-- Evaluation of `Q` at a column parameter. -/
lemma Q_eval_at_y (m : Fin (k + 1)) :
    (Q C v).eval (C.y m) = v m * ∏ l ∈ univ.erase m, (C.y m - C.y l) := by
  sorry

end CauchyAux

/-- The product `∏_{l ≠ m} (yₘ - yₗ)` is nonzero for Cauchy parameters. -/
lemma prod_sub_y_ne_zero {F : Type*} [Field F] {k : ℕ}
    (C : CauchyMatrix F (k + 1)) (m : Fin (k + 1)) :
    ∏ l ∈ Finset.univ.erase m, (C.y m - C.y l) ≠ 0 := by
  rw [Finset.prod_ne_zero_iff]
  grind [C.hy]

/-- ### Main: Non-zero determinant
     Every square submatrix of a Cauchy matrix has nonzero determinant. -/
-- TODO: CLEAN THIS PROOF UP
lemma CauchyMatrix.non_zero_determinant {F : Type*} [Field F] {t : ℕ}
    (C : CauchyMatrix F t) :
    (C.toMatrix).det ≠ 0 := by
  -- Base case: 0×0 determinant is 1
  cases t with
  | zero => simp [det_fin_zero]
  | succ k =>
    -- Lets prove it by contradiction
    by_contra hdet
    -- det = 0 implies ∃ nonzero v with C·v = 0
    rw [← Matrix.exists_mulVec_eq_zero_iff] at hdet
    obtain ⟨v, hv_ne, hCv⟩ := hdet
    apply hv_ne
    ext m -- We will show that m-th entry of v is zero for all m
    /- **Step 1**: Q(xᵢ) = 0 for all i
       Q(xᵢ) = ∑ⱼ vⱼ ∏_{l≠j}(xᵢ-yₗ) = ∏ₗ(xᵢ-yₗ) · ∑ⱼ vⱼ/(xᵢ-yⱼ) = 0 where last equality follows from
       the fact that C·v = 0 and the second factor is the i-th element of C·v  -/
    have hQ_eval : ∀ i : Fin (k + 1), (Q C v).eval (C.x i) = 0 :=
      Q_eval_eq_zero_of_mulVec_eq_zero (C := C) (v := v) hCv
    -- **Step 2** and **Step 3**: degree bound and many roots imply Q = 0
    have hQ_zero : Q C v = 0 := Q_eq_zero_of_eval_eq_zero (C := C) (v := v) hQ_eval
    -- **Step 4**: Q(yₘ) = v(m) · ∏_{l≠m}(yₘ - yₗ)
    -- All terms with j ≠ m vanish (product contains factor yₘ - yₘ = 0)
    have hQ_ym : (Q C v).eval (C.y m) = v m * ∏ l ∈ univ.erase m, (C.y m - C.y l) :=
      Q_eval_at_y (C := C) (v := v) (m := m)
    -- **Step 5**: Since Q = 0 but Q(yₘ) = v(m) · (nonzero), we get v(m) = 0
    have hContradict : v m = 0 := by
      simp only [hQ_zero, eval_zero] at hQ_ym
      have hprod_ne : ∏ l ∈ Finset.univ.erase m, (C.y m - C.y l) ≠ 0 :=
        prod_sub_y_ne_zero (C := C) (m := m)
      exact Or.resolve_right (mul_eq_zero.mp hQ_ym.symm) hprod_ne
    exact hContradict

end

end MDS
