import Mathlib
import Mathlib.LinearAlgebra.Lagrange

open scoped BigOperators
open Polynomial

namespace Lagrange

variable {K : Type*} [Field K]

/-!
# Lagrange Interpolation (existence)
-/

-- =========================
-- Polynomial Helpers
-- =========================
lemma eval_poly_sum_eq_eval_sum_poly
    {K : Type*} [Semiring K]
    {ι : Type*}
    (s : Finset ι)
    (f : ι → K[X])
    (x : K) :
    (∑ i ∈ s, f i).eval x = ∑ i ∈ s, (f i).eval x := by
  classical
  -- Let's try by induction on the finset s
  refine Finset.induction_on s ?h0 ?hstep
  · -- Base case: s = ∅
    simp only [Finset.sum_empty, eval_zero]
  · -- Inductive step
    intro a s ha IH
    simp only [Finset.sum_insert ha]
    simp only [eval_add, IH]

-- ========================
-- Lagrange Basis Polynomial Helpers
-- =======================

lemma lagrange_basis_kronecker_delta
    {K : Type*} [Field K]
    {ι : Type*} [DecidableEq ι]
    {i j : ι}
    (s : Finset ι)
    (v : ι → K)
    (hv : Set.InjOn v (s : Set ι))
    (hj : j ∈ s) :
    (Lagrange.basis s v i).eval (v j) = if i = j then 1 else 0 := by
  classical
  have formula_unwrap :
    (Lagrange.basis s v i).eval (v j)
      =
    ∏ x ∈  s.erase i, (v i - v x)⁻¹ * (v j - v x) := by
      simp [Lagrange.basis, basisDivisor, eval_prod,
          eval_mul, eval_C, eval_sub, eval_X]
  by_cases h : (i = j)
  · -- Case i = j
    rw [formula_unwrap] -- Unwrap the formula
    simp only [reduceIte, h] -- Simplify the if statement
    refine Finset.prod_eq_one ?_
    intro x hx
    have hVjx : (v j - v x) ≠ 0 := by
      refine sub_ne_zero.mpr ?_
      unfold Set.InjOn at hv
      intro hVjxEq0
      apply hv at hVjxEq0 -- Introduces 2 more goals: j ∈ s and x ∈ s
      · -- Show False (main goal)
        rw [hVjxEq0] at hx
        rw [Finset.mem_erase] at hx
        obtain hFalse := hx.1
        contradiction
      · -- Show that j ∈ s
        exact hj
      · -- Show that x ∈ s
        simp only [SetLike.mem_coe]
        apply Finset.mem_of_mem_erase at hx
        exact hx
    -- Simplify the product to 1. jVjx necessary for inv_mul_cancel₀
    simp only [ne_eq, hVjx, not_false_eq_true, inv_mul_cancel₀]
  · -- Case i ≠ j
    rw [formula_unwrap] -- Unwrap the formula
    simp only [reduceIte, h] -- Simplify the if statement
    refine Finset.prod_eq_zero (M₀ := K) (i := j) ?_ ?_
    · -- Show j ∈ s.erase i
      apply Finset.mem_erase.mpr
      constructor
      · intro hJeqI
        apply h
        exact hJeqI.symm
      · exact hj
    · -- Show that the factor at j is zero
      simp only [sub_self, mul_zero]

lemma lagrange_basis_degree_lt
    {K : Type*} [Field K]
    {ι : Type*} [DecidableEq ι]
    (s : Finset ι) -- Index set
    (v : ι → K)
    (hv : Set.InjOn v (s : Set ι))
    (hs : s.Nonempty)
    (i : ι)
    (hi : i ∈ s) :
    (Lagrange.basis s v i).natDegree < s.card := by
  classical
  simp only [Lagrange.basis]
  rw [natDegree_prod']
  · have hDeg: ∀ j ∈ s.erase i, (basisDivisor (v i) ( v j)).natDegree = 1 := by
      intro j hj
      simp only [basisDivisor]
      rw [natDegree_mul]
      rw [natDegree_X_sub_C]
      rw [natDegree_C]
      · simp only [ne_eq, map_eq_zero, inv_eq_zero]
        · rw [sub_eq_zero]
          intro hVij
          apply hv at hVij
          rw [hVij] at hj
          rw [Finset.mem_erase] at hj
          obtain hFalse := hj.1
          contradiction
          · simpa only [SetLike.mem_coe]
          · simp only [SetLike.mem_coe]
            rw [Finset.mem_erase] at hj
            exact hj.2
      · exact X_sub_C_ne_zero (v j)
    rw [Finset.card_eq_sum_ones]
    rw [Finset.sum_congr rfl hDeg]
    simp only [Finset.sum_const, smul_eq_mul, mul_one, gt_iff_lt]
    rw [Finset.card_erase_of_mem hi]
    simp only [tsub_lt_self_iff, Finset.card_pos, zero_lt_one, and_true]
    exact hs
  · rw [Finset.prod_ne_zero_iff]
    intro x hx
    rw [leadingCoeff_ne_zero]
    unfold basisDivisor
    have hXnotJ : x ≠ i := by
      refine Finset.ne_of_mem_erase hx
    have hVxnotVj : v x ≠ v i := by
      intro hVxj
      apply hv at hVxj
      contradiction
      simp only [SetLike.mem_coe]
      apply Finset.mem_of_mem_erase at hx
      exact hx
      simp only [SetLike.mem_coe]
      exact hi
      -- Re-do this step. I don't fully understand why it became 3 goals all of the sudden.
    have hVxSubVjNot0 : (v i - v x) ≠ 0 := by
      intro hVxSubVj0
      simp only [sub_eq_zero] at hVxSubVj0
      apply hVxnotVj
      exact hVxSubVj0.symm
    simp only [ne_eq, mul_eq_zero, map_eq_zero, inv_eq_zero, not_or]
    constructor
    · convert hVxSubVjNot0
    · exact X_sub_C_ne_zero (v x)


-- =========================
-- Main Theorem: Existence of Lagrange Interpolating Polynomial
-- =========================

theorem existence_of_lagrange_interpolating_polynomial
    {K : Type*} [Field K]
    (s : Finset ι) -- Index set
    (hs : s.Nonempty)
    (v : ι → K)
    (f : ι → K)
    (hv : Set.InjOn v (s : Set ι)) :
    ∃ p : K[X], p.natDegree < s.card ∧ ∀ i ∈ s, p.eval (v i) = f i := by
  classical
  let p : K[X] := ∑ i ∈ s, (f i) • (Lagrange.basis s v i)
  use p
  constructor
  · -- Prove that degree p < s.car
    -- Unwrap the sum definition
    unfold p
    -- Cases on s being empty or nonempty
    have hDegs : ∀ i ∈ s, (f i • Lagrange.basis s v i).natDegree < s.card := by
      intro i hi
      have hBasisDeg := lagrange_basis_degree_lt s v hv hs i hi
      have hNatDeg :
          (f i • Lagrange.basis s v i).natDegree ≤ (Lagrange.basis s v i).natDegree := by
        simpa using (natDegree_smul_le (a := f i) (p := Lagrange.basis s v i))
      grind only -- Remove the grind if possible

    have hle : (∑ i ∈ s, f i • Lagrange.basis s v i).natDegree ≤ s.sup (fun i => (f i • Lagrange.basis s v i).natDegree) := by
      simpa using (natDegree_sum_le (s := s) (f := fun i => f i • Lagrange.basis s v i))

    refine lt_of_le_of_lt hle ?_
    rw [Finset.sup_lt_iff]
    exact hDegs
    simp only [Nat.bot_eq_zero, Finset.card_pos]
    exact hs

  · -- Prove that p touches points (v i, f i) for i ∈ s
    intro i hi
    -- Prove that p.eval (v i) = sum (f j • Lagrange.basis s v j).eval (v i)
    simp only [p]
    rw [eval_poly_sum_eq_eval_sum_poly]
    simp only [eval_smul, smul_eq_mul] -- Gets the scalar multiplication out of the eval
    -- Now use the kronecker delta property
    rw [Finset.sum_eq_single i] -- SAVIOR!!!!!
    · rw [lagrange_basis_kronecker_delta s v hv hi]
      simp only [↓reduceIte, mul_one]
    · intro b hb hBNeqi
      rw [lagrange_basis_kronecker_delta s v hv hi]
      simp only [mul_ite, mul_one, mul_zero, ite_eq_right_iff]
      intro hBEqi
      contradiction
    intro hi'
    contradiction

end Lagrange
