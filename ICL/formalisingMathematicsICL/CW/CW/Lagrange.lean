import Mathlib
import Mathlib.LinearAlgebra.Lagrange

open scoped BigOperators
open Polynomial

namespace Lagrange

-- =========================
-- Helper Lemmas
-- =========================
lemma inj_diff_at_erase
    {K : Type*} [Field K]
    {ι : Type*} [DecidableEq ι]
    {s : Finset ι}
    {v : ι → K}
    (hv : Set.InjOn v (s : Set ι))
    {i : ι}
    {j : ι}
    (hi : i ∈ s)
    (hj : j ∈ s.erase i) :
    v i ≠ v j := by
    -- API to unfold Set.InjOn
    unfold Set.InjOn at hv
    intro hVij
    apply hv at hVij
    · -- Show False (main goal)
      -- Contradict i ∈ s.erase i
      rw [hVij] at hj
      rw [Finset.mem_erase] at hj
      obtain hFalse := hj.1
      contradiction
    · -- Show that i ∈ s
      exact hi
    · -- Show that j ∈ s
      simp only [SetLike.mem_coe]
      apply Finset.mem_of_mem_erase at hj
      exact hj

-- Couldn't find anythinng like that in Mathlib so I wrote it myself
lemma eval_poly_sum_eq_eval_sum_poly
    {K : Type*} [Semiring K]
    {ι : Type*}
    (s : Finset ι)
    (f : ι → K[X])
    (x : K) :
    (∑ i ∈ s, f i).eval x = ∑ i ∈ s, (f i).eval x := by
  classical
  -- Let's try by induction on the finset s
  refine Finset.induction_on s ?_ ?_
  · -- Base case: s = ∅
    simp only [Finset.sum_empty, eval_zero]
  · -- Inductive step
    intro a s ha IH
    simp only [Finset.sum_insert ha]
    simp only [eval_add, IH]

-- ================================
-- Lagrange Basis Polynomial Lemmas
-- ================================
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
      exact inj_diff_at_erase hv hj hx
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
  have vi_ne_vj (j : ι) (hj : j ∈ s.erase i) :  ¬C (v i - v j)⁻¹ = 0 := by
    simp only [map_eq_zero, inv_eq_zero]
    intro hVij
    rw [sub_eq_zero] at hVij
    have hContradict: v i ≠ v j := inj_diff_at_erase hv hi hj
    contradiction
  rw [natDegree_prod]
  · -- Show that the sum of degrees is less than s.card
    have hDeg: ∀ j ∈ s.erase i, (basisDivisor (v i) (v j)).natDegree = 1 := by
      -- Show that each factor has degree 1
      intro j hj
      unfold basisDivisor
      rw [natDegree_mul, natDegree_X_sub_C, natDegree_C]
      · -- Show that the inverse constant is not zero
        apply vi_ne_vj j hj
      · -- Show that the monomial is not zero
        exact X_sub_C_ne_zero (v j)
    rw [Finset.sum_congr rfl hDeg]
    -- Wrap the sum into cardinality of s.erase i
    simp only [Finset.sum_const, smul_eq_mul, mul_one, gt_iff_lt]
    rw [Finset.card_erase_of_mem hi]
    simp only [tsub_lt_self_iff, Finset.card_pos, zero_lt_one, and_true]
    exact hs
  · -- Show that each basisDivisor (v i) (v j) is not zero
    intro j hj
    unfold basisDivisor
    simp only [ne_eq, mul_eq_zero, not_or]
    constructor
    · -- Show that the inverse constant is not zero
      apply vi_ne_vj j hj
    · -- Show that the monomial is not zero
      exact X_sub_C_ne_zero (v j)

-- ============================================================
-- Main Theorem: Existence of Lagrange Interpolating Polynomial
-- ============================================================
theorem existence_of_lagrange_interpolating_polynomial
    {K : Type*} [Field K]
    (s : Finset ι) -- Index set
    (hs : s.Nonempty)
    (v : ι → K)
    (f : ι → K)
    (hv : Set.InjOn v (s : Set ι)) :
    ∃ p : K[X], p.natDegree < s.card ∧ ∀ i ∈ s, p.eval (v i) = f i := by
  classical
  -- We construct the interpolating polynomial p and prove its properties
  use (∑ i ∈ s, (f i) • (Lagrange.basis s v i))
  constructor
  · -- Prove that degree p < s.card
    -- Show that each summand has degree < s.card
    have hDegs : ∀ i ∈ s, (f i • Lagrange.basis s v i).natDegree < s.card := by
      intro i hi
      have hNatDeg :
          (f i • Lagrange.basis s v i).natDegree ≤ (Lagrange.basis s v i).natDegree := by
          -- <= because smul can lower degree if f i = 0
        apply (natDegree_smul_le (a := f i) (p := Lagrange.basis s v i))
      exact lt_of_le_of_lt hNatDeg (lagrange_basis_degree_lt s v hv hs i hi)
    -- Now show that the sum has degree < s.card by showing that its sup is < s.card
    refine lt_of_le_of_lt (b := s.sup (fun i => (f i • Lagrange.basis s v i).natDegree)) ?_ ?_
    · -- Show that natDegree p ≤ sup of degrees of summands
      apply (natDegree_sum_le (s := s) (f := fun i => f i • Lagrange.basis s v i))
    · -- Show that sup of degrees of summands < s.card
      rw [Finset.sup_lt_iff]
      · exact hDegs
      · -- Show that s is nonempty
        simp only [Nat.bot_eq_zero, Finset.card_pos]
        exact hs
  · -- Prove that p interpolates points (v i, f i) for i ∈ s
    intro i hi
    -- Simplify the eval of the sum to sum of evals with scalars taken out
    rw [eval_poly_sum_eq_eval_sum_poly]
    simp only [eval_smul, smul_eq_mul]
    -- Now use the kronecker delta property
    -- Show that only one summand is nonzero so the sum reduces to that summand
    rw [Finset.sum_eq_single i]
    · -- Show that the summand at i is f i
      rw [lagrange_basis_kronecker_delta s v hv hi]
      simp only [reduceIte, mul_one]
    · -- Show that for b ∈ s, b ≠ i, the summand is zero
      intro b hb hBNeqi
      rw [lagrange_basis_kronecker_delta s v hv hi]
      simp only [mul_ite, mul_one, mul_zero, ite_eq_right_iff]
      intro hBEqi
      contradiction
    · -- Additional goal that contradicts with i ∈ s
      intro hi'
      contradiction
end Lagrange
