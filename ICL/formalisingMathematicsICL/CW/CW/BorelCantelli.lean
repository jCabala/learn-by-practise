import Mathlib

/-!
# Borel-Cantelli

## First Borel-Cantelli lemma
Let (P, F, Ω) be a probability space. Let {A n : n ∈ ℕ } be a sequence of measurable sets.
If `∑ P (A n)` is finite, then `P (limsup A) = 0`.

## General proof strategy
First define the tail sets `B k = ⋃_{n ≥ k} A n`.
Show that `P (B k) → P (limsup A)` as `k → ∞` `(1)`, and that `P (B k) → 0` as `k → ∞` `(2)`.
Then use the uniqueness of limits to conclude that `P (limsup A) = 0`.
For `(1)`, use the continuity from above.
For `(2)`, bound P (B k) from above by the tail sum ∑_{n ≥ k} P (A n), and show
that the tail sum goes to 0 as k → ∞. Then use the squeeze test to conclude P (B k) → 0.
-/

open scoped BigOperators
open Set Filter

namespace MeasureTheory

/- Definitions of the basic sets used in the proof.
Further assumptions on α are added later to avoid warnings about unused variables. -/
variable {α : Type*}
variable (A : ℕ → Set α)

/-! ## Definition of `B k` and useful facts-/

-- Define B_k = ⋃_{n ≥ k} A_n
abbrev B (A : ℕ → Set α) (k : ℕ) : Set α := ⋃ n : {n : ℕ | n ≥ k}, A n

/-- ### Fact 1
`limsup A = ⋂_{k} B_k`. -/
lemma limsup_eq_iInter_B :
    limsup A atTop = ⋂ k, B A k := by
  -- Prove the equality of sets by showing that for all x, x ∈ limsup A ↔ x ∈ ⋂ k, B k.
  ext x
  rw [mem_iInter] -- Turn x ∈ ⋂ k, B k into "for all k, x ∈ B k"
  -- Helper used in both directions: unfold x ∈ B k into ∃ n ≥ k, x ∈ A n.
  have hB_mem : ∀ k, x ∈ B A k ↔ ∃ n ≥ k, x ∈ A n := by
    intro k
    simp only [iUnion_coe_set, ge_iff_le, mem_setOf_eq, mem_iUnion, exists_prop]
  -- Deconstructing equivalence into two implications
  constructor
  · -- → direction (x ∈ limsup A → ∀ k, x ∈ B k)
    intro hx k
    -- Unfold the filter-based limsup: hx becomes ∀ j, ∃ m, x ∈ A (m + j).
    simp only [limsup_eq_iInf_iSup_of_nat', iSup_eq_iUnion,
      iInf_eq_iInter, mem_iInter, mem_iUnion] at hx
    -- At index k we get some m with x ∈ A (m + k), so n := m + k ≥ k works.
    obtain ⟨m, hm⟩ := hx k
    rw [hB_mem k]
    use m + k
    constructor
    · -- m + k ≥ k
      omega
    · -- x ∈ A (m + k)
      exact hm
  · -- ← direction (∀ k, x ∈ B k → x ∈ limsup A)
    intro hx
    -- Unfold limsup into ⋂ k, ⋃ n, A (n + k) and introduce k.
    simp only [limsup_eq_iInf_iSup_of_nat', iSup_eq_iUnion, iInf_eq_iInter, mem_iInter,
      mem_iUnion]
    intro k
    -- Use hB_mem to extract n ≥ k with x ∈ A n from x ∈ B k.
    obtain ⟨n, hn, hxn⟩ := (hB_mem k).mp (hx k) -- mp: One direction of the iff
    -- Use n - k so that A ((n - k) + k) = A n.
    use n - k
    grind

/-- ### Fact 2
`B_k` is a decreasing sequence of sets (i.e. `B_{k+1} ⊆ B_k`).
Note: defined as `Antitone` as it is useful later in this form. -/
lemma B_antitone : Antitone (B A) := by
  -- First prove inclusion B (k + 1) ⊆ B k for all k
  have hB_inclusion : ∀ k, B A (k + 1) ⊆ B A k := by
    unfold B
    intro k x hx
    rw [mem_iUnion] at hx
    simp only [coe_setOf, mem_setOf_eq, mem_iUnion, Subtype.exists, ge_iff_le, exists_prop]
    obtain ⟨n, hn⟩ := hx
    -- We can use n
    use n
    constructor
    · -- n >= k
      -- n.property gives n ≥ k + 1
      have hk1 : k + 1 ≤ (n : ℕ) := n.property
      exact le_trans (Nat.le_succ k) hk1
    · -- x ∈ A n
      exact hn
  unfold Antitone
  simp only [le_eq_subset] -- Changing B a ≤ B b into B a ⊆ B b
  intro a b hab
  -- `Nat.le` (`a ≤ b`) is inductive: `refl : a ≤ a` and `step : a ≤ m → a ≤ m + 1`
  -- We can proceed by induction on hab.
  induction hab with
  | refl => trivial -- Goal: B a ⊆ B a
  | step _ ih =>
    -- IH (`ih`): B m ⊆ B a
    -- Goal: B (m + 1) ⊆ B a
    simp only [Nat.succ_eq_add_one]
    exact Set.Subset.trans (hB_inclusion _) ih

/- Further variable definitions.
Not defined at the top as I was getting warnings about unused variables. -/
variable [MeasurableSpace α]
variable (P : ProbabilityMeasure α)

/-- ### Fact 3
`B_k` is measurable for each `k`. -/
lemma B_measurable
    (hA : ∀ n, MeasurableSet (A n)) :
    ∀ k : ℕ, MeasurableSet (B A k) := by
  intro k
  -- B k = ⋃_{n ≥ k} A n is a countable union of measurable sets, so it is measurable.
  exact MeasurableSet.iUnion (fun n => hA n)

/-! ## Proof of the first limit: `P (B k) → P (limsup A)` as `k → ∞` (1) -/
/-- Probability of the tail set `B k` tends to the probability of `limsup A` as `k → ∞`. -/
lemma tendsto_prob_B_limsup
    (hA : ∀ n, MeasurableSet (A n)) :
    Tendsto (fun N => P (B A N)) atTop (nhds (P (limsup A atTop))) := by
  rw [limsup_eq_iInter_B (A := A)]
  /- This is a direct application of the continuity from above, which in Lean is expressed as
     `tendsto_measure_iInter_atTop`.
     Problem: our goal is NNReal convergence, but `tendsto_measure_iInter_atTop` gives
     ENNReal convergence. The next step translates NNReal convergence to ENNReal convergence
     via `ENNReal.tendsto_toNNReal.comp`.
     After that step, the remaining goal uses ↑P (the coercion to Measure α, which is
     ENNReal-valued) instead of the original NNReal-valued P. -/
  apply Filter.Tendsto.comp (ENNReal.tendsto_toNNReal ?_)
  -- -- Apply Tendsto.comp (Tendsto f l₁ l₂ → Tendsto g l₂ l₃ → Tendsto (g ∘ f) l₁ l₃) with
  -- f N := (↑P : Measure α) (B A N) and g := ENNReal.toNNReal. Lean inferred the type of ↑P itself.
  · -- Original goal but the ENNReal version (P is casted from ProbabilityMeasure to Measure)
    have hf : ∃ i, (↑P : Measure α) (B A i) ≠ ⊤ := by
      -- This hypothesis proves finitness of measure of at least one B k
      -- Because P is a probability measure, it is always finite so we can take any i.
      use 0
      simp only [ne_eq, measure_ne_top, not_false_eq_true]
    have hBnm : ∀ k : ℕ, NullMeasurableSet (B A k) P := by
      intro k
      apply MeasurableSet.nullMeasurableSet
      apply B_measurable (A := A) hA k
    exact tendsto_measure_iInter_atTop
      (μ := (↑P : Measure α)) hBnm (B_antitone (A := A)) hf
  · -- Show that the limit is finite
    apply measure_ne_top (↑P : Measure α) (⋂ k, B A k)

/-! ## Lemmas about the tails sum useful in the 2nd limit proof-/
/-- Probability of `B k` is bounded above by the tail sum of `P (A n)`. -/
lemma prob_B_le_tsum
    (hSummable : Summable (fun n : ℕ => P (A n))) :
    ∀ k, P (B A k) ≤ ∑' n : {n : ℕ | n >= k}, P (A n) := by
  intro k
  /- The subtype {n : ℕ | n ≥ k} was not easiest to work with directly, so I prove the bound on the
  if-else version of this lemma and then prove that it is equivalent to the subtype version-/
  -- Start with proving the summability of the if-else version:
  have h_summable: Summable fun n ↦ P.toFiniteMeasure (if n ≥ k then A n else ∅) := by
    -- We show it is termwise smaller than the original summable sequence, so it is also summable.
    -- The existing lemma is for ℝ valued summability, so we need to use the coercion from NNReal
    apply (NNReal.summable_coe).mp
    apply Summable.of_nonneg_of_le
      (g := fun n => ((P.toFiniteMeasure (if n ≥ k then A n else ∅) : NNReal) : ℝ))
      (f := fun n => ((P.toFiniteMeasure (A n) : NNReal) : ℝ))
    · -- Nonnegativity of the if-else version
      intro b
      simp only [ge_iff_le, ProbabilityMeasure.toFiniteMeasure_apply_eq_apply, NNReal.zero_le_coe]
    · -- The if-else version is smaller than the original sequence
      intro b
      simp only [ge_iff_le, ProbabilityMeasure.toFiniteMeasure_apply_eq_apply, NNReal.coe_le_coe]
      by_cases h : k ≤ b
      · -- k ≤ b
        rw [if_pos h]
      · -- k > b
        rw [if_neg h]
        simp only [ProbabilityMeasure.coeFn_empty, zero_le]
    · -- Show that the original sequence is summable
      simp only [ProbabilityMeasure.toFiniteMeasure_apply_eq_apply]
      apply (NNReal.summable_coe).mpr
      exact hSummable
  -- Now we can prove the bound on the if-else version using the existing σ-subadditivity lemma:
  have h_sub_if :
      P (⋃ n : ℕ, if n >= k then A n else ∅)
      ≤
      ∑' n : ℕ, P (if n >= k then A n else ∅) := by
    -- Apply existing σ-subadditivity lemma for finite measures
    exact (FiniteMeasure.apply_iUnion_le
      (μ := P.toFiniteMeasure)
      (f := fun n : ℕ => if n >= k then A n else ∅))
      (hf := h_summable)
  -- Now we show that the if-else version is equivalent to the subtype version:
  have h_union :
      (⋃ n : {n : ℕ | n ≥ k}, A n) = ⋃ n : ℕ, (if n ≥ k then A n else ∅) := by
    -- Unwrapping the subtype version of the union into the if-else version.
    ext x -- Prove the equality of sets by showing that for all x, x ∈ LHS ↔ x ∈ RHS.
    simp only [mem_iUnion] -- Turn the ∈ ⋃ into ∃
    constructor
    · -- → direction
      intro hx
      obtain ⟨n, hx_in⟩ := hx
      use n
      grind
    · -- ← direction
      intro hx
      obtain ⟨n, hx_in⟩ := hx
      by_cases h : n ≥ k
      · simp only [coe_setOf, mem_setOf_eq, Subtype.exists, ge_iff_le, exists_prop]
        use n
        grind
      · simp only [coe_setOf, mem_setOf_eq, Subtype.exists, ge_iff_le, exists_prop]
        grind
  have h_sum :
      ∑' n : {n : ℕ | n ≥ k}, P (A n) = ∑' n : ℕ, P (if n ≥ k then A n else ∅) := by
    -- Unwrapping the subtype version of the sum into the if-else version.
    convert (tsum_subtype (s := {n : ℕ | n ≥ k}) (f := fun n : ℕ => P (A n)))
    rename_i y
    by_cases hx : y ≥ k
    · simp only [ge_iff_le, hx, ↓reduceIte, mem_setOf_eq, indicator_of_mem]
    · simp only [ge_iff_le, hx, ↓reduceIte, ProbabilityMeasure.coeFn_empty,
        mem_setOf_eq, not_false_eq_true, indicator_of_notMem]
  -- Finish by applying everything together:
  unfold B
  rw [h_union, h_sum]
  exact h_sub_if

/-- The tail sum of `P (A n)` tends to 0 as `k → ∞` -/
lemma tendsto_tail_tsum_zero
    :
    Tendsto (fun k => ∑' n : {n : ℕ | n >= k}, P (A n)) atTop (nhds 0) := by
  /- Mathlib already has `NNReal.tendsto_sum_nat_add`, which says that for a summable f,
     ∑_{m=0}^∞ f(m + k) → 0  as k → ∞.
     Our tail sum is indexed over the subtype {n : ℕ | n ≥ k}, not "over ℕ with a shift".
     So we need to transform our tail into the "shifted" version to apply the existing lemma.
     To reindex, we use `Equiv.tsum_eq`: if e : β ≃ γ is a bijection, then
     ∑'(b : β) f(b) = ∑'(c : γ) f(e c). -/
  have h_eq : ∀ k, ∑' n : {n : ℕ | n >= k}, P (A n) = ∑' m, P (A (m + k)) := by
    intro k
    -- First we build e : ℕ ≃ {n : ℕ | n ≥ k} as the bijection m ↦ m + k,
    -- with inverse n ↦ n - k.
    let e : ℕ ≃ {n : ℕ | n >= k} :=
      { -- toFun maps from ℕ to the subtype. We prived the mapping (m + k) and the proof that it
        -- is in the subtype .
        toFun := fun m => ⟨m + k, by grind⟩
        -- invFun maps from the subtype to N.
        invFun := fun x => x.val - k
        -- left_inv is a proof that for all m, invFun (toFun m) = m
        left_inv := by grind
        -- right_inv is a proof that for all x, toFun (invFun x) = x.
        right_inv := by grind
      }
    -- After building the bijection we can use `Equiv.tsum_eq` to reindex the sum.
    exact (Equiv.tsum_eq (e := e) (f := fun n : {n : ℕ | n >= k} => P (A ↑n))).symm
  have h_eq_fun :
      (fun k => ∑' n : {n : ℕ | n >= k}, P (A n))
        = (fun k => ∑' m, P (A (m + k))) := by
    -- Build a function-level equality to rewrite with it
    funext k -- Prove 2 functions are equal by showing that they are equal at each input k.
    exact h_eq k
  rw [h_eq_fun]
  exact NNReal.tendsto_sum_nat_add (fun n => P (A n))

/-! ## Proof of the 2nd limit: `P (B k) → 0` as `k → ∞` (2) -/
/-- Probability of the tail set `B k` tends to 0 as `k → ∞` -/
lemma tendsto_prob_B_zero
    (hSummable : Summable (fun n : ℕ => P (A n))) :
    Tendsto (fun N => P (B A N)) atTop (nhds 0) := by
  -- Idea: Use squeeze theorem to show that P (B k) → 0 as k → ∞, using the tail sum as a bound.
  have h_tends_0 : Tendsto (fun _ : ℕ => (0 : NNReal)) atTop (nhds 0) := by
    -- Wrap a constant sequence at 0 to use as a lower bound for P (B k)
    simp only [tendsto_const_nhds]
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le
    (f := fun k : ℕ => P (B A k))
    (hg := h_tends_0) -- g(x) = 0
    (hh := tendsto_tail_tsum_zero (P := P) (A := A)) -- h(x) = tail sum
    ?_ ?_
  · -- 0 ≤ P (B k)
    intro k
    simp only [zero_le]
  · -- P (B k) ≤ tail k
    intro k
    apply prob_B_le_tsum (P := P) (A := A) hSummable

/-! ## Main Theorem -/
/-- First Borel-Cantelli lemma: If the sum of P (A n) is finite, then P (limsup A) = 0. -/
theorem first_borel_cantelli_lemma
    (hA : ∀ n, MeasurableSet (A n))
    (hSummable : Summable (fun n : ℕ => P (A n))) :
    P (limsup A atTop) = 0 := by
  /- At that point just application of the uniqueness of limits.
  We have shown that P (B k) → P (limsup A) and that P (B k) → 0, so P (limsup A) = 0 -/
  apply tendsto_nhds_unique
    (tendsto_prob_B_limsup (P := P) (A := A) hA)
    (tendsto_prob_B_zero (P := P) (A := A) hSummable)

end MeasureTheory
