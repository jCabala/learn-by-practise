import Mathlib

open scoped BigOperators
open Set Filter

namespace MeasureTheory

variable {α : Type*} [MeasurableSpace α]
variable (P : ProbabilityMeasure α)
variable (A : ℕ → Set α)

/--
**Borel–Cantelli lemma (1)**:
If `∑ μ (A n)` is finite (i.e. the series is summable), then `μ (limsup A) = 0`.
-/
theorem borel_cantelli_1
    (hA : ∀ n, MeasurableSet (A n))
    (hSummable : Summable (fun n : ℕ => P (A n))) :
    P (limsup A atTop) = 0 := by
  -- Define B_k = ⋃_{n ≥ k} A_n
  let B : ℕ → Set α := fun k => ⋃ n ≥ k, A n

  -- Proof strategy: Show that P (B k) → P (limsup A) as k → ∞, and that P (B k) → 0 as k → ∞.
  -- Then use the uniqueness of limits to conclude that P (limsup A) = 0.

  --------------------------------------------------------------------------------------------------
  -------------------------------- Useful facts about B_k: -----------------------------------------
  --------------------------------------------------------------------------------------------------
  -- Proof that limsup A = ⋂_{k} B_k
  have h_limsup_eq_infi_B :
    limsup A atTop = ⋂ k, B k := by
    ext x
    simp only [mem_iInter]
    constructor
    · -- → direction
      intro hx k
      -- Turn hx into the "for all k, exists n ≥ k, x ∈ A n" form
      have hx' : ∀ j : ℕ, ∃ n ≥ j, x ∈ A n := by
        simp only [limsup_eq_iInf_iSup_of_nat', iSup_eq_iUnion,
        iInf_eq_iInter, mem_iInter, mem_iUnion] at hx
        intro j
        specialize hx j
        obtain ⟨k, hk⟩ := hx
        use k + j
        grind
      -- Now prove x ∈ ⋂ k, B k
      specialize hx' k
      obtain ⟨n, hn_ge, hx_in⟩ := hx'
      rw [mem_iUnion]
      use n
      simp only [mem_iUnion]
      grind
    · -- ← direction
      intro hx
      simp only [limsup_eq_iInf_iSup_of_nat', iSup_eq_iUnion, iInf_eq_iInter, mem_iInter,
        mem_iUnion]
      intro k
      specialize hx k
      unfold B at hx
      simp only [ge_iff_le, mem_iUnion, exists_prop] at hx
      obtain ⟨m, hm_ge, hx_in⟩ := hx
      use m - k
      grind
  -- Proof that B_k is a decreasing sequence of sets (i.e. B_{k+1} ⊆ B_k).
  -- Defined as `Antitone` as it is useful later in this form.
  have hB_anti : Antitone B := by
    have hB_inclusion : ∀ k, B (k + 1) ⊆ B k := by
      unfold B
      intro k x hx
      rw [mem_iUnion] at hx
      rw [mem_iUnion]
      obtain ⟨n, hx_in, hn_ge⟩ := hx
      use n
      rw [mem_iUnion]
      grind
    unfold Antitone
    simp only [le_eq_subset]
    intro a b hab
    -- Use induction on the proof of a ≤ b to show that B b ⊆ B a.
    -- The base case is trivial, and the step case follows from the inclusion B (k + 1) ⊆ B k.
    induction hab with
    | refl => trivial -- Assuming B a ⊆ B a
    | step _ ih => {
      -- Assuming B k ⊆ B a, show that B (k + 1) ⊆ B a.
      -- This follows from the inclusion B (k + 1) ⊆ B k and the induction hypothesis. -}
      simp only [Nat.succ_eq_add_one]
      rename_i _ m _
      specialize hB_inclusion m
      grind
    }
  -- Proof that B_k is measurable for each k
  have hB_meas : ∀ k : ℕ, MeasurableSet (B k) P := by
    intro k
    exact (MeasurableSet.iUnion fun n =>
      MeasurableSet.iUnion fun _ => hA n).nullMeasurableSet
  --------------------------------------------------------------------------------------------------
  -------------------------------- P (B k) → P (limsup A) as k → ∞ ---------------------------------
  --------------------------------------------------------------------------------------------------
  have h_lim_Bk: Tendsto (fun N => P (B N)) atTop (nhds (P (limsup A atTop))) := by
      rw [h_limsup_eq_infi_B]
      -- This is a direct application of the continuity from above
      -- We already proved the measurability and monotonicity conditions.

      -- Further statements that we need to apply the continuity from above:
      have hf : ∃ i, (↑P : Measure α) (B i) ≠ ⊤ := ⟨0, measure_ne_top _ _⟩
      

      -- Continuity from above: convert ENNReal → NNReal convergence
      exact (ENNReal.tendsto_toNNReal (measure_ne_top _ _)).comp
        (tendsto_measure_iInter_atTop (μ := (↑P : Measure α)) hB_meas hB_anti hf)

  --------------------------------------------------------------------------------------------------
  ------------------------------ P (B k) → 0 -------------------------------------------------------
  --------------------------------------------------------------------------------------------------
  have h_lim_Bk_zero : Tendsto (fun N => P (B N)) atTop (nhds 0) := by
    -- Idea: First show that P (B k) is boundd from above by the tail sum ∑_{n ≥ k} P (A n), then show that the tail sum goes to 0 as k → ∞. After that use squeeze test to conclude that P (B k) → 0.
    -- Use subadditivity of P to show that P (B k) ≤ ∑_{n ≥ k} P (A n)
    have h_Bk_bound :
      ∀ k, P (B k) ≤ ∑' n : {n : ℕ | n >= k}, P (A n) := by
      intro k
      -- Apply σ-subadditivity for the NNReal-valued finite measure first
      have h_sub :
          P (⋃ n : {n : ℕ | n ≥ k}, A n) ≤ ∑' n : {n : ℕ | n >= k}, P (A n) := by
          ------------------------------------------------------------------------
          have h_sub_if:
            P (⋃ n : ℕ, if n >= k then A n else ∅)
            ≤
            ∑' n : ℕ, P (if n >= k then A n else ∅) := by
            apply (FiniteMeasure.apply_iUnion_le
              (μ := P.toFiniteMeasure)
              (f := fun n : ℕ => if n >= k then A n else ∅))
            simp only [ge_iff_le, ProbabilityMeasure.toFiniteMeasure_apply_eq_apply]

            -- Now prove summability
            -- Because of some type issues, first prove in ℝ and then convert to NNReal
            have hR :
                Summable (fun n : ℕ => (P (if k ≤ n then A n else ∅) : ℝ)) := by
              refine Summable.of_nonneg_of_le
                (g := fun n : ℕ => (P (if k ≤ n then A n else ∅) : ℝ))
                (f := fun n : ℕ => (P (A n) : ℝ))
                ?_ ?_ ?_
              · intro n
                simp only [NNReal.zero_le_coe]
              · intro b
                by_cases h : k ≤ b
                · grind
                · simp only [if_neg h, ProbabilityMeasure.coeFn_empty, NNReal.coe_zero,
                  NNReal.zero_le_coe]
              · apply (NNReal.summable_coe (f := fun n : ℕ => P (A n))).2 hSummable
            -- Now convert to NNReal
            exact (NNReal.summable_coe (f := fun n : ℕ => P (if k ≤ n then A n else ∅))).1 hR
          -------------------------------------------------------------------------
          have h_union:
            (⋃ n : {n : ℕ | n ≥ k}, A n) = ⋃ n : ℕ, (if n ≥ k then A n else ∅) := by
            -- What has happened to n, why is my hypothesis n >= k gone?
            ext x
            simp only [mem_iUnion]
            constructor
            · intro hx
              obtain ⟨n, hx_in⟩ := hx
              use n
              simp only [ge_iff_le]
              grind
            · intro hx
              obtain ⟨n, hx_in⟩ := hx
              by_cases h : n ≥ k
              · simp only [coe_setOf, mem_setOf_eq, Subtype.exists, ge_iff_le, exists_prop]
                use n
                grind
              · simp only [coe_setOf, mem_setOf_eq, Subtype.exists, ge_iff_le, exists_prop]
                grind
          have h_sum:
            ∑' n : {n : ℕ | n ≥ k}, P (A n) = ∑' n : ℕ, P (if n ≥ k then A n else ∅) := by
                convert (tsum_subtype ({n : ℕ | n ≥ k} : Set ℕ) (fun n : ℕ => P (A n)))
                rename_i y
                by_cases hx : y ≥ k
                · simp only [hx, ↓reduceIte, mem_setOf_eq, indicator_of_mem]
                · simp only [ge_iff_le, hx, ↓reduceIte, ProbabilityMeasure.coeFn_empty,
                  mem_setOf_eq, not_false_eq_true, indicator_of_notMem]

          rw [h_sum]
          grind
      have hB_subtype : B k = ⋃ n : {n : ℕ | n ≥ k}, A n := by
        ext x
        unfold B
        simp only [ge_iff_le, mem_iUnion, exists_prop, coe_setOf, mem_setOf_eq, Subtype.exists]
      simpa [hB_subtype] using h_sub

    -- Use the summability of P (A n) to show that ∑_{n ≥ k} P (A n) → 0 as k → ∞
    -- ----------------- FULLY AISSISTED PROOF ----------------------------------:
    have h_sum_tail_zero :
        Tendsto (fun k => ∑' n : {n : ℕ | n >= k}, P (A n)) atTop (nhds 0) := by
        -- Reindex: ∑_{n ≥ k} f(n) = ∑_m f(m + k) via the equivalence m ↦ m + k
        have h_eq : ∀ k, ∑' n : {n : ℕ | n >= k}, P (A n) = ∑' m, P (A (m + k)) := by
          intro k
          -- Build an equivalence ℕ ≃ {n : ℕ | n >= k} via m ↦ m + k
          -- Using `let` so Lean can unfold ↑(e m) = m + k definitionally
          let e : ℕ ≃ {n : ℕ | n >= k} :=
            ⟨fun m => ⟨m + k, show m + k >= k by omega⟩,
             fun ⟨n, _⟩ => n - k,
             fun m => by show m + k - k = m; omega,
             fun ⟨n, (h : n >= k)⟩ => by ext; show n - k + k = n; omega⟩
          exact (e.tsum_eq (fun n : {n : ℕ | n >= k} => P (A ↑n))).symm
        simp_rw [h_eq]
        exact NNReal.tendsto_sum_nat_add (fun n => P (A n))
    -- ----------------------------------------------------------------------

    -- Wrap a constant sequence at 0 to use as a lower bound for P (B k)
    have h_tends_0 : Tendsto (fun _ : ℕ => (0 : NNReal)) atTop (nhds 0) := by
      simp only [tendsto_const_nhds]

    -- Now squeeze P (B k) between 0 and the tail sum to conclude that P (B k) → 0
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le
      (f := fun k : ℕ => P (B k))
      (g := fun _ : ℕ => (0 : NNReal))
      (h := fun k => ∑' n : {n : ℕ | n >= k}, P (A n))
      (b := atTop)
      (a := (0 : NNReal))
      (hg := h_tends_0)
      (hh := h_sum_tail_zero)
      ?_ ?_
    · -- 0 ≤ P (B k)
      intro k
      simp only [zero_le]
    · -- P (B k) ≤ tail k
      intro k
      apply h_Bk_bound

  apply tendsto_nhds_unique h_lim_Bk h_lim_Bk_zero


end MeasureTheory
