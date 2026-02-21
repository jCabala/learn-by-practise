import Mathlib

open scoped BigOperators
open Set Filter

namespace MeasureTheory

variable {α : Type*} [MeasurableSpace α]
variable (P : ProbabilityMeasure α)
variable (A : ℕ → Set α)

/- **First Borel–Cantelli lemma**:
   Let (P, F, Ω) be a probability space. Let {A n : n ∈ ℕ } be a sequence of measurable sets.
   If `∑ P (A n)` is finite, then `P (limsup A) = 0`. -/
theorem first_borel_cantelli_lemma
    (hA : ∀ n, MeasurableSet (A n))
    (hSummable : Summable (fun n : ℕ => P (A n))) :
    P (limsup A atTop) = 0 := by
  -- General proof strategy:
  -- Define B_k = ⋃_{n ≥ k} A_n
  let B : ℕ → Set α := fun k => ⋃ n ≥ k, A n
  /- Show that P (B k) → P (limsup A) as k → ∞, and that P (B k) → 0 as k → ∞.
     Then use the uniqueness of limits to conclude that P (limsup A) = 0.
     For the first part, use the continuity from above.
     For the second part, bound P (B k) from above by the tail sum ∑_{n ≥ k} P (A n), and show
     that the tail sum goes to 0 as k → ∞. Then use the squeeze test to conclude P (B k) → 0. -/
  --------------------------------------------------------------------------------------------------
  -------------------------------- Useful facts about B_k: -----------------------------------------
  --------------------------------------------------------------------------------------------------
  -- Fact 1: limsup A = ⋂_{k} B_k
  have h_limsup_eq_infi_B :
    limsup A atTop = ⋂ k, B k := by
    -- Prove the equality of sets by showing that for all x, x ∈ limsup A ↔ x ∈ ⋂ k, B k.
    ext x
    simp only [mem_iInter] -- Turn x ∈ ⋂ k, B k into "for all k, x ∈ B k"
    -- Helper used in both directions: unfold x ∈ B k into ∃ n ≥ k, x ∈ A n.
    have hB_mem : ∀ k, x ∈ B k ↔ ∃ n ≥ k, x ∈ A n := by
      intro k; simp only [B, ge_iff_le, mem_iUnion, exists_prop]
    -- Deconstructing equivalence into two implications
    constructor
    · -- → direction (x ∈ limsup A → ∀ k, x ∈ B k)
      intro hx k
      -- Unfold the filter-based limsup: hx becomes ∀ j, ∃ m, x ∈ A (m + j).
      simp only [limsup_eq_iInf_iSup_of_nat', iSup_eq_iUnion,
        iInf_eq_iInter, mem_iInter, mem_iUnion] at hx
      -- At index k we get some m with x ∈ A (m + k), so n := m + k ≥ k works.
      obtain ⟨m, hm⟩ := hx k
      exact (hB_mem k).mpr ⟨m + k, by omega, hm⟩
    · -- ← direction (∀ k, x ∈ B k → x ∈ limsup A)
      intro hx
      -- Unfold limsup into ⋂ k, ⋃ n, A (n + k) and introduce k.
      simp only [limsup_eq_iInf_iSup_of_nat', iSup_eq_iUnion, iInf_eq_iInter, mem_iInter,
        mem_iUnion]
      intro k
      -- Use hB_mem to extract n ≥ k with x ∈ A n from x ∈ B k.
      -- Re-index: use n - k so that A ((n - k) + k) = A n.
      obtain ⟨n, hn, hxn⟩ := (hB_mem k).mp (hx k)
      use n - k
      -- Prove n - k + k = n and use assumption
      rw [Nat.sub_add_cancel]
      · exact hxn
      · convert hn
  -- Fact 2: B_k is a decreasing sequence of sets (i.e. B_{k+1} ⊆ B_k).
  -- Note: defined as `Antitone` as it is useful later in this form.
  have hB_anti : Antitone B := by
    -- First prove inclusion B (k + 1) ⊆ B k for all k
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
    simp only [le_eq_subset] -- Changing B a ≤ B b into B a ⊆ B b
    intro a b hab
    -- `Nat.le` (`a ≤ b`) is inductive: `refl : a ≤ a` and `step : a ≤ m → a ≤ m + 1`
    -- We can proceed by induction on hab.
    induction hab with
    | refl => trivial -- Goal: B a ⊆ B a
    | step _ ih => {
      -- IH (`ih`): B m ⊆ B a
      -- Goal: B (m + 1) ⊆ B a
      simp only [Nat.succ_eq_add_one]
      rename_i _ m _
      specialize hB_inclusion m
      grind
    }
  -- Fact 3: B k can be written as a subtype-indexed union ⋃ n : {n | n ≥ k}, A n.
  -- (Our definition uses ⋃ n ≥ k, which Lean treats differently from the subtype form.)
  have hB_subtype : ∀ k, B k = ⋃ n : {n : ℕ | n ≥ k}, A n := by
    intro k
    ext x
    simp only [B, ge_iff_le, mem_iUnion, exists_prop, coe_setOf, mem_setOf_eq, Subtype.exists]
  -- Fact 4: B_k is measurable for each k
  have hB_meas : ∀ k : ℕ, MeasurableSet (B k) := by
    intro k
    -- B k = ⋃_{n ≥ k} A n is a countable union of measurable sets, so it is measurable.
    rw [hB_subtype k]
    exact MeasurableSet.iUnion (fun n => hA n)
  --------------------------------------------------------------------------------------------------
  -------------------------------- P (B k) → P (limsup A) as k → ∞ ---------------------------------
  --------------------------------------------------------------------------------------------------
  have h_lim_Bk: Tendsto (fun N => P (B N)) atTop (nhds (P (limsup A atTop))) := by
      rw [h_limsup_eq_infi_B]
      /- This is a direct application of the continuity from above, which in Lean is expressed as
         `tendsto_measure_iInter_atTop`.
         Problem: our goal is NNReal convergence, but `tendsto_measure_iInter_atTop` gives
         ENNReal convergence. The next step translates ENNReal convergence to NNReal convergence
         via `ENNReal.tendsto_toNNReal.comp`.
         After that step, the remaining goal uses ↑P (the coercion to Measure α, which is
         ENNReal-valued) instead of the original NNReal-valued P. -/
      apply (ENNReal.tendsto_toNNReal ?_).comp
      · -- Original goal but the NNReal version
        have hf : ∃ i, (↑P : Measure α) (B i) ≠ ⊤ := by
          -- Because P is a probability measure, it is always finite so we can take any i.
          use 0
          simp only [ne_eq, measure_ne_top, not_false_eq_true]
        have hBnm : ∀ k: ℕ, NullMeasurableSet (B k) P := by
          intro k
          apply MeasurableSet.nullMeasurableSet
          specialize hB_meas k
          apply hB_meas
        exact tendsto_measure_iInter_atTop (μ := (↑P : Measure α)) hBnm hB_anti hf
      · -- Show that the limit is finite
        apply measure_ne_top (↑P : Measure α) (⋂ k, B k)
  --------------------------------------------------------------------------------------------------
  ---------------------------------- P (B k) ≤ ∑_{n ≥ k} P (A n) -----------------------------------
  --------------------------------------------------------------------------------------------------
  have h_Bk_bound :
      ∀ k, P (B k) ≤ ∑' n : {n : ℕ | n >= k}, P (A n) := by
      intro k
      /- The idea is to unwrap B into the subtype-indexed union, prove σ-subadditivity there, and
         convert back. The subtype {n : ℕ | n ≥ k} was not easiest to work with directly, so we
         convert both the union and the sum into "if-else" versions over ℕ, prove the bound there,
         and then convert back. -/
      ----------------------------------------------------------------------------------------------
      have h_sub :
          P (⋃ n : {n : ℕ | n ≥ k}, A n) ≤ ∑' n : {n : ℕ | n >= k}, P (A n) := by
          -- The subtype version of the lemma.
          ------------------------------------------------------------------------------------------
          have h_sub_if:
            P (⋃ n : ℕ, if n >= k then A n else ∅)
            ≤
            ∑' n : ℕ, P (if n >= k then A n else ∅) := by
            apply (FiniteMeasure.apply_iUnion_le
              (μ := P.toFiniteMeasure)
              (f := fun n : ℕ => if n >= k then A n else ∅))
            simp only [ge_iff_le, ProbabilityMeasure.toFiniteMeasure_apply_eq_apply]
            -- The if version of the lemma.
            -- I had some problems when trying to prove directly on NNReal, so I switched to ℝ
            -- and then converted back to NNReal at the end.
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
          ------------------------------------------------------------------------------------------
          have h_union:
            (⋃ n : {n : ℕ | n ≥ k}, A n) = ⋃ n : ℕ, (if n ≥ k then A n else ∅) := by
            -- Unwrapping the subtype version of the union into the if-else version.
            ext x -- Prove the equality of sets by showing that for all x, x ∈ LHS ↔ x ∈ RHS.
            simp only [mem_iUnion] -- Turn the ∈ ⋃ into ∃
            constructor
            · -- → direction
              intro hx
              obtain ⟨n, hx_in⟩ := hx
              use n
              simp only [ge_iff_le]
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
          ------------------------------------------------------------------------------------------
          have h_sum:
            ∑' n : {n : ℕ | n ≥ k}, P (A n) = ∑' n : ℕ, P (if n ≥ k then A n else ∅) := by
                -- Unwrapping the subtype version of the sum into the if-else version.
                convert (tsum_subtype (s := {n : ℕ | n ≥ k}) (f := fun n : ℕ => P (A n)))
                rename_i y
                by_cases hx : y ≥ k
                · simp only [hx, ↓reduceIte, mem_setOf_eq, indicator_of_mem]
                · simp only [ge_iff_le, hx, ↓reduceIte, ProbabilityMeasure.coeFn_empty,
                  mem_setOf_eq, not_false_eq_true, indicator_of_notMem]
          ------------------------------------------------------------------------------------------
          rw [h_union, h_sum]
          exact h_sub_if
      -------------------------------------------------------------------------------------------
      rw [hB_subtype k]
      exact h_sub
  --------------------------------------------------------------------------------------------------
  ------------------------------- Tail sum ∑_{n ≥ k} P (A n) → 0 as k → ∞ --------------------------
  --------------------------------------------------------------------------------------------------
  have h_sum_tail_zero :
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
            toFun := fun m => ⟨m + k, by grind⟩,
            -- invFun maps from the subtype to N.
            invFun := fun x => x.val - k,
            -- left_inv is a proof that for all m, invFun (toFun m) = m
            left_inv := by
              -- Note: The ⟨ m + k, _⟩ idnicate being an element of a subtype.
              intro m
              simp only [add_tsub_cancel_right]
            -- right_inv is a proof that for all x, toFun (invFun x) = x.
            right_inv := by
              intro m
              simp only [coe_setOf, mem_setOf_eq]
              grind
          }
        -- After building the bijection we can
        exact (Equiv.tsum_eq (e := e) (f := fun n : {n : ℕ | n >= k} => P (A ↑n))).symm
      -- Now reindex using simp_rw (none of those tactics separately seemed to work)
      simp_rw [h_eq]
      exact NNReal.tendsto_sum_nat_add (fun n => P (A n))
  --------------------------------------------------------------------------------------------------
  ------------------------------ P (B k) → 0 -------------------------------------------------------
  --------------------------------------------------------------------------------------------------
  have h_lim_Bk_zero : Tendsto (fun N => P (B N)) atTop (nhds 0) := by
    -- Idea: Use squeeze theorem to show that P (B k) → 0 as k → ∞, using the tail sum as a bound.
    have h_tends_0 : Tendsto (fun _ : ℕ => (0 : NNReal)) atTop (nhds 0) := by
      -- Wrap a constant sequence at 0 to use as a lower bound for P (B k)
      simp only [tendsto_const_nhds]
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
  --------------------------------------------------------------------------------------------------
  ----------------------- Using uniqueness of limits to conclude P (limsup A) = 0 ------------------
  --------------------------------------------------------------------------------------------------
  apply tendsto_nhds_unique h_lim_Bk h_lim_Bk_zero


end MeasureTheory
