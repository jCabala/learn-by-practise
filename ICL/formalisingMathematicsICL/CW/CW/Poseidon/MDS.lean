import Mathlib
import CW.Poseidon.CauchyMatrix
import CW.Poseidon.CMDet

/-!
# MDS Matrices
-/

namespace MDS

open Matrix

/-! ## Definitions of MDS matrix -/

/-- A `t × t` matrix over a commutative ring is **MDS** if every `k × k` submatrix obtained
    by choosing `k` distinct rows and `k` distinct columns has nonzero determinant. -/
def IsMDS {R : Type*} [CommRing R] {t : ℕ} (M : Matrix (Fin t) (Fin t) R) : Prop :=
  -- `↪` denotes injective functions. It automatically forces `k ≤ t`.
  ∀ (k : ℕ) (f g : Fin k ↪ Fin t), (M.submatrix f g).det ≠ 0

/-- Every Cauchy matrix is an MDS matrix. -/
theorem CauchyMatrix.isMDS {F : Type*} [Field F] {t : ℕ}
    (C : CauchyMatrix F t) :
    IsMDS C.toMatrix := by
  /- Use the facts that every square submatrix of a Cauchy matrix is a Cauchy matrix, and every
     Cauchy matrix has nonzero determinant. -/
  intro k f g
  obtain ⟨C', hC'⟩ := C.submatrix_is_Cauchy k f g
  grind [C'.non_zero_determinant]

/-! ## Existance Theorem -/

/-- **Existence of MDS matrices over prime fields.**
    If `2t ≤ p` then there exists a `t × t` MDS matrix over `ZMod p`. -/
theorem exists_mds_matrix {p : ℕ} [Fact (Nat.Prime p)] {t : ℕ}
    (h : 2 * t ≤ p) :
    ∃ M : Matrix (Fin t) (Fin t) (ZMod p), IsMDS M := by
  /- We construct a Cauchy matrix with row parameters `0, 1, ..., t-1` and column parameters
    `t, t+1, ..., 2t-1`. The conditions on the row and column parameters are satisfied because
    `2t <= p`, so all these numbers are distinct mod p. Then we use the fact that Cauchy matrices
     are MDS. -/
  /- First, we prove a result allowing for deduplication of the injectivity proof for the row and
     column parameters. -/
  have finCast_injective : Function.Injective (fun i : Fin t => (i : ZMod p)) := by
    /- Result used later to build the Cauchy matrix.
       Since `t < p`, the map `i ↦ (i : ZMod p)` is injective on `Fin t`.-/
    intro i j hij -- hij: `(i.val : ZMod p) = (j.val : ZMod p)`
    apply Fin.ext -- To show i = j, it suffices to show i.val = j.val
    rw [ZMod.natCast_eq_natCast_iff'] at hij -- Convert to nat equality mod p
    have hi : (i.val : ℕ) < p := by omega
    have hj : (j.val : ℕ) < p := by omega
    -- Use the fact that `i.val ≡ j.val (mod p)` and both are `< p` to conclude `i.val = j.val`
    rwa [Nat.mod_eq_of_lt hi, Nat.mod_eq_of_lt hj] at hij
  -- Now we construct the Cauchy matrix.
  let C : CauchyMatrix (ZMod p) t :=
    { x := fun i => i
      y := fun j => t + j
      hx := by exact finCast_injective -- Prove that all row parameters are distinct
      hy := by -- Prove that all column parameters are distinct
        intro i j hij
        -- Cancel t from both sides to get the same goal as hx and use the finCast_injective lemma
        apply add_left_cancel at hij
        exact finCast_injective hij
      hxy := by
        /- Prove that no row parameter equals any column parameter.
           By contradiction: if `x i = y j`, then `(i : ZMod p) = (t + j : ZMod p)`, so
          `i.val ≡ t + j.val (mod p)`. Since both sides are < p, the congruence implies
          `i.val = t + j.val`, but `i.val < t` while `t + j.val ≥ t`, a contradiction.
        -/
        intro i j hij
        have hi : (i.val : ℕ) < p := by omega
        have hj : (t + j.val : ℕ) < p := by omega
        have hij' : (i.val : ZMod p) = ((t + j.val : ℕ) : ZMod p) := by grind
        have hnat : (i.val : ℕ) = t + j.val := by
          -- Use the fact that `i.val ≡ t + j.val (mod p)` and both are `< p`
          rwa [ZMod.natCast_eq_natCast_iff', Nat.mod_eq_of_lt hi, Nat.mod_eq_of_lt hj] at hij'
        grind
    }
  -- Finally, we use our constructed matrix and the fact that all Cauchy matrices are MDS
  use C.toMatrix
  exact C.isMDS

end MDS
