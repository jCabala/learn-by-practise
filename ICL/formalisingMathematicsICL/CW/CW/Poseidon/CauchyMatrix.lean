import Mathlib

/-!
# Cauchy Matrices
-/

namespace MDS

open Matrix

/-- A **Cauchy matrix** over a field, bundled with the validity conditions
    that all row parameters are distinct, all column parameters are distinct,
    and no row parameter equals any column parameter. -/
structure CauchyMatrix (F : Type*) [Field F] (t : ℕ) where
  x : Fin t → F -- Row parameters
  y : Fin t → F -- Column parameters
  hx : Function.Injective x -- Row parameters are pairwise distinct.
  hy : Function.Injective y -- Column parameters are pairwise distinct.
  hxy : ∀ i j, x i ≠ y j -- No row parameter equals any column parameter.

/-- The underlying `t × t` matrix of a Cauchy matrix, with entry `(i, j) = (x i - y j)⁻¹`. -/
def CauchyMatrix.toMatrix {F : Type*} [Field F] {t : ℕ}
    (C : CauchyMatrix F t) : Matrix (Fin t) (Fin t) F :=
  Matrix.of fun i j => (C.x i - C.y j)⁻¹

/-- Every square submatrix of a Cauchy matrix is again a Cauchy matrix,
    obtained by restricting the row and column parameters along injective maps. -/
lemma CauchyMatrix.submatrix {F : Type*} [Field F] {t : ℕ}
    (C : CauchyMatrix F t) (k : ℕ) (f g : Fin k ↪ Fin t) :
    ∃ C' : CauchyMatrix F k, C'.toMatrix = C.toMatrix.submatrix f g := by
    -- Construct the Cauchy sub-matrix
    let C' : CauchyMatrix F k :=
      { x := C.x ∘ f -- C.x is the rows of C
        y := C.y ∘ g -- C.y is the columns of C
        hx := by -- C.x is injective and f is injective, so their composition is injective
          exact Function.Injective.comp C.hx f.injective
        hy := by -- C.y is injective and g is injective, so their composition is injective
          exact Function.Injective.comp C.hy g.injective
        hxy := by
          -- No row parameter equals any column parameter: follows from the original condition
          intro i j
          exact C.hxy (f i) (g j)
      }
    use C'
    -- The matrix entries match by definition
    ext i j
    rfl

end MDS
