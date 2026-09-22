mod control;
mod factorization;

pub use control::{UmfpackControl, UmfpackInfo};
pub use factorization::{UmfpackError, UmfpackLU};

#[cfg(test)]
mod tests {
    use crate::sparse::{CscMatrix, SparseError, SparseTriplet};

    use super::*;

    #[test]
    fn factorizes_and_solves_sparse_matrix() {
        let n = 5;
        let mut triplet = SparseTriplet::new();
        let entries = [
            (0, 0, 2.),
            (0, 1, 3.),
            (1, 0, 3.),
            (1, 2, 4.),
            (1, 4, 6.),
            (2, 1, -1.),
            (2, 2, -3.),
            (2, 3, 2.),
            (3, 2, 1.),
            (4, 1, 4.),
            (4, 2, 2.),
            (4, 4, 1.),
        ];
        for (row, col, value) in entries {
            triplet.add(row, col, value);
        }

        let mut lu = UmfpackLU::new();
        let control = UmfpackControl::new();
        let mut info = UmfpackInfo::default();
        lu.factorize(n, &triplet, &control, &mut info).unwrap();

        let rhs = [8., 45., -3., 3., 19.];
        let mut solution = [0.0; 5];
        lu.solve(&rhs, &mut solution, &control, &mut info).unwrap();
        let expected = [1., 2., 3., 4., 5.];
        for (actual, wanted) in solution.iter().zip(expected.iter()) {
            assert!((actual - wanted).abs() < 1e-9);
        }

        assert_eq!(
            lu.solve(&rhs[..4], &mut solution, &control, &mut info),
            Err(UmfpackError::InvalidVectorLength {
                vector: "rhs",
                expected: 5,
                actual: 4,
            })
        );
        lu.clear();
        assert_eq!(
            lu.solve(&rhs, &mut solution, &control, &mut info),
            Err(UmfpackError::NotFactorized)
        );
        lu.factorize(n, &triplet, &control, &mut info).unwrap();
        lu.solve(&rhs, &mut solution, &control, &mut info).unwrap();
    }

    #[test]
    fn validates_indices_and_coalesces_duplicates() {
        let mut triplet = SparseTriplet::new();
        triplet.add(0, 0, 1.0);
        triplet.add(0, 0, 2.0);
        let matrix = CscMatrix::from_triplet(2, &triplet).unwrap();
        assert_eq!(matrix.nonzeros(), 1);
        assert_eq!(matrix.values(), &[3.0]);

        triplet.add(2, 0, 1.0);
        assert_eq!(
            CscMatrix::from_triplet(2, &triplet).unwrap_err(),
            SparseError::IndexOutOfBounds {
                entry: 2,
                row: 2,
                col: 0,
                n: 2,
            }
        );
        assert_eq!(
            CscMatrix::from_triplet(0, &SparseTriplet::new()).unwrap_err(),
            SparseError::EmptyMatrix
        );
    }
}
