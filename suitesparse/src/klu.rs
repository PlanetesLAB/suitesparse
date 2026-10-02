mod common;
mod factorization;

pub use common::KluCommon;
pub use factorization::{KluError, KluLU};

#[cfg(test)]
mod tests {
    use crate::sparse::SparseTriplet;

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

        let mut lu = KluLU::new();
        let mut common = KluCommon::default();
        lu.factorize(n, &triplet, &mut common).unwrap();

        let rhs = [8., 45., -3., 3., 19.];
        let mut solution = [0.0; 5];
        lu.solve(&rhs, &mut solution, &mut common).unwrap();
        let expected = [1., 2., 3., 4., 5.];
        for (actual, wanted) in solution.iter().zip(expected.iter()) {
            assert!((actual - wanted).abs() < 1e-9);
        }

        assert_eq!(
            lu.solve(&rhs[..4], &mut solution, &mut common),
            Err(KluError::InvalidVectorLength {
                vector: "rhs",
                expected: 5,
                actual: 4,
            })
        );
        lu.clear();
        assert_eq!(
            lu.solve(&rhs, &mut solution, &mut common),
            Err(KluError::NotFactorized)
        );
        lu.factorize(n, &triplet, &mut common).unwrap();
        lu.solve(&rhs, &mut solution, &mut common).unwrap();
    }

    #[test]
    fn reports_singular_matrix() {
        let mut triplet = SparseTriplet::new();
        triplet.add(0, 0, 1.0);
        triplet.add(1, 0, 1.0);
        let mut lu = KluLU::new();
        let mut common = KluCommon::default();
        assert_eq!(
            lu.factorize(2, &triplet, &mut common),
            Err(KluError::KluStatus {
                operation: "numeric factorization",
                code: suitesparse_sys::KLU_SINGULAR as i32,
            })
        );
    }
}
