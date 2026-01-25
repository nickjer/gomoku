use crate::gene::Gene;

/// Performs inversion mutation: reverses a random segment of the sequence.
#[must_use]
pub fn inversion_mutate(genes: &[Gene], rng: &mut fastrand::Rng) -> Vec<Gene> {
    mutate_internal(genes, rng).0
}

fn mutate_internal(genes: &[Gene], rng: &mut fastrand::Rng) -> (Vec<Gene>, (usize, usize)) {
    assert!(genes.len() >= 2, "inversion requires at least 2 genes");

    let a = rng.usize(..genes.len());
    let b = rng.usize(..genes.len() - 1);
    let b = if b >= a { b + 1 } else { b };

    let (start, end) = if a < b { (a, b) } else { (b, a) };

    let mut child = genes.to_vec();
    child[start..=end].reverse();

    (child, (start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn genes(values: &[usize]) -> Vec<Gene> {
        values.iter().copied().map(Gene::new).collect()
    }

    #[test]
    fn reverses_segment_and_preserves_rest() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let (child, (start, end)) = mutate_internal(&g, &mut rng);

            // Elements before segment unchanged
            for i in 0..start {
                assert_eq!(child[i], g[i]);
            }

            // Segment is reversed
            for i in start..=end {
                assert_eq!(child[i], g[end - (i - start)]);
            }

            // Elements after segment unchanged
            for i in (end + 1)..g.len() {
                assert_eq!(child[i], g[i]);
            }
        }
    }

    #[test]
    fn preserves_length() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, _) = mutate_internal(&g, &mut rng);

        assert_eq!(child.len(), g.len());
    }

    #[test]
    fn start_less_than_end() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let (_, (start, end)) = mutate_internal(&g, &mut rng);
            assert!(start < end);
        }
    }

    #[test]
    #[should_panic(expected = "inversion requires at least 2 genes")]
    fn panics_with_single_element() {
        let g = genes(&[0]);
        let mut rng = fastrand::Rng::with_seed(42);

        mutate_internal(&g, &mut rng);
    }

    #[test]
    fn preserves_all_elements() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (mut child, _) = mutate_internal(&g, &mut rng);
        child.sort();

        assert_eq!(child, genes(&[0, 1, 2, 3, 4]));
    }
}
