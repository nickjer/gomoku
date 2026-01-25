use crate::gene::Gene;

/// Performs swap mutation: exchanges two random elements.
#[must_use]
pub fn swap_mutate(genes: &[Gene], rng: &mut fastrand::Rng) -> Vec<Gene> {
    mutate_internal(genes, rng).0
}

fn mutate_internal(genes: &[Gene], rng: &mut fastrand::Rng) -> (Vec<Gene>, (usize, usize)) {
    assert!(genes.len() >= 2, "swap requires at least 2 genes");

    let a = rng.usize(..genes.len());
    let b = rng.usize(..genes.len() - 1);
    let b = if b >= a { b + 1 } else { b };

    let mut child = genes.to_vec();
    child.swap(a, b);

    (child, (a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn genes(values: &[usize]) -> Vec<Gene> {
        values.iter().copied().map(Gene::new).collect()
    }

    #[test]
    fn swaps_two_elements() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, (a, b)) = mutate_internal(&g, &mut rng);

        assert_eq!(child[a], g[b]);
        assert_eq!(child[b], g[a]);
    }

    #[test]
    fn preserves_length() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, _) = mutate_internal(&g, &mut rng);

        assert_eq!(child.len(), g.len());
    }

    #[test]
    fn indices_are_different() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let (_, (a, b)) = mutate_internal(&g, &mut rng);
            assert_ne!(a, b);
        }
    }

    #[test]
    #[should_panic(expected = "swap requires at least 2 genes")]
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
