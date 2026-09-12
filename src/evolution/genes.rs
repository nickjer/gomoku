/// Genes the genetic algorithm can evolve: a fixed shape holding numbers.
///
/// The numbers come in groups that belong together, always in the same
/// order. Operators change the numbers inside the groups but never the shape.
pub trait EvolvableGenes: Clone {
    /// The groups of numbers, in a fixed order.
    fn gene_groups(&self) -> impl Iterator<Item = &[f32]>;

    /// The same groups in the same order, for changing in place.
    fn gene_groups_mut(&mut self) -> impl Iterator<Item = &mut [f32]>;
}
