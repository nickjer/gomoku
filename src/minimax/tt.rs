use std::sync::LazyLock;

use crate::position_id::PositionId;

use super::score::Score;

// ── Zobrist keys ──────────────────────────────────────────────────────────────

/// Random u64 per `(position, stone)`. Indexed as `[position_index][usize::from(stone)]`.
pub static ZOBRIST: LazyLock<[[u64; 2]; PositionId::COUNT]> = LazyLock::new(|| {
    let mut rng = fastrand::Rng::with_seed(0xDEAD_BEEF_CAFE_1234);
    std::array::from_fn(|_| [rng.u64(..), rng.u64(..)])
});

// ── Bound ─────────────────────────────────────────────────────────────────────

/// Indicates how the stored score relates to the true minimax value.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Bound {
    Exact = 0,
    Lower = 1, // fail-high: true score >= stored score
    Upper = 2, // fail-low:  true score <= stored score
}

// ── Entry ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct TtEntry {
    key: u32,             // top 32 bits of Zobrist hash (collision guard)
    depth: u8,            // remaining plies when this entry was stored
    bound: Option<Bound>, // None = unoccupied; 1 byte via niche optimisation
    score: Score,
}

const EMPTY_ENTRY: TtEntry = TtEntry {
    key: 0,
    depth: 0,
    bound: None,
    score: Score::DRAW,
};

// ── Table ─────────────────────────────────────────────────────────────────────

const TT_SIZE: usize = 1 << 20; // 1 048 576 entries ≈ 12 MB

/// u64 mask for the low `TT_SIZE.ilog2()` bits. A u64 literal avoids any `as`
/// cast when masking a u64 hash before converting the (now-small) result to usize.
const TT_MASK_U64: u64 = (1u64 << 20) - 1;

/// Fixed-size transposition table using Zobrist hashing with replace-always eviction.
pub struct TranspositionTable {
    entries: Vec<TtEntry>,
}

impl TranspositionTable {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: vec![EMPTY_ENTRY; TT_SIZE],
        }
    }

    /// Returns `(score, bound)` if a usable entry exists for `hash` at `depth`.
    #[must_use]
    pub fn probe(&self, hash: u64, depth: u32) -> Option<(Score, Bound)> {
        let idx = usize::try_from(hash & TT_MASK_U64).expect("TT-masked hash fits in usize");
        let entry = &self.entries[idx];
        let bound = entry.bound?;
        if entry.key != u32::try_from(hash >> 32).expect("top 32 bits fit in u32") {
            return None;
        }
        if entry.depth < u8::try_from(depth).expect("depth fits in u8") {
            return None;
        }
        Some((entry.score, bound))
    }

    /// Stores a result, always replacing the existing entry (replace-always policy).
    pub fn store(&mut self, hash: u64, depth: u32, score: Score, bound: Bound) {
        let idx = usize::try_from(hash & TT_MASK_U64).expect("TT-masked hash fits in usize");
        self.entries[idx] = TtEntry {
            key: u32::try_from(hash >> 32).expect("top 32 bits fit in u32"),
            depth: u8::try_from(depth).expect("depth fits in u8"),
            bound: Some(bound),
            score,
        };
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two hashes that land in the same table slot but have different keys (top 32 bits differ).
    const HASH_A: u64 = 0x0000_0001_0000_0042;
    const HASH_B: u64 = 0x0000_0002_0000_0042;

    #[test]
    fn empty_table_returns_none() {
        let tt = TranspositionTable::new();
        assert!(tt.probe(HASH_A, 4).is_none());
    }

    #[test]
    fn exact_bound_round_trips() {
        let mut tt = TranspositionTable::new();
        tt.store(HASH_A, 4, Score::new(300), Bound::Exact);
        let (score, bound) = tt.probe(HASH_A, 4).expect("should hit");
        assert_eq!(score, Score::new(300));
        assert_eq!(bound, Bound::Exact);
    }

    #[test]
    fn lower_bound_round_trips() {
        let mut tt = TranspositionTable::new();
        tt.store(HASH_A, 3, Score::new(1_500), Bound::Lower);
        let (score, bound) = tt.probe(HASH_A, 3).expect("should hit");
        assert_eq!(score, Score::new(1_500));
        assert_eq!(bound, Bound::Lower);
    }

    #[test]
    fn upper_bound_round_trips() {
        let mut tt = TranspositionTable::new();
        tt.store(HASH_A, 2, Score::new(20), Bound::Upper);
        let (score, bound) = tt.probe(HASH_A, 2).expect("should hit");
        assert_eq!(score, Score::new(20));
        assert_eq!(bound, Bound::Upper);
    }

    #[test]
    fn negative_score_round_trips() {
        let mut tt = TranspositionTable::new();
        tt.store(HASH_A, 1, -Score::new(50_000), Bound::Exact);
        let (score, _) = tt.probe(HASH_A, 1).expect("should hit");
        assert_eq!(score, -Score::new(50_000));
    }

    #[test]
    fn probe_succeeds_at_stored_depth() {
        let mut tt = TranspositionTable::new();
        tt.store(HASH_A, 5, Score::DRAW, Bound::Exact);
        assert!(tt.probe(HASH_A, 5).is_some());
    }

    #[test]
    fn probe_succeeds_at_shallower_depth_than_stored() {
        let mut tt = TranspositionTable::new();
        tt.store(HASH_A, 5, Score::DRAW, Bound::Exact);
        assert!(tt.probe(HASH_A, 3).is_some());
    }

    #[test]
    fn probe_fails_when_requested_depth_exceeds_stored() {
        let mut tt = TranspositionTable::new();
        tt.store(HASH_A, 3, Score::DRAW, Bound::Exact);
        assert!(tt.probe(HASH_A, 4).is_none());
    }

    #[test]
    fn probe_fails_on_key_mismatch_same_slot() {
        // HASH_A and HASH_B share the same low 20 bits (same slot) but differ in
        // the top 32 bits (different keys), so storing one must not satisfy a probe
        // for the other.
        let mut tt = TranspositionTable::new();
        tt.store(HASH_A, 4, Score::new(300), Bound::Exact);
        assert!(tt.probe(HASH_B, 4).is_none());
    }

    #[test]
    fn replace_always_overwrites_previous_entry() {
        let mut tt = TranspositionTable::new();
        tt.store(HASH_A, 4, Score::new(300), Bound::Lower);
        tt.store(HASH_A, 6, Score::new(50_000), Bound::Exact);
        let (score, bound) = tt.probe(HASH_A, 6).expect("should hit");
        assert_eq!(score, Score::new(50_000));
        assert_eq!(bound, Bound::Exact);
    }

    #[test]
    fn win_score_round_trips() {
        let mut tt = TranspositionTable::new();
        let win = Score::win_at_depth(9);
        tt.store(HASH_A, 7, win, Bound::Exact);
        let (score, _) = tt.probe(HASH_A, 7).expect("should hit");
        assert_eq!(score, win);
    }
}
