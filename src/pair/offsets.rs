use crate::offset::Offset;

/// Orthogonal neighbors (up, down, left, right)
pub const NN1: [Offset; 4] = [
    Offset::new(-1, 0),
    Offset::new(1, 0),
    Offset::new(0, -1),
    Offset::new(0, 1),
];

/// Diagonal neighbors
pub const NN2: [Offset; 4] = [
    Offset::new(-1, -1),
    Offset::new(-1, 1),
    Offset::new(1, -1),
    Offset::new(1, 1),
];

/// Extended orthogonal neighbors (distance 2)
pub const NN3: [Offset; 4] = [
    Offset::new(-2, 0),
    Offset::new(2, 0),
    Offset::new(0, -2),
    Offset::new(0, 2),
];

/// Knight-move neighbors
pub const NN4: [Offset; 8] = [
    Offset::new(-2, -1),
    Offset::new(-2, 1),
    Offset::new(-1, -2),
    Offset::new(-1, 2),
    Offset::new(1, -2),
    Offset::new(1, 2),
    Offset::new(2, -1),
    Offset::new(2, 1),
];
