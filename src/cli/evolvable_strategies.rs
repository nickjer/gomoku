use serde::{Deserialize, Serialize};

use crate::cluster::strategy::{NN1, NN2, NN3, NN4};

/// A homogeneous collection of strategies that can be evolved together.
#[derive(Debug, Serialize, Deserialize)]
pub enum EvolvableStrategies {
    Nn1 { strategies: Vec<NN1> },
    Nn2 { strategies: Vec<NN2> },
    Nn3 { strategies: Vec<NN3> },
    Nn4 { strategies: Vec<NN4> },
}
