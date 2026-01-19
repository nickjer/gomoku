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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::{EvolvableStrategy, Strategy};

    #[test]
    fn serialization_round_trip_nn1() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = vec![
            NN1::random("test1", &mut rng),
            NN1::random("test2", &mut rng),
        ];
        let original = EvolvableStrategies::Nn1 { strategies };

        let serialized = ron::to_string(&original).unwrap();
        let deserialized: EvolvableStrategies = ron::from_str(&serialized).unwrap();

        match deserialized {
            EvolvableStrategies::Nn1 { strategies } => {
                assert_eq!(strategies.len(), 2);
                assert_eq!(strategies[0].label(), "test1");
                assert_eq!(strategies[1].label(), "test2");
            }
            _ => panic!("Expected Nn1 variant"),
        }
    }

    #[test]
    fn serialization_round_trip_nn2() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = vec![NN2::random("test", &mut rng)];
        let original = EvolvableStrategies::Nn2 { strategies };

        let serialized = ron::to_string(&original).unwrap();
        let deserialized: EvolvableStrategies = ron::from_str(&serialized).unwrap();

        match deserialized {
            EvolvableStrategies::Nn2 { strategies } => {
                assert_eq!(strategies.len(), 1);
                assert_eq!(strategies[0].label(), "test");
            }
            _ => panic!("Expected Nn2 variant"),
        }
    }

    #[test]
    fn serialization_round_trip_nn3() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = vec![NN3::random("test", &mut rng)];
        let original = EvolvableStrategies::Nn3 { strategies };

        let serialized = ron::to_string(&original).unwrap();
        let deserialized: EvolvableStrategies = ron::from_str(&serialized).unwrap();

        match deserialized {
            EvolvableStrategies::Nn3 { strategies } => {
                assert_eq!(strategies.len(), 1);
                assert_eq!(strategies[0].label(), "test");
            }
            _ => panic!("Expected Nn3 variant"),
        }
    }

    #[test]
    fn serialization_round_trip_nn4() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = vec![NN4::random("test", &mut rng)];
        let original = EvolvableStrategies::Nn4 { strategies };

        let serialized = ron::to_string(&original).unwrap();
        let deserialized: EvolvableStrategies = ron::from_str(&serialized).unwrap();

        match deserialized {
            EvolvableStrategies::Nn4 { strategies } => {
                assert_eq!(strategies.len(), 1);
                assert_eq!(strategies[0].label(), "test");
            }
            _ => panic!("Expected Nn4 variant"),
        }
    }
}
