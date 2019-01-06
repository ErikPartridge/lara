use crate::clustering::cluster::Cluster;
use ndarray::{Array, Array1, Array2, Axis};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::f32;

/// The initialization options for the centroids. Choices are `Random`.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum InitAlgorithm {
    /// Randomly select the initial centroids
    Random,
}

/// The struct representing a KMeans algorithm
#[derive(Debug, Clone, PartialEq)]
pub struct KMeans {
    centroids: Array2<f32>,
    max_iter: usize,
    tolerance: f64,
}

impl KMeans {
    /// ```
    /// use lara::clustering::cluster::Cluster;
    /// use lara::clustering::kmeans::{KMeans, InitAlgorithm};
    /// use ndarray::arr2;
    /// let xs = arr2(&[[0.23, 1.20, 9.300, 0.23, 1.20, 9.300],
    ///				 [-3.2, 1.8, 2.399, -3.2, 1.8, 2.398],
    ///				 [-1.0, 2.33, 0.22, -100.0, 9.0, 19.1],
    ///				 [0.235, -1.20, 9.300, -0.23, 11.20, 99.300],
    ///				 [-3.2, 10.8, 239.9, 3.2, 1.7, 2.0],
    ///				 [-1.0, 2959.2, 0.22, -102.0, 9.0, 19.1]]);
    /// let clf = KMeans::new(&xs, 100, 4, InitAlgorithm::Random);
    /// assert_eq!(clf.get_centroids().rows(), 4);
    /// ```
    pub fn new(
        xs: &Array2<f32>,
        max_iter: usize,
        num_clusters: usize,
        init_fn: InitAlgorithm,
    ) -> KMeans {
        match init_fn {
            InitAlgorithm::Random => {
                let mut rng = thread_rng();
                let range = Uniform::from(0..xs.rows());
                let mut initial_indexes = Vec::with_capacity(num_clusters);
                while initial_indexes.len() < num_clusters && initial_indexes.len() < xs.rows() {
                    let generated = range.sample(&mut rng);
                    if !initial_indexes.contains(&generated) {
                        initial_indexes.push(generated);
                    }
                }
                let centroids = xs.select(Axis(0), &initial_indexes);
                KMeans {
                    centroids,
                    max_iter,
                    tolerance: 0.001,
                }
            }
        }
    }

    fn distance(&self, x: &Array1<f32>, centroid: &Array1<f32>) -> f32 {
        let difference = x - centroid;
        difference.dot(&difference.t())
    }

    pub fn get_centroids(&self) -> &Array2<f32> {
        &self.centroids
    }
}

impl Cluster for KMeans {
    /// Fit the KMeans algorithm
    /// ```
    /// use lara::clustering::cluster::Cluster;
    /// use lara::clustering::kmeans::{KMeans, InitAlgorithm};
    /// use ndarray::arr2;
    /// let xs = arr2(&[[0.0, 1.0],
    ///				 [-1.0, 1.0],
    ///				 [-2.0, 0.0],
    ///				 [0.667, 1.2],
    ///              [0.7, 3.0]]);
    /// let mut clf = KMeans::new(&xs, 2, 2, InitAlgorithm::Random);
    /// clf.fit(&xs);
    /// ```
    fn fit(&mut self, xs: &Array2<f32>) {
        for _ in 1..=self.max_iter {
            let classifications = self.predict(xs);
            let classifications_vec = classifications.to_vec();
            let mut centroids_vec: Vec<f32> = vec![];
            for i in 0..self.centroids.rows() {
                let matched: Vec<usize> = xs
                    .genrows()
                    .into_iter()
                    .enumerate()
                    .filter(|(index, _)| classifications_vec[*index] == i)
                    .map(|(index, _)| index)
                    .collect();
                let matched_rows: Vec<f32> =
                    xs.select(Axis(0), &matched).mean_axis(Axis(0)).to_vec();
                centroids_vec.extend(matched_rows);
            }
            self.centroids = Array::from_shape_vec(
                (self.centroids.rows(), self.centroids.cols()),
                centroids_vec,
            )
            .unwrap();
            let new_classifications = self.predict(xs);
            let mut misses = 0.0;
            for i in 0..xs.rows() {
                if classifications[i] != new_classifications[i] {
                    misses += 1.0;
                }
            }
            misses /= xs.rows() as f64;
            if misses < self.tolerance {
                break;
            }
        }
    }

    /// Fit the KMeans algorithm
    /// ```
    /// use lara::clustering::cluster::Cluster;
    /// use lara::clustering::kmeans::{KMeans, InitAlgorithm};
    /// use ndarray::arr2;
    /// let xs = arr2(&[[0.0, 1.0],
    ///				 [-1.0, 1.0],
    ///				 [-2.0, 0.0],
    ///				 [0.667, 1.2],
    ///              [0.7, 3.0]]);
    /// let mut clf = KMeans::new(&xs, 2, 2, InitAlgorithm::Random);
    /// clf.fit(&xs);
    /// assert_eq!(clf.predict(&arr2(&[[0.0002, 1.1111]]))[0], clf.predict(&arr2(&[[0.004, 1.121]]))[0]);
    /// assert!(clf.predict(&arr2(&[[0.7, 3.0]]))[0] != clf.predict(&arr2(&[[-2.0, -1.0]]))[0])
    /// ```
    fn predict(&self, xs: &Array2<f32>) -> Array1<usize> {
        Array::from_iter(xs.genrows().into_iter().map(|r| {
            let row = r.to_owned();
            let mut best = 0;
            let mut best_score = f32::INFINITY;
            for (index, centroid) in self.centroids.genrows().into_iter().enumerate() {
                if self.distance(&row, &centroid.to_owned()) < best_score {
                    best = index;
                    best_score = self.distance(&row, &centroid.to_owned());
                }
            }
            best
        }))
    }
}
