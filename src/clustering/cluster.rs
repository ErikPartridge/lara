use ndarray::{Array1, Array2};

///
pub trait Cluster {
    ///
    fn fit(&mut self, xs: &Array2<f32>);
    ///
    fn predict(&self, xs: &Array2<f32>) -> Array1<usize>;
}
